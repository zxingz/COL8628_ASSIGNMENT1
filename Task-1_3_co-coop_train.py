# %%
import os
import json
from collections import defaultdict
from datetime import datetime

import clip

import pandas as pd

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

# %%

# =============================================================================
# 1. Setup and Helper Functions
# =============================================================================
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Variables
base_folder = f"data{os.sep}orthonet"
data_folder = f"{base_folder}{os.sep}orthonet data{os.sep}orthonet data"
weights_folder = "weights"
results_folder = "results"
os.makedirs(weights_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)


def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates accuracy, F1-score, and AUC-ROC."""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Convert labels to numerical format for AUC-ROC
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

# =============================================================================
# 2. Dataset Definition
# =============================================================================
class OrthonetDataset(Dataset):
    """Custom dataset for the Orthonet data."""
    def __init__(self, csv_file, img_dir, preprocess):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filenames'])
        image = Image.open(img_name).convert('RGB')
        image = self.preprocess(image)
        label = self.data.iloc[idx]['labels']
        return image, label

# =============================================================================
# 3. CoOp Model Definitions (Original for Reference)
# =============================================================================
class CoOpPromptLearner(nn.Module):
    """Original CoOp prompt learner."""
    def __init__(self, clip_model, n_ctx=16, n_cls=12, tokenized_prompts=None, ctx_init=""):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        
        ctx_vectors = torch.empty(n_ctx, clip_model.ln_final.weight.shape[0])
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.ctx.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :]) # SOS token
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) # Class and EOS tokens
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

class CoOpCLIP(nn.Module):
    """Original CoOp wrapper around CLIP."""
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
    def forward(self, image):
        prompts = self.prompt_learner()
        image_features = self.clip_model.encode_image(image)
        
        x = prompts + self.clip_model.positional_embedding
        x = x.to(self.clip_model.dtype)
        
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        eot_indices = self.tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.clip_model.text_projection
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features @ text_features.t()
        return logits

# =============================================================================
# 4. CoCoOp Model Definitions (NEW IMPLEMENTATION)
# =============================================================================
class CoCoOpPromptLearner(nn.Module):
    """
    CoCoOp Prompt Learner. Conditions the prompt on image features.
    """
    def __init__(self, clip_model, n_ctx=16, n_cls=12, tokenized_prompts=None):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        
        clip_im_dim = clip_model.visual.output_dim
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Base learnable context vectors (like CoOp)
        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors) # self.ctx is float32 by default
        
        # Meta-network to generate instance-specific token from image features
        self.meta_net = nn.Sequential(
            nn.Linear(clip_im_dim, clip_im_dim // 16),
            nn.ReLU(),
            nn.Linear(clip_im_dim // 16, ctx_dim)
        ) # self.meta_net weights are float32 by default
        
        # Pre-computed embeddings for prompt components
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.ctx.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

    def forward(self, image_features):
        """
        Generates dynamic prompts based on image features.
        """
        # image_features are float16 from CLIP's encoder
        
        # === FIX STARTS HERE ===
        # Cast image_features to float32 to match the meta_net's weights.
        # We use self.ctx.dtype as a robust way to get the target type (float32).
        image_features_float32 = image_features.type(self.ctx.dtype)
        delta = self.meta_net(image_features_float32) # shape: (batch_size, ctx_dim)
        # === FIX ENDS HERE ===
        
        # Add delta to base context vectors
        ctx_shifted = self.ctx.unsqueeze(0) + delta.unsqueeze(1)
        # Broadcasting: (1, n_ctx, dim) + (b, 1, dim) -> (b, n_ctx, dim)
        
        b = image_features.shape[0]
        c = self.n_cls
        
        # Expand components for batch-wise concatenation
        prefix = self.token_prefix.unsqueeze(0).expand(b, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(b, -1, -1, -1)
        ctx_shifted_expanded = ctx_shifted.unsqueeze(1).expand(-1, c, -1, -1)
        
        prompts = torch.cat([prefix, ctx_shifted_expanded, suffix], dim=2)
        return prompts

class CoCoOpCLIP(nn.Module):
    """
    CoCoOp wrapper around CLIP, using the CoCoOpPromptLearner.
    """
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
    def forward(self, image):
        # 1. Encode image. Output is likely float16.
        image_features = self.clip_model.encode_image(image) # (b, feat_dim)
        
        # 2. Generate dynamic prompts based on image features.
        # The prompt_learner now handles the dtype conversion internally.
        prompts = self.prompt_learner(image_features) # (b, c, prompt_len, embed_dim)
        
        b, c, prompt_len, dim = prompts.shape
        
        # 3. Encode text prompts
        prompts = prompts.view(b * c, prompt_len, dim)
        
        x = prompts + self.clip_model.positional_embedding
        # This line correctly casts the combined prompts back to the CLIP model's
        # expected dtype (e.g., float16) before passing to the transformer.
        x = x.to(self.clip_model.dtype)
        
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        eot_indices = self.tokenized_prompts.argmax(dim=-1).repeat(b)
        
        text_features_pre = x[torch.arange(x.shape[0]), eot_indices]
        text_features = text_features_pre @ self.clip_model.text_projection
        
        text_features = text_features.view(b, c, -1)
        
        # 4. Normalize features and calculate logits
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * torch.sum(image_features.unsqueeze(1) * text_features, dim=-1)
        
        return logits

# %%
# =============================================================================
# 5. Data Loading and Preprocessing
# =============================================================================

# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)
# Freeze CLIP weights
for param in model.parameters():
    param.requires_grad = False

# Dataset paths
train_csv = f"{base_folder}{os.sep}train.csv"
test_csv = f"{base_folder}{os.sep}test.csv"

# Create datasets
train_dataset = OrthonetDataset(train_csv, data_folder, preprocess)
test_dataset = OrthonetDataset(test_csv, data_folder, preprocess)

# Create label encoder for consistent label mapping
le = LabelEncoder()
unique_labels = sorted(list(set(train_dataset.data['labels'])))
le.fit(unique_labels)
num_classes = len(unique_labels)

# Fixed class-specific texts (prompts)
class_prompts = {
 "Hip_SmithAndNephew_Polarstem_NilCol": "A collarless, cementless femoral stem with a distinct tapered wedge shape. Its proximal section has a textured coating for bone ingrowth, while the distal section is smooth and polished.",
 "Knee_SmithAndNephew_GenesisII": "A total knee arthroplasty where the femoral component shows a characteristic J-shape on lateral X-rays. The space between the metal femoral and tibial components is a clear gap, representing the polyethylene liner.",
 "Hip_Stryker_Exeter": "A cemented, collarless femoral stem with a highly polished, smooth surface. It has a double-tapered shape and is surrounded by a visible mantle of bone cement, which appears as a grey layer between the implant and bone.",
 "Knee_Depuy_Synthes_Sigma": "A total knee arthroplasty. A posterior-stabilized (PS) version is identified by a central 'box' on the femoral component. A rotating platform (RP) version shows a distinct circular track on the tibial baseplate.",
 "Hip_DepuySynthes_Corail_Collar": "A cementless femoral stem with a flared proximal body and a tapered shape. Its key identifier is the medial collar resting on the femoral neck. The entire stem has a matte texture from its hydroxyapatite coating.",
 "Hip_DepuySynthes_Corail_NilCol": "A cementless, collarless femoral stem identical to the collared Corail. It features a flared proximal body, a tapered shape, and a matte texture from a full hydroxyapatite coating, but lacks a collar at the neck junction.",
 "Hip_SmithAndNephew_Anthology": "A cementless femoral stem with a pronounced proximal flare that is wider side-to-side than front-to-back. The stem tapers sharply to a thin distal tip, and the proximal section has a textured porous coating.",
 "Hip_JRIOrtho_FurlongEvolution_Collar": "A straight, tapered, cementless femoral stem with a full hydroxyapatite (HA) coating, giving it a matte appearance. Its defining feature is the prominent collar that rests on the cut surface of the femoral neck.",
 "Knee_SmithAndNephew_Legion2": "A modern total knee arthroplasty. It consists of a highly polished cobalt-chrome femoral component and a titanium tibial tray. The specific shape of the keel or stem extending into the tibia can be a distinguishing feature.",
 "Hip_Stryker_AccoladeII": "A cementless, tapered wedge femoral stem. Its most distinctive feature is a gentle medial curve or bow, designed to match the femur's natural anatomy. The proximal portion has a textured coating for bone ingrowth.",
 "Hip_JRIOrtho_FurlongEvolution_NilCol": "A straight, tapered, cementless femoral stem with a full hydroxyapatite (HA) coating, giving it a matte appearance. It is identical to its counterpart but is distinguished by the complete absence of a collar.",
 "Knee_ZimmerBiomet_Oxford": "A unicompartmental knee arthroplasty (UKA). Implants are seen on only one side of the knee joint. It consists of a small, C-shaped femoral runner and a small metal tibial tray, making it much smaller than a TKA."
}
prompts = [class_prompts[name] for name in unique_labels]
n_ctx = 16 # Number of context tokens
prompts_with_ctx = [ " ".join(["X"] * n_ctx) + " " + p for p in prompts]
tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_with_ctx]).to(device)

def process_labels(labels):
    """Encodes labels into numerical indices."""
    return torch.tensor(le.transform(list(labels)), device=device, dtype=torch.long)

# %%
# =============================================================================
# 6. CoCoOp Training Pipeline
# =============================================================================
def train_cocoop():
    """Trains the CoCoOp model."""
    print("\n--- Starting CoCoOp Training ---")
    
    # Initialize models
    prompt_learner = CoCoOpPromptLearner(
        clip_model=model,
        n_cls=num_classes,
        n_ctx=n_ctx,
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    cocoop_clip = CoCoOpCLIP(model, prompt_learner, tokenized_prompts).to(device)
    # Load pre-trained weights if the file exists
    model_weights_path = f'weights{os.sep}task_1_3_best_cocoop_model.pth'
    if os.path.exists(model_weights_path):
        print(f"Loading existing weights from: {model_weights_path}")
        cocoop_clip.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Weights loaded successfully. Resuming training...")
    else:
        print("No existing weights found. Starting training from scratch.")
    
    # Training settings
    optimizer = optim.AdamW(prompt_learner.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping settings
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    history = defaultdict(list)
    
    for epoch in range(100):
        cocoop_clip.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/100') as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = process_labels(labels)
                
                optimizer.zero_grad()
                logits = cocoop_clip(images)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Early stopping and model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(cocoop_clip.state_dict(), f'{weights_folder}{os.sep}task_1_3_best_cocoop_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
        
    # Save training history
    history_path = f'{results_folder}{os.sep}task_1_3_cocoop_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")

# =============================================================================
# 7. CoCoOp Inference Pipeline
# =============================================================================
def evaluate_cocoop_model(model_path):
    """Loads a trained CoCoOp-CLIP model and evaluates it on the test set."""
    print("\n--- Running CoCoOp Evaluation ---")
    
    # Initialize a new model instance
    prompt_learner = CoCoOpPromptLearner(
        clip_model=model,
        n_cls=num_classes,
        n_ctx=n_ctx,
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    loaded_model = CoCoOpCLIP(model, prompt_learner, tokenized_prompts).to(device)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    all_labels = []
    all_preds = []
    all_pred_probas = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            logits = loaded_model(images)
            probas = logits.softmax(dim=-1)
            preds = probas.argmax(dim=-1)

            all_labels.extend(labels)
            all_preds.extend(le.inverse_transform(preds.cpu().numpy()))
            all_pred_probas.extend(probas.cpu().numpy())

    results = evaluate_metrics(all_labels, all_preds, all_pred_probas)
    return results

# %%
# =============================================================================
# 8. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # Train the CoCoOp model
    train_cocoop()
    
    # # --- INFERENCE AND EVALUATION ---
    # model_weights_path = f'{weights_folder}{os.sep}task_1_3_best_cocoop_model.pth'
    
    # if os.path.exists(model_weights_path):
    #     metrics = evaluate_cocoop_model(model_weights_path)
        
    #     print("\n--- CoCoOp Evaluation Results ---")
    #     print(f"  Accuracy: {metrics['accuracy']:.4f}")
    #     print(f"  F1-Score: {metrics['f1_score']:.4f}")
    #     print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")
        
    #     # Save Results to JSON
    #     results_path = f"{results_folder}{os.sep}task-1_3_cocoop_results.json"
    #     with open(results_path, "w") as file:
    #         json.dump({
    #             "Top-1 Accuracy": f"{metrics['accuracy']:.4f}",
    #             "F1-Score": f"{metrics['f1_score']:.4f}",
    #             "AUC-ROC": f"{metrics['auc_roc']:.4f}"
    #         }, file, indent=4)
    #     print(f"\nResults saved to {results_path}")
        
    # else:
    #     print(f"Model weights not found at '{model_weights_path}'.")
    #     print("Please train the model first by running this script.")