# %%
import os
import json
from collections import defaultdict
from datetime import datetime
import copy

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


def evaluate_metrics(y_true, y_pred, y_pred_proba, label_encoder):
    """
    Calculates accuracy, F1-score, and AUC-ROC.
    Accepts a pre-fitted label_encoder to ensure consistent class mapping.
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Use the provided, globally-fitted label encoder to transform true labels
    y_true_encoded = label_encoder.transform(y_true)
    
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
# 3. MaPLe Model Definitions
# =============================================================================
class MaPLePromptLearner(nn.Module):
    """
    MaPLe Prompt Learner. Extends CoCoOp by adding deep, layer-specific prompts
    for both the vision and text encoders.
    """
    def __init__(self, clip_model, n_cls=12, n_ctx=2, deep_n_ctx=2, num_deep_layers=3):
        super().__init__()
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx # Number of context tokens for input-level text prompt
        self.deep_n_ctx = deep_n_ctx # Number of context tokens for deep prompts
        self.num_deep_layers = num_deep_layers # Number of transformer layers to inject prompts

        # --- FIX START ---
        # Get the correct dimensions from the CLIP model
        ctx_dim = clip_model.ln_final.weight.shape[0]              # Text feature dimension (512)
        vis_transformer_dim = clip_model.visual.transformer.width  # Vision transformer dimension (768)
        vis_output_dim = clip_model.visual.output_dim              # Vision final output dimension (512)
        # --- FIX END ---
        
        # --- Input-level Text Prompt (similar to CoCoOp) ---
        # Base learnable context vectors
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Meta-network for instance-specific conditioning
        self.meta_net = nn.Sequential(
            # --- FIX START ---
            # The meta_net takes the FINAL 512-dim image features as input
            nn.Linear(vis_output_dim, vis_output_dim // 16, dtype=clip_model.dtype),
            # --- FIX END ---
            nn.ReLU(),
            nn.Linear(vis_output_dim // 16, ctx_dim, dtype=clip_model.dtype)
        )
        
        # --- Deep Prompts (MaPLe's key addition) ---
        # Learnable prompts for the first N layers of the text encoder
        self.deep_text_prompts = nn.Parameter(
            torch.empty(self.num_deep_layers, self.deep_n_ctx, ctx_dim, dtype=clip_model.dtype)
        )
        nn.init.normal_(self.deep_text_prompts, std=0.02)
        
        # Learnable prompts for the first N layers of the vision encoder
        self.deep_vision_prompts = nn.Parameter(
            # --- FIX START ---
            # Deep vision prompts must have the same dimension as the transformer blocks (768)
            torch.empty(self.num_deep_layers, self.deep_n_ctx, vis_transformer_dim, dtype=clip_model.dtype)
            # --- FIX END ---
        )
        nn.init.normal_(self.deep_vision_prompts, std=0.02)

    def forward(self, image_features):
        """
        Generates the dynamic input-level prompt. Deep prompts are parameters
        and are accessed directly by the main MaPLeCLIP model.
        """
        # Generate the delta for the input-level text prompt
        delta = self.meta_net(image_features)
        
        # Add delta to the base context vectors
        ctx_shifted = self.ctx.unsqueeze(0) + delta.unsqueeze(1)
        # Broadcasting: (1, n_ctx, dim) + (b, 1, dim) -> (b, n_ctx, dim)
        
        return ctx_shifted

class MaPLeCLIP(nn.Module):
    """
    MaPLe wrapper around CLIP. Injects learnable prompts at multiple layers
    of both the vision and text encoders.
    """
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.tokenized_prompts = tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer # The text transformer part
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Unpack text encoder components
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
    def forward(self, image):
        # ================== 1. Vision Forward Pass with Deep Prompts ==================
        
        # Initial image encoding
        x = self.image_encoder.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        # --- FIX: Cast input to the correct dtype before the transformer ---
        x = x.to(self.dtype)
        
        x = x.permute(1, 0, 2) # (L, N, D)
        
        # Loop through vision transformer blocks and inject prompts
        for i, res_block in enumerate(self.image_encoder.transformer.resblocks):
            if i < self.prompt_learner.num_deep_layers:
                # Prepend deep vision prompt for this layer
                deep_prompt = self.prompt_learner.deep_vision_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x, deep_prompt], dim=0)
            
            x = res_block(x)
            
            if i < self.prompt_learner.num_deep_layers:
                # Remove the prompt tokens after the block
                x = x[:-self.prompt_learner.deep_n_ctx, :, :]

        x = x.permute(1, 0, 2)
        x = self.image_encoder.ln_post(x[:, 0, :])
        image_features = x @ self.image_encoder.proj
        
        # ================== 2. Text Forward Pass with Deep Prompts ===================
        
        # Get the instance-conditioned input-level prompt
        input_text_prompts = self.prompt_learner(image_features.detach())
        b, c, prompt_len, dim = input_text_prompts.shape[0], self.prompt_learner.n_cls, self.prompt_learner.n_ctx, input_text_prompts.shape[-1]
        
        # Prepare tokens
        token_prefix = self.token_embedding(self.tokenized_prompts[:, :1])
        token_suffix = self.token_embedding(self.tokenized_prompts[:, 1 + prompt_len:])
        
        # Create full prompts for each image in the batch
        prompts = []
        for i in range(b):
            # For each image, create prompts for all classes
            p_i = input_text_prompts[i].unsqueeze(0).expand(c, -1, -1)
            prefix_i = token_prefix.expand(b, -1, -1, -1)[0] # Use class-agnostic prefix
            suffix_i = token_suffix.expand(b, -1, -1, -1)[0]
            
            prompts_i = torch.cat([prefix_i, p_i, suffix_i], dim=1)
            prompts.append(prompts_i)
        
        x = torch.cat(prompts, dim=0) # (B * C, L, D)
        x = x + self.positional_embedding

        # --- FIX: Cast input to the correct dtype before the transformer ---
        x = x.to(self.dtype)

        x = x.permute(1, 0, 2) # (L, B * C, D)
        
        # Loop through text transformer blocks and inject prompts
        for i, res_block in enumerate(self.text_encoder.resblocks):
            if i < self.prompt_learner.num_deep_layers:
                # Prepend deep text prompt for this layer
                deep_prompt = self.prompt_learner.deep_text_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                x = torch.cat([x, deep_prompt], dim=0)
                
            # --- FIX START ---
            # Store the original mask to restore it later
            original_mask = res_block.attn_mask
            current_seq_len = x.shape[0]

            # If the sequence length is different from the mask's, create a new one
            if original_mask is None or original_mask.shape[0] != current_seq_len:
                # Create a new causal mask with the correct dimensions
                new_mask = torch.empty(current_seq_len, current_seq_len, device=x.device)
                new_mask.fill_(float("-inf"))
                new_mask.triu_(1) # Ensure tokens only attend to previous tokens
                res_block.attn_mask = new_mask.to(x.dtype)
            
            x = res_block(x)

            # Restore the original mask
            res_block.attn_mask = original_mask
            # --- FIX END ---
            
            if i < self.prompt_learner.num_deep_layers:
                # Remove the prompt tokens after the block
                x = x[:-self.prompt_learner.deep_n_ctx, :, :]
        
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        
        eot_indices = self.tokenized_prompts.argmax(dim=-1).repeat(b)
        text_features_pre = x[torch.arange(x.shape[0]), eot_indices]
        text_features = text_features_pre @ self.text_projection
        text_features = text_features.view(b, c, -1)
        
        # ====================== 3. Compute Logits =======================
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = self.logit_scale.exp() * torch.sum(image_features.unsqueeze(1) * text_features, dim=-1)
        
        return logits

# %%
# =============================================================================
# 4. Data Loading and Preprocessing
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
n_ctx = 2 # Number of context tokens for input-level prompts
prompts_with_ctx = [ " ".join(["X"] * n_ctx) + " " + p for p in prompts]
tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_with_ctx]).to(device)

def process_labels(labels):
    """Encodes labels into numerical indices."""
    return torch.tensor(le.transform(list(labels)), device=device, dtype=torch.long)

# %%
# =============================================================================
# 5. MaPLe Training Pipeline
# =============================================================================
def train_maple():
    """Trains the MaPLe model."""
    print("\n--- Starting MaPLe Training ---")
    
    # Initialize models
    # We use deep prompts for the first 3 layers of each transformer
    prompt_learner = MaPLePromptLearner(
        clip_model=model,
        n_cls=num_classes,
        n_ctx=n_ctx,
        deep_n_ctx=2,
        num_deep_layers=3
    ).to(device)
    
    maple_clip = MaPLeCLIP(model, prompt_learner, tokenized_prompts).to(device)
    
    # Training settings
    optimizer = optim.AdamW(prompt_learner.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping settings
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Reduced batch size for higher memory usage
    history = defaultdict(list)
    
    for epoch in range(100):
        maple_clip.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/100') as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = process_labels(labels)
                
                optimizer.zero_grad()
                logits = maple_clip(images)
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
            torch.save(maple_clip.state_dict(), f'{weights_folder}{os.sep}best_maple_model.pth')
            
            # Save training history
            history_path = f'{results_folder}{os.sep}maple_training_history_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'
            with open(history_path, 'w') as f:
                json.dump(history, f)
            print(f"Training history saved to {history_path}")
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

# =============================================================================
# 6. MaPLe Inference Pipeline
# =============================================================================
def evaluate_maple_model(model_path):
    """Loads a trained MaPLe-CLIP model and evaluates it on the test set."""
    print("\n--- Running MaPLe Evaluation ---")
    
    # Initialize a new model instance
    prompt_learner = MaPLePromptLearner(
        clip_model=model,
        n_cls=num_classes,
        n_ctx=n_ctx,
        deep_n_ctx=2,
        num_deep_layers=3
    ).to(device)
    
    loaded_model = MaPLeCLIP(model, prompt_learner, tokenized_prompts).to(device)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
            
            # Ensure probabilities are handled correctly for metrics
            probas_np = probas.cpu().numpy().astype(np.float64)
            all_pred_probas.extend(probas_np)

    results = evaluate_metrics(all_labels, all_preds, all_pred_probas, le)
    return results

# %%
# =============================================================================
# 7. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # Train the MaPLe model
    train_maple()
# %%