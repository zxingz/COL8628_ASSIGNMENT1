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

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Variables
base_folder = f"data{os.sep}orthonet"
data_folder = f"{base_folder}{os.sep}orthonet data{os.sep}orthonet data"
weights_folder = "weights"
os.makedirs(weights_folder, exist_ok=True)


def evaluate_metrics(y_true, y_pred, y_pred_proba):
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

# Classes Definitions
class OrthonetDataset(Dataset):
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
    
    
class CoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx=16, n_cls=12, tokenized_prompts=None, ctx_init=""):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        
        ctx_vectors = torch.empty(n_ctx, clip_model.ln_final.weight.shape[0])
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Embedd Prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.ctx.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :]) # SOS token
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) # Class token and EOS token
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )
        
        return prompts

class CoOpCLIP(nn.Module):
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        # Store tokenized prompts to find the End-of-Text (EOT) token position
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
    def forward(self, image):
        # Get the custom prompt embeddings from the learner (these are float32)
        prompts = self.prompt_learner()
        
        # Encode the image as usual
        image_features = self.clip_model.encode_image(image)
        
        # --- Corrected Text Encoding ---
        # Manually pass embeddings through the rest of the text encoder
        x = prompts + self.clip_model.positional_embedding
        
        # FIX: Cast the combined tensor to the CLIP model's expected dtype (e.g., float16)
        x = x.to(self.clip_model.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)

        # Take features from the EOT token
        eot_indices = self.tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.clip_model.text_projection
        
        # Normalize features for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate logits
        logits = image_features @ text_features.t()
        return logits

# %%
# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

# %%
# Dataset paths
train_csv = f"{base_folder}/train.csv"
test_csv = f"{base_folder}/test.csv"

# Create datasets
train_dataset = OrthonetDataset(train_csv, data_folder, preprocess)
test_dataset = OrthonetDataset(test_csv, data_folder, preprocess)

# Create label encoder for consistent label mapping
le = LabelEncoder()
unique_labels = sorted(list(set(train_dataset.data['labels'])))
le.fit(unique_labels)

# Fixed class-specific texts
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
tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

def process_labels(labels):
    # Extract main categories
    main_labels = [label for label in labels]
    # Convert to numerical indices
    return torch.tensor(le.transform(main_labels), device=device, dtype=torch.long)

def train_coop():
    
    # Initialize models
    prompt_learner = CoOpPromptLearner(clip_model=model, \
                                         n_cls=len(unique_labels), \
                                         tokenized_prompts=tokenized_prompts).to(device)
    coop_clip = CoOpCLIP(model, prompt_learner, tokenized_prompts).to(device)
    
    # Load pre-trained weights if the file exists
    model_weights_path = f'weights{os.sep}task_1_3_best_coop_model.pth'
    if os.path.exists(model_weights_path):
        print(f"Loading existing weights from: {model_weights_path}")
        coop_clip.load_state_dict(torch.load(model_weights_path, map_location=device))
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    history = defaultdict(list)
    
    for epoch in range(10):
        coop_clip.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/10') as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = process_labels(labels)
                
                optimizer.zero_grad()
                logits = coop_clip(images)
                loss = criterion(logits, labels)
                
                loss.backward()
                clip_grad_norm_(prompt_learner.parameters(), 0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            torch.save(coop_clip.state_dict(), f'weights{os.sep}task_1_3_best_coop_model.pth')
            
            # Save training history
            with open(f'results{os.sep}task_1_3_coop_training_history_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json', 'w') as f:
                json.dump(history, f)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
    
    return coop_clip, history

def evaluate_coop_model(model_path):
    """
    Loads a trained CoOp-CLIP model, runs inference on the test set,
    and returns the evaluation metrics.
    """
    # Initialize a new model instance to load the weights into
    prompt_learner = CoOpPromptLearner(
        clip_model=model,
        n_cls=len(unique_labels),
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    loaded_model = CoOpCLIP(model, prompt_learner, tokenized_prompts).to(device)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval() # Set the model to evaluation mode

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    all_labels = []
    all_preds = []
    all_pred_probas = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            
            # Get model predictions (logits)
            logits = loaded_model(images)
            
            # Convert logits to probabilities using softmax
            probas = logits.softmax(dim=-1)
            
            # Get the predicted class index
            preds = probas.argmax(dim=-1)

            # Append batch results to lists
            all_labels.extend(labels)
            all_preds.extend(le.inverse_transform(preds.cpu().numpy()))
            all_pred_probas.extend(probas.cpu().numpy())

    # Calculate and return the metrics
    results = evaluate_metrics(all_labels, all_preds, all_pred_probas)
    return results

# %%
if __name__ == "__main__":
    
    # Train CoOp
    coop_model, history = train_coop()
