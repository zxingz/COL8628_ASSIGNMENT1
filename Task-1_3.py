# %%
import os
import json
from collections import defaultdict

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
        
        self.register_buffer("token_prefix", embedding)
        self.register_buffer("token_suffix", None)
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if suffix is None:
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                ], dim=1
            )
        else:
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ], dim=1
            )
        
        return prompts

class CoOpCLIP(nn.Module):
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        
    def forward(self, image):
        prompts = self.prompt_learner().long()
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features @ text_features.t()
        return logits

# %%
# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

# %%
# Dataset paths
train_csv = f"{base_folder}/train.csv"
test_csv = f"{base_folder}/train.csv"

# Create datasets
train_dataset = OrthonetDataset(train_csv, data_folder, preprocess)
test_dataset = OrthonetDataset(test_csv, data_folder, preprocess)

# Create label encoder for consistent label mapping
le = LabelEncoder()
unique_labels = sorted(list({label for label in train_dataset.data['labels']}))
le.fit(unique_labels)

# Fixed class-specific texts
prompts = [f"X-ray image of {name.lower()} implant" for name in unique_labels]
tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device).long()

def process_labels(labels):
    # Extract main categories
    main_labels = [label for label in labels]
    # Convert to numerical indices
    return torch.tensor(le.transform(main_labels), device=device)

def train_coop():
    
    # Initialize models
    prompt_learner = CoOpPromptLearner(clip_model=model, \
                                        n_cls=len(unique_labels), \
                                        tokenized_prompts=tokenized_prompts).to(device)
    coop_clip = CoOpCLIP(model, prompt_learner).to(device)
    
    # Training settings
    optimizer = optim.AdamW(prompt_learner.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping settings
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    history = defaultdict(list)
    
    for epoch in range(100):
        coop_clip.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/100') as pbar:
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
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')
    
    return coop_clip, history


# %%
if __name__ == "__main__":
    
    # Train CoOp
    coop_model, history = train_coop()

# %%
