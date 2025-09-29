# %%
import os
import json
from collections import defaultdict
from datetime import datetime
from glob import glob

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

# Model hyperparameters
n_ctx = 16  # Number of context tokens
n_deep = 3  # Number of deep prompt layers

# Variables
base_folder = f"data{os.sep}pacemakers"
train_data_folder = f"data{os.sep}pacemakers{os.sep}Train"  
test_data_folder = f"data{os.sep}pacemakers{os.sep}Test"
weights_folder = "weights"
results_folder = "results"
os.makedirs(weights_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)


def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates accuracy, F1-score, AUC-ROC, and Top-3 accuracy."""
    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Convert labels to numerical format for AUC-ROC
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
    
    # Calculate Top-3 accuracy
    top3_preds = np.argsort(y_pred_proba, axis=1)[:, -3:]  # Get indices of top 3 predictions
    y_true_array = np.array(y_true_encoded).reshape(-1, 1)
    top3_correct = np.any(top3_preds == y_true_array, axis=1)
    top3_accuracy = np.mean(top3_correct)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'top3_accuracy': top3_accuracy
    }

# =============================================================================
# 2. Dataset Definition
# =============================================================================
class PacemakerDataset(Dataset):
    def __init__(self, data, preprocess, label_map=None):
        self.data = data
        self.preprocess = preprocess
        
        if label_map is None:
            self.labels = sorted(self.data['labels'].unique())
            self.label_map = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label_map = label_map
            self.labels = sorted(list(self.label_map.keys()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filenames']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['labels']
        
        if self.preprocess:
            image = self.preprocess(image)
            
        return image, label

# =============================================================================
# 3. MaPLe Model Definitions (NEW IMPLEMENTATION)
# =============================================================================

class MaPLePromptLearner(nn.Module):
    """
    MaPLe Prompt Learner. Creates shallow and deep prompts for multi-level
    adaptation in both vision and text encoders.
    """
    def __init__(self, clip_model, n_ctx=16, n_cls=12, tokenized_prompts=None, n_deep=3):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.n_deep = n_deep
        
        # Get dimensions for both text and vision encoders
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.transformer.width

        # Shallow learnable context vectors (for text encoder, dim=512)
        ctx_vectors = torch.empty(n_ctx, text_ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Deep learnable prompts for the vision encoder (dim=768)
        self.deep_vision_prompts = nn.Parameter(torch.empty(n_deep, n_ctx, vision_ctx_dim))
        nn.init.normal_(self.deep_vision_prompts, std=0.02)
        
        # Pre-computed embeddings for prompt components
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.ctx.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

    def forward(self):
        """
        Returns the shallow and deep prompts.
        """
        return self.ctx, self.deep_vision_prompts


class MaPLeCLIP(nn.Module):
    """
    MaPLe wrapper around CLIP. Injects learnable prompts at multiple layers
    of both the vision and text transformers.
    """
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        # Get dimensions for the projector
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.transformer.width
        
        # Projector to create deep text prompts (512-dim) from deep vision prompts (768-dim)
        self.prompt_projector = nn.Sequential(
            nn.Linear(vision_ctx_dim, vision_ctx_dim // 16),
            nn.ReLU(),
            nn.Linear(vision_ctx_dim // 16, text_ctx_dim)
        )

    def encode_image(self, image, deep_vision_prompts):
        # Cast image to float32 before processing
        x = self.clip_model.visual.conv1(image.type(torch.float32))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        # Inject deep prompts into specified layers
        for i, resblock in enumerate(self.clip_model.visual.transformer.resblocks):
            if i < self.prompt_learner.n_deep:
                # Prepend deep prompts to the input sequence
                prompt = deep_vision_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                # Keep the [CLS] token at the beginning
                cls_token, patches = x[:1], x[1:]
                x = torch.cat([cls_token, prompt, patches], dim=0)
            
            x = resblock(x)
            
            # After processing, remove the prompt tokens to maintain sequence length integrity
            if i < self.prompt_learner.n_deep:
                 x = torch.cat([x[:1], x[1+self.prompt_learner.n_ctx:]], dim=0)

        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.ln_post(x[:, 0, :])
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj
        
        return x

    def encode_text(self, ctx, deep_text_prompts):
        """
        Encodes text prompts with shallow and deep prompt injection.
        """
        # Form the initial shallow prompt
        prefix = self.prompt_learner.token_prefix
        suffix = self.prompt_learner.token_suffix
        ctx_expanded = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx_expanded, suffix], dim=1)
        
        x = prompts + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)
        
        # Calculate new sequence length with deep prompts
        seq_len = x.shape[0]
        for i in range(self.prompt_learner.n_deep):
            seq_len += self.prompt_learner.n_ctx
        
        # Create new attention mask of correct size
        attn_mask = torch.empty(seq_len, seq_len, device=x.device)
        attn_mask.fill_(float("-inf"))
        attn_mask.triu_(1)  # Zero out the upper triangle

        # Inject deep prompts into specified layers
        for i, resblock in enumerate(self.clip_model.transformer.resblocks):
            if i < self.prompt_learner.n_deep:
                # Prepend deep prompts after the [SOS] token
                prompt = deep_text_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                sos_token, rest_of_tokens = x[:1], x[1:]
                x = torch.cat([sos_token, prompt, rest_of_tokens], dim=0)
                
                # Update the attention mask for this layer
                resblock.attn_mask = attn_mask[:x.shape[0], :x.shape[0]]
            
            x = resblock(x)

            # After processing, remove the prompt tokens
            if i < self.prompt_learner.n_deep:
                x = torch.cat([x[:1], x[1+self.prompt_learner.n_ctx:]], dim=0)

        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        
        eot_indices = self.tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.clip_model.text_projection
        return text_features

    def forward(self, image):
        # 1. Get shallow and deep prompts from the learner
        ctx, deep_vision_prompts = self.prompt_learner()

        # 2. Encode image with injected deep vision prompts
        image_features = self.encode_image(image, deep_vision_prompts)

        # 3. Project deep vision prompts to get deep text prompts
        deep_text_prompts = self.prompt_projector(deep_vision_prompts)

        # 4. Encode text with injected shallow and deep text prompts
        text_features = self.encode_text(ctx, deep_text_prompts)

        # 5. Normalize features and calculate logits
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features @ text_features.t()
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

# Create DataFrame for train data
train_pacemaker_data = pd.DataFrame()
for folder in glob(f"{train_data_folder}{os.sep}*"):
    label, version = os.path.basename(folder).split(' - ')
    for file in glob(f"{folder}{os.sep}*"):
        train_pacemaker_data = pd.concat([train_pacemaker_data, pd.DataFrame([{
            'filenames': file,
            'labels': f"({label})",
        }])], ignore_index=True)
train_pacemaker_data = train_pacemaker_data.reset_index(drop=True)

# Create DataFrame for test data
test_pacemaker_data = pd.DataFrame()
for folder in glob(f"{test_data_folder}{os.sep}*"):
    label, version = os.path.basename(folder).split(' - ')
    for file in glob(f"{folder}{os.sep}*"):
        test_pacemaker_data = pd.concat([test_pacemaker_data, pd.DataFrame([{
            'filenames': file,
            'labels': f"({label})",
        }])], ignore_index=True)
test_pacemaker_data = test_pacemaker_data.reset_index(drop=True)

# Create datasets
train_dataset = PacemakerDataset(train_pacemaker_data, preprocess)
test_dataset = PacemakerDataset(test_pacemaker_data, preprocess)

# Create label encoder for consistent label mapping
le = LabelEncoder()
unique_labels = sorted(list(set(train_dataset.data['labels'])))
le.fit(unique_labels)
num_classes = len(unique_labels)

# Fixed class-specific texts
class_prompts = {x: f"a photo of a pacemaker {x}" for x in unique_labels}
prompts = [class_prompts[name] for name in unique_labels]
n_ctx = 16
prompts_with_ctx = [" ".join(["X"] * n_ctx) + " " + p for p in prompts]
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
    prompt_learner = MaPLePromptLearner(
        clip_model=model,
        n_cls=num_classes,
        n_ctx=n_ctx,
        n_deep=n_deep,
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    maple_clip = MaPLeCLIP(model, prompt_learner, tokenized_prompts).to(device)
    
    # Ensure all components use float32
    maple_clip = maple_clip.float()
    prompt_learner = prompt_learner.float()
    
    # ========================= FIX =========================
    # Cast all learnable parameters (prompts and projector) to the CLIP model's native dtype.
    # This is typically torch.float16 on GPU, which resolves the mismatch.
    maple_clip.to(model.dtype)
    # =======================================================
    
    # Load pre-trained weights if the file exists
    model_weights_path = f'weights{os.sep}task_2_3_st1_best_maple_model.pth'
    if os.path.exists(model_weights_path):
        print(f"Loading existing weights from: {model_weights_path}")
        maple_clip.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Weights loaded successfully. Resuming training...")
    else:
        print("No existing weights found. Starting training from scratch.")
    
    # Training settings
    # Optimizer includes parameters from both the prompt learner and the projector
    optimizer = optim.AdamW(
        list(prompt_learner.parameters()) + list(maple_clip.prompt_projector.parameters()), 
        lr=0.002
    )
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping settings
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    history = defaultdict(list)
    
    for epoch in range(1):
        maple_clip.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/10') as pbar:
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
            torch.save(maple_clip.state_dict(), f'{weights_folder}{os.sep}task_2_3_st1_best_maple_model.pth')
            
            # Save training history
            history_path = f'{results_folder}{os.sep}task_2_3_st1_maple_training_history_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'
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
        n_deep=n_deep,
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    loaded_model = MaPLeCLIP(model, prompt_learner, tokenized_prompts).to(device)

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    
    # ========================= FIX =========================
    # Also cast the model to the correct dtype for evaluation.
    loaded_model.to(model.dtype)
    # =======================================================
    
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
# 7. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # # Train the MaPLe model
    # train_maple()
    
    # --- INFERENCE AND EVALUATION ---
    model_weights_path = f'{weights_folder}{os.sep}task_2_3_st1_best_maple_model.pth'
    
    if os.path.exists(model_weights_path):
        metrics = evaluate_maple_model(model_weights_path)
        
        # In the main execution block:
        print("\n--- MaPLe Evaluation Results ---")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:  {metrics['auc_roc']:.4f}")

        # Save Results to JSON
        results_path = f"{results_folder}{os.sep}task-2_3_st1_maple_results.json"
        with open(results_path, "w") as file:
            json.dump({
                "Top-1 Accuracy": f"{metrics['accuracy']:.4f}",
                "Top-3 Accuracy": f"{metrics['top3_accuracy']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}",
                "AUC-ROC": f"{metrics['auc_roc']:.4f}"
            }, file, indent=4)
        
    else:
        print(f"Model weights not found at '{model_weights_path}'.")
        print("Please train the model first by running this script.")