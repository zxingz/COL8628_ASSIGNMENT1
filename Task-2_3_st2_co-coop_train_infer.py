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

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Configuration ---
# Paths for the original, pre-trained model and data
OLD_MODEL_PATH = f"weights{os.sep}task_2_3_st1_best_cocoop_model.pth"
OLD_TRAIN_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Train"

# Paths for the new model and the new, expanded dataset
# IMPORTANT: Update these paths to point to your new dataset with additional classes
NEW_TRAIN_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Train"
NEW_TEST_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Test"

# Output paths for the newly trained model and its results
NEW_MODEL_PATH = f"weights{os.sep}task_2_3_st2_best_cocoop_model.pth"
NEW_HISTORY_PATH_PREFIX = f"results{os.sep}task_2_3_st2_cocoop_training_history"
NEW_RESULTS_PATH = f"results{os.sep}task-2_3_st2_cocoop.json"

# Create necessary folders
os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)

N_CTX = 16 # Number of context tokens

# %%

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates and returns evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    if len(np.unique(y_true_encoded)) != y_pred_proba.shape[1]:
         print(f"Warning: Class mismatch. AUC may be incorrect.")
         auc_roc = 0.5 
    else:
        auc_roc = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc
    }

# Classes Definitions
class PacemakerDataset(Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filenames']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['labels']
        
        if self.preprocess:
            image = self.preprocess(image)
            
        return image, label


class CoCoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, n_cls, tokenized_prompts):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        
        clip_im_dim = clip_model.visual.output_dim
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.meta_net = nn.Sequential(
            nn.Linear(clip_im_dim, clip_im_dim // 16),
            nn.ReLU(),
            nn.Linear(clip_im_dim // 16, ctx_dim)
        )
        self.meta_net.to(clip_model.dtype)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

    def forward(self, image_features):
        delta = self.meta_net(image_features)
        ctx_shifted = self.ctx.unsqueeze(0) + delta.unsqueeze(1)
        
        b = image_features.shape[0]
        c = self.n_cls
        
        prefix = self.token_prefix.unsqueeze(0).expand(b, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(b, -1, -1, -1)
        ctx_shifted_expanded = ctx_shifted.unsqueeze(1).expand(-1, c, -1, -1)
        
        prompts = torch.cat([prefix, ctx_shifted_expanded, suffix], dim=2)
        return prompts

class CoCoOpCLIP(nn.Module):
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        prompts = self.prompt_learner(image_features)
        
        b, c, prompt_len, dim = prompts.shape
        prompts = prompts.view(b * c, prompt_len, dim)
        
        x = prompts + self.clip_model.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        eot_indices = self.tokenized_prompts.argmax(dim=-1).repeat(b)
        
        text_features_pre = x[torch.arange(x.shape[0]), eot_indices]
        text_features = text_features_pre @ self.clip_model.text_projection
        text_features = text_features.view(b, c, -1)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features.unsqueeze(1) * text_features
        return logits.sum(dim=-1)


def load_data_from_folder(folder_path):
    """Loads image paths and labels from a directory into a DataFrame."""
    pacemaker_data = pd.DataFrame()
    for folder in glob(f"{folder_path}{os.sep}*"):
        try:
            label, version = os.path.basename(folder).split(' - ')
        except ValueError:
            print(f"Skipping folder: {folder}")
            continue
        
        for file in glob(f"{folder}{os.sep}*"):
            pacemaker_data = pd.concat([pacemaker_data, pd.DataFrame([{'filenames': file, 'labels': f"({label}, {version})"}])], ignore_index=True)
    return pacemaker_data.reset_index(drop=True)


def train_cocoop(model, train_loader, label_encoder):
    """Trains the CoCoOp-CLIP model."""
    optimizer = optim.AdamW(model.prompt_learner.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    history = defaultdict(list)
    
    for epoch in range(1):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/50') as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = torch.tensor(label_encoder.transform(labels), dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), NEW_MODEL_PATH)
            print(f"Model saved to {NEW_MODEL_PATH} with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}.')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

    history_path = f'{NEW_HISTORY_PATH_PREFIX}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")


def evaluate_cocoop_model(model_path, test_dataset, label_encoder, unique_labels, tokenized_prompts, clip_model):
    """Evaluates the trained CoCoOp-CLIP model."""
    prompt_learner = CoCoOpPromptLearner(clip_model, n_ctx=N_CTX, n_cls=len(unique_labels), tokenized_prompts=tokenized_prompts).to(device)
    loaded_model = CoCoOpCLIP(clip_model, prompt_learner, tokenized_prompts).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    all_labels, all_preds, all_pred_probas = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            logits = loaded_model(images)
            probas = logits.softmax(dim=-1)
            preds = probas.argmax(dim=-1)

            all_labels.extend(labels)
            all_preds.extend(label_encoder.inverse_transform(preds.cpu().numpy()))
            all_pred_probas.extend(probas.cpu().numpy())

    return evaluate_metrics(all_labels, all_preds, np.array(all_pred_probas))

# %%
if __name__ == "__main__":
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    new_train_df = load_data_from_folder(NEW_TRAIN_DATA_FOLDER)
    new_unique_labels = sorted(new_train_df['labels'].unique())
    print(f"Found {len(new_unique_labels)} total classes in the new dataset.")

    new_prompts = [f"a photo of a pacemaker {name}" for name in new_unique_labels]
    prompts_with_ctx = [" ".join(["X"] * N_CTX) + " " + p for p in new_prompts]
    new_tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_with_ctx]).to(device)

    new_prompt_learner = CoCoOpPromptLearner(clip_model, n_ctx=N_CTX, n_cls=len(new_unique_labels), tokenized_prompts=new_tokenized_prompts).to(device)
    new_cocoop_model = CoCoOpCLIP(clip_model, new_prompt_learner, new_tokenized_prompts).to(device)

    if os.path.exists(OLD_MODEL_PATH):
        print(f"Loading weights from pre-trained model: {OLD_MODEL_PATH}")
        old_state_dict = torch.load(OLD_MODEL_PATH, map_location=device)
        
        # CoCoOp's prompt learner (ctx and meta_net) is class-agnostic.
        # We only transfer its learnable parameters and exclude the token buffers,
        # which are class-dependent and cause the size mismatch error.
        weights_to_load = {}
        for k, v in old_state_dict.items():
            if k.startswith("prompt_learner.") and "token" not in k:
                weights_to_load[k] = v
        
        # Load the filtered weights. `strict=False` handles any other potential mismatches.
        new_cocoop_model.load_state_dict(weights_to_load, strict=False)
        print("Successfully transferred weights from the pre-trained prompt learner.")
    else:
        print(f"Warning: No pre-trained model found at {OLD_MODEL_PATH}. Starting fresh.")

    print("\n" + "="*30 + "\nStarting Training on Expanded Dataset\n" + "="*30)
    
    new_test_df = load_data_from_folder(NEW_TEST_DATA_FOLDER)
    new_train_dataset = PacemakerDataset(new_train_df, preprocess)
    new_test_dataset = PacemakerDataset(new_test_df, preprocess)
    
    label_encoder = LabelEncoder().fit(new_unique_labels)
    train_loader = DataLoader(new_train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    train_cocoop(new_cocoop_model, train_loader, label_encoder)

    print("\n" + "="*30 + "\nRunning Final Evaluation\n" + "="*30)

    if os.path.exists(NEW_MODEL_PATH):
        metrics = evaluate_cocoop_model(NEW_MODEL_PATH, new_test_dataset, label_encoder, new_unique_labels, new_tokenized_prompts, clip_model)
        print(f"Evaluation Metrics: {metrics}")
        with open(NEW_RESULTS_PATH, "w") as f:
            json.dump({k: f"{v:.4f}" for k, v in metrics.items()}, f, indent=4)
        print(f"Results saved to {NEW_RESULTS_PATH}")
    else:
        print(f"Model weights not found at '{NEW_MODEL_PATH}'. Evaluation skipped.")

