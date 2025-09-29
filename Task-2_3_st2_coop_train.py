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
OLD_MODEL_PATH = f"weights{os.sep}task_2_3_st1_best_coop_model.pth"
OLD_TRAIN_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Train"

# Paths for the new model and the new, expanded dataset
# IMPORTANT: Update these paths to point to your new dataset with additional classes
NEW_TRAIN_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Train"
NEW_TEST_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Test"

# Output paths for the newly trained model and its results
NEW_MODEL_PATH = f"weights{os.sep}task_2_3_st2_best_coop_model.pth"
NEW_HISTORY_PATH_PREFIX = f"results{os.sep}task_2_3_st2_coop_training_history"
NEW_RESULTS_PATH = f"results{os.sep}task-2_3_st2_coop.json"

# Create necessary folders
os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)


# %%

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates and returns evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    # Ensure all classes are present in y_pred_proba for roc_auc_score
    if len(np.unique(y_true_encoded)) != y_pred_proba.shape[1]:
         print(f"Warning: Number of classes in y_true ({len(np.unique(y_true_encoded))}) does not match y_pred_proba ({y_pred_proba.shape[1]}). AUC may be incorrect.")
         # Handle case where not all classes are predicted
         auc_roc = 0.5 # A neutral value
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
        self.labels = sorted(self.data['labels'].unique())
        self.label_map = {label: i for i, label in enumerate(self.labels)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filenames']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['labels']
        
        if self.preprocess:
            image = self.preprocess(image)
            
        return image, label


class CoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, n_cls, tokenized_prompts):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype  # Get CLIP model's dtype
        
        # Initialize context vectors with correct dtype
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Register token buffers
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts

class CoOpCLIP(nn.Module):
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.dtype = clip_model.dtype  # Store dtype
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
    def forward(self, image):
        prompts = self.prompt_learner()
        
        # Ensure image is in correct dtype
        image = image.to(self.dtype)
        image_features = self.clip_model.encode_image(image)
        
        # Ensure prompts use correct dtype
        prompts = prompts.to(self.dtype)
        x = prompts + self.clip_model.positional_embedding
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


def load_data_from_folder(folder_path, old_format=False):
    """Loads image paths and labels from a directory into a DataFrame."""
    pacemaker_data = pd.DataFrame()
    for folder in glob(f"{folder_path}{os.sep}*"):
        try:
            label, version = os.path.basename(folder).split(' - ')
        except ValueError:
            print(f"Skipping folder with unexpected name format: {folder}")
            continue
        
        for file in glob(f"{folder}{os.sep}*"):
            
            if old_format:
                pacemaker_data = pd.concat([pacemaker_data, pd.DataFrame([{
                    'filenames': file,
                    'labels': f"({label})",
                }])], ignore_index=True)
            else:
                pacemaker_data = pd.concat([pacemaker_data, pd.DataFrame([{
                    'filenames': file,
                    'labels': f"({label}, {version})",
                }])], ignore_index=True)
            
    return pacemaker_data.reset_index(drop=True)


def train_coop(model, train_loader, label_encoder):
    """Trains the CoOp-CLIP model."""
    optimizer = optim.AdamW(model.prompt_learner.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    history = defaultdict(list)
    
    for epoch in range(1): # Increased epochs for potentially more complex task
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
                clip_grad_norm_(model.prompt_learner.parameters(), 0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), NEW_MODEL_PATH)
            print(f"Model saved to {NEW_MODEL_PATH} with improved loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}.')
            break
            
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

    # Save final training history
    history_path = f'{NEW_HISTORY_PATH_PREFIX}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")

    return model, history


def evaluate_coop_model(model_path, test_dataset, label_encoder, unique_labels, tokenized_prompts):
    """Evaluates the trained CoOp-CLIP model on the test set."""
    # Initialize a model instance to load the state dict into
    prompt_learner = CoOpPromptLearner(
        clip_model=clip_model,
        n_ctx=16,
        n_cls=len(unique_labels),
        tokenized_prompts=tokenized_prompts
    ).to(device)
    
    loaded_model = CoOpCLIP(clip_model, prompt_learner, tokenized_prompts).to(device)
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
    
    # --- 1. Load CLIP Model ---
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()

    # --- 2. Get Old and New Class Labels ---
    old_train_df = load_data_from_folder(OLD_TRAIN_DATA_FOLDER, old_format=True)
    old_unique_labels = sorted(old_train_df['labels'].unique())
    print(f"Found {len(old_unique_labels)} original classes.")

    new_train_df = load_data_from_folder(NEW_TRAIN_DATA_FOLDER)
    new_unique_labels = sorted(new_train_df['labels'].unique())
    print(f"Found {len(new_unique_labels)} total classes in the new dataset.")

    # --- 3. Initialize New Model with Expanded Classes ---
    print("Initializing new CoOp-CLIP model for expanded classes...")
    # Create prompts and tokens for the new, full set of labels
    new_prompts = [f"a photo of a pacemaker {name}" for name in new_unique_labels]
    new_tokenized_prompts = torch.cat([clip.tokenize(p) for p in new_prompts]).to(device)

    # Initialize the new prompt learner and the full CoOp model
    new_prompt_learner = CoOpPromptLearner(
        clip_model=clip_model,
        n_ctx=16,
        n_cls=len(new_unique_labels),
        tokenized_prompts=new_tokenized_prompts
    ).to(device).to(clip_model.dtype)
    new_coop_model = CoOpCLIP(clip_model, new_prompt_learner, new_tokenized_prompts).to(device).to(clip_model.dtype)

    # --- 4. Transfer Weights from Old Model ---
    if os.path.exists(OLD_MODEL_PATH):
        print(f"Loading weights from pre-trained model: {OLD_MODEL_PATH}")
        old_state_dict = torch.load(OLD_MODEL_PATH, map_location=device)
        
        # Extract the learned context vectors (prompts) from the old model
        old_ctx = old_state_dict['prompt_learner.ctx']
        
        # Get the context vector parameter from the new, randomly initialized model
        new_ctx = new_coop_model.prompt_learner.ctx
        
        print("Transferring learned prompt weights for common classes...")
        transferred_count = 0
        with torch.no_grad():
            for i, label in enumerate(new_unique_labels):
                if label in old_unique_labels:
                    # Find the index of this class in the old model's label list
                    old_idx = old_unique_labels.index(label)
                    # Copy the learned prompt from the old model to the new one
                    new_ctx[i] = old_ctx[old_idx]
                    transferred_count += 1
        
        print(f"Successfully transferred weights for {transferred_count} out of {len(old_unique_labels)} original classes.")
    else:
        print(f"Warning: No pre-trained model found at {OLD_MODEL_PATH}. Starting training from scratch.")

    # --- 5. Train the New Model ---
    print("\n" + "="*30)
    print("Starting Training on Expanded Dataset")
    print("="*30)
    
    # Create new datasets and dataloader
    new_test_df = load_data_from_folder(NEW_TEST_DATA_FOLDER)
    new_train_dataset = PacemakerDataset(new_train_df, preprocess)
    new_test_dataset = PacemakerDataset(new_test_df, preprocess)
    
    # Fit a label encoder on the new, full set of labels
    label_encoder = LabelEncoder().fit(new_unique_labels)

    train_loader = DataLoader(new_train_dataset, batch_size=128, shuffle=True)
    
    train_coop(new_coop_model, train_loader, label_encoder)

    # --- 6. Evaluate the Final Model ---
    print("\n" + "="*30)
    print("Running Final Evaluation")
    print("="*30)

    if os.path.exists(NEW_MODEL_PATH):
        metrics = evaluate_coop_model(
            NEW_MODEL_PATH,
            new_test_dataset,
            label_encoder,
            new_unique_labels,
            new_tokenized_prompts
        )
        print(f"Evaluation Metrics: {metrics}")
        # Save the final results to a JSON file
        with open(NEW_RESULTS_PATH, "w") as f:
            json.dump({k: f"{v:.4f}" for k, v in metrics.items()}, f)
        print(f"Results saved to {NEW_RESULTS_PATH}")
    else:
        print(f"Model weights not found at '{NEW_MODEL_PATH}'. Evaluation skipped.")
