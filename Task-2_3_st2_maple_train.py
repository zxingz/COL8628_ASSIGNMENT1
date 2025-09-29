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
OLD_MODEL_PATH = f"weights{os.sep}task_2_3_st1_best_maple_model.pth"

# Paths for the new model and the new, expanded dataset
NEW_TRAIN_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Train"
NEW_TEST_DATA_FOLDER = f"data{os.sep}pacemakers{os.sep}Test"

# Output paths for the newly trained model and its results
NEW_MODEL_PATH = f"weights{os.sep}task_2_3_st2_best_maple_model.pth"
NEW_HISTORY_PATH_PREFIX = f"results{os.sep}task_2_3_st2_maple_training_history"
NEW_RESULTS_PATH = f"results{os.sep}task-2_3_st2_maple.json"

# Create necessary folders
os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Model hyperparameters
N_CTX = 16  # Number of context tokens
N_DEEP = 3  # Number of deep prompt layers

# %%

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculates and returns evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    if len(np.unique(y_true_encoded)) != y_pred_proba.shape[1]:
         print(f"Warning: Class mismatch in evaluation. AUC may be incorrect.")
         auc_roc = 0.5 
    else:
        auc_roc = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
    
    return {'accuracy': accuracy, 'f1_score': f1, 'auc_roc': auc_roc}

# Dataset Definition
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
        return self.preprocess(image), label

# MaPLe Model Definitions
class MaPLePromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, n_cls, tokenized_prompts, n_deep):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.n_deep = n_deep
        
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.transformer.width

        ctx_vectors = torch.empty(n_ctx, text_ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.deep_vision_prompts = nn.Parameter(torch.empty(n_deep, n_ctx, vision_ctx_dim))
        nn.init.normal_(self.deep_vision_prompts, std=0.02)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

    def forward(self):
        return self.ctx, self.deep_vision_prompts

class MaPLeCLIP(nn.Module):
    def __init__(self, clip_model, prompt_learner, tokenized_prompts):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        
        text_ctx_dim = clip_model.ln_final.weight.shape[0]
        vision_ctx_dim = clip_model.visual.transformer.width
        
        self.prompt_projector = nn.Sequential(
            nn.Linear(vision_ctx_dim, text_ctx_dim),
        )

        # --- RUNTIME ERROR FIX START ---
        # The original attention mask in CLIP's text transformer is a fixed 77x77 matrix.
        # Since MaPLe injects deep prompts and increases the sequence length to 77 + N_CTX,
        # this fixed mask becomes invalid. We must disable it by setting it to None.
        for resblock in self.clip_model.transformer.resblocks:
            resblock.attn_mask = None
        # --- RUNTIME ERROR FIX END ---


    def encode_image(self, image, deep_vision_prompts):
        x = self.clip_model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        for i, resblock in enumerate(self.clip_model.visual.transformer.resblocks):
            if i < self.prompt_learner.n_deep:
                prompt = deep_vision_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                cls_token, patches = x[:1], x[1:]
                x = torch.cat([cls_token, prompt, patches], dim=0)
            x = resblock(x)
            if i < self.prompt_learner.n_deep:
                x = torch.cat([x[:1], x[1+self.prompt_learner.n_ctx:]], dim=0)

        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.ln_post(x[:, 0, :])
        
        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj
        return x

    def encode_text(self, ctx, deep_text_prompts):
        prefix = self.prompt_learner.token_prefix
        suffix = self.prompt_learner.token_suffix
        ctx_expanded = ctx.unsqueeze(0).expand(self.prompt_learner.n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx_expanded, suffix], dim=1)
        
        x = prompts + self.clip_model.positional_embedding.to(prompts.dtype)
        x = x.permute(1, 0, 2)

        # --- BUG FIX START: Correctly inject deep text prompts ---
        for i, resblock in enumerate(self.clip_model.transformer.resblocks):
            if i < self.prompt_learner.n_deep:
                # Inject the learned deep prompt
                prompt = deep_text_prompts[i].unsqueeze(1).expand(-1, x.shape[1], -1)
                sos_token, rest_of_sequence = x[:1], x[1:]
                x = torch.cat([sos_token, prompt, rest_of_sequence], dim=0)
            
            x = resblock(x)

            if i < self.prompt_learner.n_deep:
                # Remove the injected prompt after the block
                x = torch.cat([x[:1], x[1+self.prompt_learner.n_ctx:]], dim=0)
        # --- BUG FIX END ---
        
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        
        eot_indices = self.tokenized_prompts.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.clip_model.text_projection
        return text_features

    def forward(self, image):
        ctx, deep_vision_prompts = self.prompt_learner()
        image_features = self.encode_image(image, deep_vision_prompts)
        deep_text_prompts = self.prompt_projector(deep_vision_prompts)
        text_features = self.encode_text(ctx, deep_text_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features @ text_features.t()

def load_data_from_folder(folder_path):
    """Loads image paths and labels from a directory into a DataFrame."""
    df = pd.DataFrame()
    for folder in glob(f"{folder_path}{os.sep}*"):
        try: label, _ = os.path.basename(folder).split(' - ')
        except ValueError: continue
        for file in glob(f"{folder}{os.sep}*"):
            df = pd.concat([df, pd.DataFrame([{'filenames': file, 'labels': f"({label})"}])], ignore_index=True)
    return df.reset_index(drop=True)

def train_maple(model, train_loader, label_encoder):
    """Trains the MaPLe model."""
    optimizer = optim.AdamW(list(model.prompt_learner.parameters()) + list(model.prompt_projector.parameters()), lr=0.002, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    patience, best_loss, patience_counter = 10, float('inf'), 0
    history = defaultdict(list)
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/50'):
            images = images.to(device, dtype=model.clip_model.dtype)
            labels = torch.tensor(label_encoder.transform(labels), dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels.to(logits.device))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), NEW_MODEL_PATH)
            print(f"Model saved to {NEW_MODEL_PATH}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping.')
            break
            
    with open(f'{NEW_HISTORY_PATH_PREFIX}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json', 'w') as f:
        json.dump(history, f)

def evaluate_maple_model(model_path, test_dataset, label_encoder, unique_labels, tokenized_prompts, clip_model):
    """Evaluates the trained MaPLe model."""
    prompt_learner = MaPLePromptLearner(clip_model, n_ctx=N_CTX, n_cls=len(unique_labels), tokenized_prompts=tokenized_prompts, n_deep=N_DEEP).to(device)
    loaded_model = MaPLeCLIP(clip_model, prompt_learner, tokenized_prompts).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.to(clip_model.dtype).eval()

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    all_labels, all_preds, all_pred_probas = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            logits = loaded_model(images.to(device, dtype=clip_model.dtype))
            probas = logits.softmax(dim=-1)
            preds = probas.argmax(dim=-1)
            all_labels.extend(labels)
            all_preds.extend(label_encoder.inverse_transform(preds.cpu().numpy()))
            all_pred_probas.extend(probas.cpu().numpy())

    return evaluate_metrics(all_labels, all_preds, np.array(all_pred_probas))

# %%
if __name__ == "__main__":
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters(): param.requires_grad = False

    new_train_df = load_data_from_folder(NEW_TRAIN_DATA_FOLDER)
    new_unique_labels = sorted(new_train_df['labels'].unique())
    print(f"Found {len(new_unique_labels)} total classes.")

    prompts = [" ".join(["X"] * N_CTX) + f" a photo of a pacemaker {name}" for name in new_unique_labels]
    new_tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    new_prompt_learner = MaPLePromptLearner(clip_model, n_ctx=N_CTX, n_cls=len(new_unique_labels), tokenized_prompts=new_tokenized_prompts, n_deep=N_DEEP).to(device)
    new_maple_model = MaPLeCLIP(clip_model, new_prompt_learner, new_tokenized_prompts).to(device)
    
    # --- RUNTIME ERROR FIX START ---
    # Cast the entire model to the CLIP model's dtype (e.g., float16)
    new_maple_model.to(clip_model.dtype)
    
    # LayerNorm layers are unstable in float16, so we cast them back to float32
    for name, module in new_maple_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.to(torch.float32)
    # --- RUNTIME ERROR FIX END ---

    if os.path.exists(OLD_MODEL_PATH):
        print(f"Loading weights from pre-trained model: {OLD_MODEL_PATH}")
        old_state_dict = torch.load(OLD_MODEL_PATH, map_location=device)
        
        weights_to_load = {}
        for k, v in old_state_dict.items():
            if k.startswith("prompt_learner.") and "token" not in k:
                weights_to_load[k] = v
            # --- FIX START ---
            # The prompt_projector weights from the old model have an incompatible shape
            # (output dim 48 vs 512). This indicates an issue in the original training script.
            # We will skip loading these weights and let the new model learn them from scratch.
            # elif k.startswith("prompt_projector."):
            #     weights_to_load[k] = v
            # --- FIX END ---
        
        new_maple_model.load_state_dict(weights_to_load, strict=False)
        print("Successfully transferred weights from the pre-trained prompt learner.")
        if any(k.startswith("prompt_projector.") for k in old_state_dict.keys()):
            print("Note: Incompatible 'prompt_projector' weights were skipped and will be re-initialized.")

    else:
        print(f"Warning: No pre-trained model found at {OLD_MODEL_PATH}. Training from scratch.")

    print("\nStarting Training on Expanded Dataset")
    
    new_test_df = load_data_from_folder(NEW_TEST_DATA_FOLDER)
    label_encoder = LabelEncoder().fit(new_unique_labels)
    train_loader = DataLoader(PacemakerDataset(new_train_df, preprocess), batch_size=64, shuffle=True)
    
    train_maple(new_maple_model, train_loader, label_encoder)

    print("\nRunning Final Evaluation")

    if os.path.exists(NEW_MODEL_PATH):
        test_dataset = PacemakerDataset(new_test_df, preprocess)
        metrics = evaluate_maple_model(NEW_MODEL_PATH, test_dataset, label_encoder, new_unique_labels, new_tokenized_prompts, clip_model)
        print(f"Evaluation Metrics: {metrics}")
        with open(NEW_RESULTS_PATH, "w") as f:
            json.dump({k: f"{v:.4f}" for k, v in metrics.items()}, f, indent=4)
        print(f"Results saved to {NEW_RESULTS_PATH}")
    else:
        print(f"Model weights not found. Evaluation skipped.")