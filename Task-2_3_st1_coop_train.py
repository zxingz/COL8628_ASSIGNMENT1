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

# Variables
base_folder = f"data{os.sep}pacemakers"
train_data_folder = f"data{os.sep}pacemakers{os.sep}Train"
test_data_folder = f"data{os.sep}pacemakers{os.sep}Test"
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

# Update the unique labels
unique_labels = sorted(list(set(train_pacemaker_data['labels'])))
le = LabelEncoder()
le.fit(unique_labels)

# Fixed class-specific texts
class_prompts = {x: f"a photo of a pacemaker {x}" for x in unique_labels}
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
    model_weights_path = f'weights{os.sep}task_2_3_st1_best_coop_model.pth'
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
            torch.save(coop_clip.state_dict(), f'weights{os.sep}task_2_3_st1_best_coop_model.pth')
            
            # Save training history
            with open(f'results{os.sep}task_2_3_st1_coop_training_history_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json', 'w') as f:
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
    
    # # --- INFERENCE AND EVALUATION ---
    # print("\n" + "="*30)
    # print("Running Inference and Evaluation")
    # print("="*30)
    
    # model_weights_path = f'weights{os.sep}task_2_3_st1_best_coop_model.pth'
    
    # if os.path.exists(model_weights_path):
    #     metrics = evaluate_coop_model(model_weights_path)
    #     # Save Results
    #     with open(f"results{os.sep}task-2_3_st1_coop.json", "w") as file:
    #         file.write(json.dumps({
    #             "Top-1 Accuracy": f"{metrics['accuracy']:.4f}", \
    #             "F1-Score": f"{metrics['f1_score']:.4f}", \
    #             "AUC-ROC": f"{metrics['auc_roc']:.4f}"
    #         }))
    # else:
    #     print(f"Model weights not found at '{model_weights_path}'.")
    #     print("Please train the model first by running the train_coop() function.")
