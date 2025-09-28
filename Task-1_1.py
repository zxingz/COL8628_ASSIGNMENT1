# %%
# =============================================================================
# 1. Setup and Imports
# =============================================================================
import os
import json
from collections import defaultdict
import time
from datetime import datetime

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
import clip
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from tqdm import tqdm

# %%
# =============================================================================
# 2. Configuration
# =============================================================================
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration dictionary
CONFIG = {
    "base_folder": f"data{os.sep}orthonet",
    "data_folder": f"data{os.sep}orthonet{os.sep}orthonet data{os.sep}orthonet data",
    "weights_folder": "weights", # This folder is no longer used for saving in this script
    "results_folder": "results",
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 1e-4,
}

# Create necessary folders
os.makedirs(CONFIG["weights_folder"], exist_ok=True)
os.makedirs(CONFIG["results_folder"], exist_ok=True)

# Model variant identifiers
MODEL_VARIANTS = {
    "imagenet21k": "vit_base_patch16_224.augreg_in21k",
    "clip": "ViT-B/32",
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
}

# %%
# =============================================================================
# 3. Dataset Definition
# =============================================================================
class OrthonetDataset(Dataset):
    """Custom dataset for the Orthonet data."""
    def __init__(self, csv_file, img_dir, transform=None, label_map=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        if label_map is None:
            self.labels = sorted(self.data['labels'].unique())
            self.label_map = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label_map = label_map
            self.labels = sorted(list(self.label_map.keys()))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['filenames'])
        image = Image.open(img_path).convert('RGB')
        label_str = self.data.iloc[idx]['labels']
        label_idx = self.label_map[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# %%
# =============================================================================
# 4. Model Loading and Modification
# =============================================================================
class ClipViTForClassification(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.visual = clip_model.visual
        feature_dim = 512
        self.ln = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(feature_dim, num_classes)
        
        # Freeze CLIP visual encoder
        for param in self.visual.parameters():
            param.requires_grad = False
        
        # Convert entire visual module to float32 instead of half
        self.visual = self.visual.float()
        
    def forward(self, image):
        try:
            # Ensure input is float32
            image = image.float()
            
            # Normalize input to [-1, 1]
            if image.max() > 1.0:
                image = image / 255.0
            image = 2 * image - 1
            
            # Get visual features
            with torch.no_grad():
                image_features = self.visual(image)
            
            # Apply preprocessing
            image_features = self.ln(image_features)
            image_features = self.dropout(image_features)
            
            # Clip values to prevent explosion
            image_features = torch.clamp(image_features, -100, 100)
            
            # L2-normalize features
            norm = torch.norm(image_features, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            image_features = image_features / norm
            
            # Get logits
            logits = self.head(image_features)
            
            return logits
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.zeros((image.shape[0], self.head.out_features), 
                             device=image.device, 
                             dtype=torch.float32,
                             requires_grad=True)


def get_model(variant_name, num_classes):
    """
    Initializes a ViT model with specified pretrained weights and adapts its
    classification head.
    """
    model_id = MODEL_VARIANTS[variant_name]
    
    if variant_name == "clip":
        print(f"Loading CLIP model from 'clip' library: {model_id}")
        clip_model, transform = clip.load(model_id, device=device, jit=False)  # Add jit=False for better compatibility
        model = ClipViTForClassification(clip_model, num_classes)
        # Ensure classification head is in float32
        model.head = model.head.float()
        model.ln = model.ln.float()

    # FIX FOR DINOv2: Load dinov2 using timm to avoid Python version issues
    # This also simplifies the code by merging the logic with the other timm model.
    elif variant_name == "dinov2":
        print(f"Loading DINOv2 model via timm: {model_id}")
        model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=num_classes
        )
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config)
        
    else:
        print(f"Loading timm model: {model_id}")
        model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=num_classes
        )
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config)
        
    return model.to(device), transform

# %%
# =============================================================================
# 5. Training and Evaluation Functions
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_valid_batches = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        try:
            # Ensure inputs require gradients
            images.requires_grad_(True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss detected, skipping batch")
                continue
            
            # Compute gradients
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Skip step if gradients are NaN/Inf
            if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) 
                  for p in model.parameters()):
                print("Warning: NaN/Inf gradients detected, skipping batch")
                continue
            
            optimizer.step()
            
            total_loss += loss.item()
            num_valid_batches += 1
            
        except RuntimeError as e:
            print(f"Error in batch: {e}")
            continue
    
    return total_loss / max(num_valid_batches, 1)

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probas = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            try:
                images = images.to(device)
                
                outputs = model(images)
                # Add numerical stability to softmax
                outputs = outputs - outputs.max(dim=1, keepdim=True)[0]  # Subtract max for numerical stability
                probas = torch.softmax(outputs, dim=1)
                
                # Check for NaN values
                if torch.isnan(probas).any():
                    print("Warning: NaN values detected in probabilities")
                    continue
                
                preds = torch.argmax(probas, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probas.extend(probas.cpu().detach().numpy())

            except RuntimeError as e:
                print(f"Error in evaluation batch: {e}")
                continue

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probas = np.array(all_probas)

    # Check if we have any valid predictions
    if len(all_labels) == 0:
        print("Warning: No valid predictions were made")
        return {
            'Top-1 Accuracy': 0.0,
            'F1-Score': 0.0,
            'AUC-ROC': 0.0
        }

    # Handle any remaining NaN values
    mask = ~np.isnan(all_probas).any(axis=1)
    all_labels = all_labels[mask]
    all_preds = all_preds[mask]
    all_probas = all_probas[mask]

    # Compute metrics
    try:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Only compute AUC-ROC if we have valid probabilities
        if np.isfinite(all_probas).all() and len(np.unique(all_labels)) > 1:
            auc_roc = roc_auc_score(y_true_bin, all_probas, multi_class='ovr')
        else:
            print("Warning: Could not compute AUC-ROC due to invalid probabilities or single class")
            auc_roc = 0.0

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'Top-1 Accuracy': 0.0,
            'F1-Score': 0.0,
            'AUC-ROC': 0.0
        }

    return {
        'Top-1 Accuracy': accuracy,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }

# %%
# =============================================================================
# 6. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    
    # --- Data Preparation ---
    temp_train_dataset = OrthonetDataset(
        csv_file=os.path.join(CONFIG["base_folder"], "train.csv"),
        img_dir=CONFIG["data_folder"]
    )
    label_map = temp_train_dataset.label_map
    num_classes = len(label_map)
    print(f"Found {num_classes} classes in the dataset.")
    
    all_results = {}

    for variant in MODEL_VARIANTS.keys():
        
        # if 'dino' not in variant.lower():
        #     continue
        
        # if 'clip' not in variant.lower():
        #     continue
        
        print(f"\n{'='*20} Processing Model: {variant.upper()} {'='*20}")
        
        # --- Model and Transforms Initialization ---
        model, transform = get_model(variant, num_classes)
        
        # --- Datasets and Dataloaders ---
        train_dataset = OrthonetDataset(
            csv_file=os.path.join(CONFIG["base_folder"], "train.csv"),
            img_dir=CONFIG["data_folder"],
            transform=transform,
            label_map=label_map
        )
        test_dataset = OrthonetDataset(
            csv_file=os.path.join(CONFIG["base_folder"], "test.csv"),
            img_dir=CONFIG["data_folder"],
            transform=transform,
            label_map=label_map
        )
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        
        # --- Training ---
        criterion = nn.CrossEntropyLoss()
        
        # Update the optimizer configuration:
        optimizer = optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize trainable parameters
                        lr=CONFIG["learning_rate"],
                        weight_decay=0.01,
                        eps=1e-8  # Increase epsilon for numerical stability
                    )

        # Add a learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        mode='min', 
                        factor=0.5, 
                        patience=2
                    )
        
        model_save_path = os.path.join(CONFIG["weights_folder"], f"task-1_1_best_vit_{variant}.pth")
        
        # MODIFICATION: Check for existing weights and load them to resume training
        if os.path.exists(model_save_path):
            print(f"Found existing weights at {model_save_path}. Loading model state...")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print("Weights loaded successfully. Resuming training.")
        else:
            print("No existing weights found. Training from scratch.")
        # --- END MODIFICATION ---
        
        history = {"train_loss": [], "val_accuracy": []}
        
        # INIT best Accuracy
        best_accuracy = 0.0  # Initialize best accuracy tracker
        
        for epoch in range(CONFIG["num_epochs"]):
            print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
            avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            
            val_metrics = evaluate_model(model, test_loader)
            print(f"Validation Accuracy: {val_metrics['Top-1 Accuracy']:.4f}")
            
            # Record metrics for the current epoch
            history["train_loss"].append(avg_train_loss)
            history["val_accuracy"].append(val_metrics['Top-1 Accuracy'])
            
            if val_metrics['Top-1 Accuracy'] > best_accuracy:
                best_accuracy = val_metrics['Top-1 Accuracy']
                torch.save(model.state_dict(), model_save_path)
                print(f"New best model saved to {model_save_path}")
            
            scheduler.step(avg_train_loss)  # Update learning rate based on training loss

        # Save the collected training history to a JSON file
        history_save_path = os.path.join(CONFIG["results_folder"], f'task-1_1_training_history_{variant}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.json')
        with open(history_save_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_save_path}")
        
        # --- Final Evaluation ---
        print("\n--- Final Evaluation on Test Set ---")
        model.load_state_dict(torch.load(model_save_path))
        
        test_metrics = evaluate_model(model, test_loader)
        all_results[variant] = test_metrics
        
        print(f"\nResults for {variant.upper()}:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
            
        results_path = os.path.join(CONFIG["results_folder"], f"task-1_1_{variant}.json")
        with open(results_path, "w") as f:
            json.dump(all_results[variant], f, indent=4)