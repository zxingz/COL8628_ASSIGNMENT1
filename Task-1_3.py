# %%
import os
import json

import torch

import clip

import pandas as pd

import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader

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

# Classes
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

# %%
# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

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