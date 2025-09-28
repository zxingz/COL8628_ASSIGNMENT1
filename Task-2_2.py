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
base_folder = f"data{os.sep}pacemakers"
test_folder = f"{base_folder}{os.sep}Test"

# Functions
def zero_shot_prediction(model, dataset, class_prompts, device):
    # Encode text prompts
    text_inputs = clip.tokenize([class_prompts[c] for c in class_prompts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Initialize lists for predictions and true labels
    all_predictions = []
    all_probs = []
    all_labels = []
    
    loader = DataLoader(dataset, batch_size=32)
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions
            predictions = [list(class_prompts.keys())[idx] for idx in similarity.argmax(dim=1)]
            
            all_predictions.extend(predictions)
            all_probs.extend(similarity.cpu().numpy())
            all_labels.extend(labels)
    
    return all_predictions, np.array(all_probs), all_labels

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
class PacemakerDataset(Dataset):
    def __init__(self, data, img_dir, preprocess):
        self.data = data
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

# %%
# # GEMINI 2.5 pro prompt generation
# For each of the following pacemakers manufacturer and verison/model name describe the location, structure, shape and other visual attributes that would be apparent in an X ray image.
# output in json format where the key are the (manufacturer, model name) and values are the descriptions. make the values as strings and limit the token length of values to 75 tokens.

# Manufacturer: BIO and Model Name: Actros_Philos;Manufacturer: BIO and Model Name: Cyclos;Manufacturer: BIO and Model Name: Evia;Manufacturer: BOS and Model Name: Altrua_Insignia;Manufacturer: BOS and Model Name: Autogen_Teligen_Energen_Cognis;Manufacturer: BOS and Model Name: Contak Renewal 4;Manufacturer: BOS and Model Name: Contak Renewal TR2;Manufacturer: BOS and Model Name: ContakTR_Discovery_Meridian_Pulsar Max;Manufacturer: BOS and Model Name: Emblem;Manufacturer: BOS and Model Name: Ingenio;Manufacturer: BOS and Model Name: Proponent;Manufacturer: BOS and Model Name: Ventak Prizm;Manufacturer: BOS and Model Name: Visionist;Manufacturer: BOS and Model Name: Vitality;Manufacturer: MDT and Model Name: AT500;Manufacturer: MDT and Model Name: Adapta_Kappa_Sensia_Versa;Manufacturer: MDT and Model Name: Advisa;Manufacturer: MDT and Model Name: Azure;Manufacturer: MDT and Model Name: C20_T20;Manufacturer: MDT and Model Name: C60 DR;Manufacturer: MDT and Model Name: Claria_Evera_Viva;Manufacturer: MDT and Model Name: Concerto_Consulta_Maximo_Protecta_Secura;Manufacturer: MDT and Model Name: EnRhythm;Manufacturer: MDT and Model Name: Insync III;Manufacturer: MDT and Model Name: Maximo;Manufacturer: MDT and Model Name: REVEAL;Manufacturer: MDT and Model Name: REVEAL LINQ;Manufacturer: MDT and Model Name: Sigma;Manufacturer: MDT and Model Name: Syncra;Manufacturer: MDT and Model Name: Vita II;Manufacturer: SOR and Model Name: Elect;Manufacturer: SOR and Model Name: Elect XS Plus;Manufacturer: SOR and Model Name: MiniSwing;Manufacturer: SOR and Model Name: Neway;Manufacturer: SOR and Model Name: Ovatio;Manufacturer: SOR and Model Name: Reply;Manufacturer: SOR and Model Name: Rhapsody_Symphony;Manufacturer: SOR and Model Name: Thesis;Manufacturer: STJ and Model Name: Accent;Manufacturer: STJ and Model Name: Allure Quadra;Manufacturer: STJ and Model Name: Ellipse;Manufacturer: STJ and Model Name: Identity;Manufacturer: STJ and Model Name: Quadra Assura_Unify;Manufacturer: STJ and Model Name: Victory;Manufacturer: STJ and Model Name: Zephyr;

class_prompts = {
  "(BIO, Actros_Philos)": "Located in the upper chest, this older pacemaker has a rectangular can with rounded corners. The radiopaque identifier is typically the word 'BIOTRONIK' or 'BIO' visible on the device circuitry.",
  "(BIO, Cyclos)": "An older pacemaker with a rectangular body and rounded corners, typically placed in a subcutaneous pectoral pocket. Its radiopaque marking is 'BIOTRONIK' or 'BIO'.",
  "(BIO, Evia)": "A modern pacemaker with a smooth, contoured oval or teardrop shape. It is placed in the upper chest and shows a 'BIO' radiopaque marker, often followed by a model code like 'DR-T'.",
  "(BOS, Altrua_Insignia)": "An older, larger pacemaker or ICD with a D-shape or rectangular can, located in the pectoral region. Its radiopaque identifier is typically 'BSCI' or a 'B' inside a circle.",
  "(BOS, Autogen_Teligen_Energen_Cognis)": "A modern ICD/CRT-D series with a rounded, asymmetric D-shape. Placed in the upper chest, it shows 'BSCI' as the radiopaque marker. CRT-D versions have a three-port header.",
  "(BOS, Contak Renewal 4)": "An older, large CRT-D with a distinct D-shape, located in the pectoral region. The radiopaque marking is 'GUIDANT' or 'BSCI', sometimes with 'CONTAK' visible.",
  "(BOS, Contak Renewal TR2)": "An older CRT-D, similar to the Renewal 4 model, with a large D-shaped can. Found in the upper chest, it displays a 'GUIDANT' or 'BSCI' radiopaque identifier.",
  "(BOS, ContakTR_Discovery_Meridian_Pulsar Max)": "A large family of older devices, typically D-shaped or rectangular. Located in the pectoral pocket, their radiopaque identifier is 'GUIDANT' or 'BSCI'.",
  "(BOS, Emblem)": "A subcutaneous ICD (S-ICD). The large generator is located on the lateral chest wall (near the armpit), not the pectoral region. The lead is subcutaneous, running parallel to the sternum.",
  "(BOS, Ingenio)": "A modern pacemaker with a small, rounded, ergonomic D-shape and a gently curved header. It is found in the upper chest and is identified by the radiopaque 'BSCI' marker.",
  "(BOS, Proponent)": "A CRT-P (pacemaker) with a rounded D-shape, implanted in the upper chest. Its radiopaque identifier is 'BSCI' along with the model code.",
  "(BOS, Ventak Prizm)": "An older ICD with a relatively thick, D-shaped can, located in the pectoral region. The visible radiopaque identifier is 'GUIDANT' or 'BSCI'.",
  "(BOS, Visionist)": "A CRT-P (pacemaker) featuring a rounded D-shape. It is located in the upper chest and shows the 'BSCI' radiopaque marker on X-ray.",
  "(BOS, Vitality)": "An older ICD series with a characteristic D-shape. It is found in the subcutaneous pectoral pocket and is identified by its 'GUIDANT' or 'BSCI' radiopaque marking.",
  "(MDT, AT500)": "An older CRT-D with a large, D-shaped can that has a distinct angled or clipped corner. Its radiopaque identifier is the Medtronic logo and 'AT500'.",
  "(MDT, Adapta_Kappa_Sensia_Versa)": "A very common pacemaker family with a rounded teardrop or 'mouse' shape. Located in the upper chest, it shows the Medtronic logo (a pyramid-like symbol) as its radiopaque marker.",
  "(MDT, Advisa)": "A modern, MRI-compatible pacemaker with a small, rounded teardrop shape, very similar to the Adapta. Its radiopaque identifier, seen in the upper chest, is the Medtronic logo.",
  "(MDT, Azure)": "A recent, very small pacemaker with a rounded teardrop shape and often a visible antenna band. Implanted pectorally, it displays the Medtronic logo as its radiopaque identifier.",
  "(MDT, C20_T20)": "An older Vitatron-branded pacemaker, often more rectangular than typical Medtronic devices. It is identified by the radiopaque 'VITATRON' text.",
  "(MDT, C60 DR)": "An older pacemaker from Vitatron (acquired by Medtronic) with a rectangular can. It is located in the upper chest and shows 'VITATRON' as its radiopaque marker.",
  "(MDT, Claria_Evera_Viva)": "A modern ICD/CRT-D family with a distinctive rounded, contoured, 'whale-like' shape. Implanted pectorally, it is identified by the Medtronic logo.",
  "(MDT, Concerto_Consulta_Maximo_Protecta_Secura)": "A large family of ICDs/CRT-Ds, typically D-shaped or rounded-rectangular. Located in the upper chest, their main identifier is the prominent Medtronic logo.",
  "(MDT, EnRhythm)": "A pacemaker with a rounded teardrop or 'mouse' shape, visually similar to the Adapta and Kappa series. It is implanted in the upper chest and displays the Medtronic logo.",
  "(MDT, Insync III)": "An older CRT-D with a large, D-shaped can and a prominent, angular header where leads connect. It is found in the pectoral region and identified by the Medtronic logo.",
  "(MDT, Maximo)": "An ICD or CRT-D from a large device family, featuring a standard D-shape. It is located in the upper chest and is identified by the Medtronic logo on X-ray.",
  "(MDT, REVEAL)": "An implantable loop recorder (ILR), not a pacemaker. It appears as a small, thin, rectangular device, like a USB stick, placed subcutaneously over the left chest.",
  "(MDT, REVEAL LINQ)": "A tiny, injectable loop recorder (ILR). It is extremely small and thin, appearing as a single metal rod (about 1.5 inches long) placed subcutaneously in the upper left chest.",
  "(MDT, Sigma)": "An older pacemaker that is more rectangular than modern Medtronic devices. It is found in the upper chest and shows the Medtronic logo as its radiopaque identifier.",
  "(MDT, Syncra)": "A CRT-P (pacemaker) with Medtronic's characteristic rounded teardrop shape. It is implanted in the pectoral region and identified by the Medtronic logo.",
  "(MDT, Vita II)": "A Vitatron-branded pacemaker (from Medtronic) that often has a more rectangular shape than other Medtronic devices. It is identified by the 'VITATRON' radiopaque logo.",
  "(SOR, Elect)": "A pacemaker with a small, rounded oval shape, located in the upper chest. Its radiopaque identifier is 'SORIN' along with a symbol, often a circle with a central dot.",
  "(SOR, Elect XS Plus)": "A small pacemaker with a rounded, compact can, visually similar to the Elect model. It is placed pectorally and shows 'SORIN' as its radiopaque marker.",
  "(SOR, MiniSwing)": "A very small and compact pacemaker with a rounded shape. It is found in the upper chest and is identified by the radiopaque text 'SORIN' and 'MiniSwing'.",
  "(SOR, Neway)": "A pacemaker with a small, rounded shape, implanted in the subcutaneous pectoral pocket. The radiopaque identifier visible on X-ray is 'SORIN'.",
  "(SOR, Ovatio)": "An ICD that is larger and more D-shaped than Sorin pacemakers. It is located in the upper chest and displays 'SORIN' as its radiopaque identifier.",
  "(SOR, Reply)": "A common pacemaker with a small, rounded oval shape and often a transparent header. Located in the upper chest, it is identified by the 'SORIN' radiopaque marker.",
  "(SOR, Rhapsody_Symphony)": "An older series of pacemakers that are more rectangular than modern Sorin devices. They are located pectorally and show 'SORIN' as the radiopaque text.",
  "(SOR, Thesis)": "A pacemaker with a small, rounded shape, located in the subcutaneous pectoral pocket. Its radiopaque identifier visible on X-ray is 'SORIN'.",
  "(STJ, Accent)": "A modern pacemaker with a small, oval or elliptical shape. Located in the upper chest, its radiopaque identifier is 'ST JUDE MEDICAL' or 'SJM', often with a crescent moon symbol.",
  "(STJ, Allure Quadra)": "A CRT-D with a rounded D-shape. Its key feature is a four-port header for a quadripolar lead. It's identified by the 'SJM' radiopaque marker and crescent symbol.",
  "(STJ, Ellipse)": "A modern ICD with a distinctive narrow, elliptical shape designed for a smaller pocket. Found in the upper chest, its radiopaque identifier is 'SJM' and a crescent symbol.",
  "(STJ, Identity)": "A pacemaker with a small, oval can shape, placed in the subcutaneous pectoral pocket. It is identified by the radiopaque text 'SJM' or 'ST JUDE MEDICAL'.",
  "(STJ, Quadra Assura_Unify)": "A modern ICD/CRT-D family with a rounded D-shape. The 'Quadra' models feature a 4-port header. The radiopaque identifier is 'SJM' with a crescent symbol.",
  "(STJ, Victory)": "An older pacemaker with a standard oval shape, located in the upper chest. Its radiopaque identifier is 'SJM' or 'ST JUDE MEDICAL'.",
  "(STJ, Zephyr)": "A pacemaker with a small, oval-shaped can, implanted in the subcutaneous pectoral pocket. The radiopaque identifier visible on X-ray is 'SJM'."
}
print(len(class_prompts), "class prompts defined.")

# %%
if __name__ == "__main__":
    
    # Dataset paths
    
    # Create datasets
    test_dataset = PacemakerDataset(test_csv, data_folder, preprocess)
    
    # Get predictions
    predictions, probs, true_labels = zero_shot_prediction(model, test_dataset, class_prompts, device)
    
    # Calculate metrics
    metrics = evaluate_metrics(true_labels, predictions, probs)
    
    # Save Results
    with open(f"results{os.sep}task-2_1.json", "w") as file:
        file.write(json.dumps({
            "Top-1 Accuracy": f"{metrics['accuracy']:.4f}", \
            "F1-Score": f"{metrics['f1_score']:.4f}", \
            "AUC-ROC": f"{metrics['auc_roc']:.4f}"
        }))
    