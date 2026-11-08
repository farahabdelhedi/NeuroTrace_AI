
import torch
import torch.nn as nn
import pandas as pd

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pathlib import Path

# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
TESTING_DIR = BASE_DIR / "datasets" / "Testing"

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# 2. PRETRAITEMENT
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# 3. CHARGEMENT DES IMAGES
# =====================================

dataset = datasets.ImageFolder(
    root=TESTING_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

# =====================================
# 4. CHARGEMENT DU MODELE
# =====================================

print("Chargement du modèle...")

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.safetensors"
)

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model.load_state_dict(load_file(model_path))
model.to(DEVICE)
model.eval()

print("Modèle chargé !")

# =====================================
# 5. IDENTIFICATION DES ERREURS
# =====================================

errors = []
image_index = 0

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):

            real_index = labels[i].item()
            predicted_index = predictions[i].item()

            if real_index != predicted_index:

                errors.append({
                    "image_path": str(dataset.samples[image_index][0]),
                    "real_class": CLASSES[real_index],
                    "predicted_class": CLASSES[predicted_index],
                    "confidence": round(
                        confidences[i].item() * 100, 2
                    )
                })

            image_index += 1

# =====================================
# 6. EXPORT CSV
# =====================================

df = pd.DataFrame(errors)

output_path = BASE_DIR / "misclassified_images.csv"

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n========== RESULTATS ==========")
print("Nombre d'images mal classées :", len(df))
print("Fichier CSV créé :", output_path)

print("\nRépartition des erreurs :")
print(df.groupby(
    ["real_class", "predicted_class"]
).size())

print("\nExemples d'erreurs :")
print(df.head(10).to_string(index=False))
