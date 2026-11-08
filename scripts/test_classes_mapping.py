import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from pathlib import Path
from collections import Counter

# Chemins
BASE_DIR = Path(__file__).resolve().parent.parent
TESTING_DIR = BASE_DIR / "datasets" / "Testing"

# Chargement du modèle
print("Chargement du modèle...")

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

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
model.eval()

print("Modèle chargé !")

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test de 20 images par classe
for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:

    class_dir = TESTING_DIR / class_name
    images = list(class_dir.glob("*"))[:20]

    predictions = Counter()

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_index = outputs.argmax(dim=1).item()

        predictions[predicted_index] += 1

    print(f"\nClasse réelle : {class_name}")
    print("Nombre de prédictions par indice :")

    for index in range(4):
        print(f"Indice {index} : {predictions[index]}")

print("\nTest terminé.")