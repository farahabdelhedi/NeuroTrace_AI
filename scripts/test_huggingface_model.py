
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pathlib import Path

# 1. Configuration
REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Charger le modèle
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

state_dict = load_file(model_path)
model.load_state_dict(state_dict)
model.eval()

print("Modèle chargé avec succès !")

# 3. Préparer une image
image_path = (
    BASE_DIR
    / "datasets_preprocessed"
    / "Testing"
    / "glioma"
    / "Te-gl_1.jpg"
)

if not image_path.exists():
    raise FileNotFoundError(
        f"Image introuvable : {image_path}"
    )

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# 4. Effectuer la prédiction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

print("\nScores obtenus :")
print(probabilities)

print("\nTest terminé.")
