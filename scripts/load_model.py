
import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

# Télécharger les poids
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.safetensors"
)

# Construire ResNet50
model = models.resnet50(weights=None)

# Reconstruire la tête de classification
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

# Charger les poids
state_dict = load_file(model_path)
model.load_state_dict(state_dict)

# Mode évaluation
model.eval()

print("Modèle ResNet50 chargé avec succès !")
print("Nombre de classes :", model.fc[-1].out_features)
print("\n=== STRUCTURE DU CLASSIFICATEUR ===")

for name, module in model.named_modules():
    if name.startswith("fc"):
        print(name, ":", module)