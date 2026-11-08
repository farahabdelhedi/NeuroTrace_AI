
"""
NeuroTrace - Model Loader

Responsabilités :
- Construire l'architecture ResNet50.
- Récupérer les poids depuis Hugging Face.
- Charger les poids du modèle.
- Placer le modèle sur CPU ou GPU.
"""

import torch
import torch.nn as nn

from torchvision import models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config.model_config import (
    MODEL_REPO_ID,
    MODEL_FILENAME
)


class ModelLoader:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_model(self):

        print("Construction de ResNet50...")

        # 1. Construire l'architecture
        model = models.resnet50(weights=None)

        # 2. Reproduire exactement la couche finale
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )

        print("Récupération des poids du modèle...")

        # 3. Télécharger ou récupérer le fichier en cache
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME
        )

        # 4. Charger les poids
        state_dict = load_file(model_path)

        model.load_state_dict(state_dict)

        # 5. Placer le modèle sur le périphérique disponible
        model.to(self.device)

        # 6. Mode évaluation
        model.eval()

        print("Modèle chargé avec succès !")
        print("Appareil utilisé :", self.device)

        return model, self.device
