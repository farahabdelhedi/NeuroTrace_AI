
"""
NeuroTrace - MRI Predictor

Responsabilités :
- Charger et transformer une image.
- Effectuer la prédiction avec ResNet50.
- Calculer les probabilités des quatre classes.
- Retourner un résultat structuré.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config.model_config import (
    CLASSES,
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    MODEL_REPO_ID
)

from neurotrace.ai.model_loader import ModelLoader


# ==================================================
# RESULTAT DE LA PREDICTION
# ==================================================

@dataclass
class PredictionResult:

    predicted_class: str
    confidence: float
    class_scores: dict
    model_id: str


# ==================================================
# CLASSE DE PREDICTION
# ==================================================

class MRIPredictor:

    def __init__(self):

        # Charger le modèle
        loader = ModelLoader()

        self.model, self.device = loader.load_model()

        # Préparation de l'image
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD
            )
        ])

    def predict(self, image_path: str) -> PredictionResult:

        image_path = Path(image_path)

        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(
                f"Image introuvable : {image_path}"
            )

        # 1. Charger l'image
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            # 2. Appliquer les transformations
            image_tensor = self.transform(image)

        # 3. Ajouter la dimension du batch
        image_tensor = image_tensor.unsqueeze(0)

        # 4. Envoyer l'image sur le même appareil que le modèle
        image_tensor = image_tensor.to(self.device)

        # 5. Effectuer la prédiction
        with torch.no_grad():

            outputs = self.model(image_tensor)

            # Transformer les scores en probabilités
            probabilities = torch.softmax(outputs, dim=1)[0]

        # 6. Récupérer la classe prédite
        predicted_index = torch.argmax(probabilities).item()

        predicted_class = CLASSES[predicted_index]

        confidence = probabilities[predicted_index].item()

        # 7. Récupérer les scores de toutes les classes
        class_scores = {
            class_name: round(
                probabilities[index].item() * 100, 2
            )
            for index, class_name in enumerate(CLASSES)
        }

        # 8. Construire le résultat
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=round(confidence * 100, 2),
            class_scores=class_scores,
            model_id=MODEL_REPO_ID
        )
