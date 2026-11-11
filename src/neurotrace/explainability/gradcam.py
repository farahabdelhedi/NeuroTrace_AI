
"""
NeuroTrace - Grad-CAM Explainability

Génère une carte thermique indiquant les régions de l'image
qui contribuent à une classe prédite par ResNet50.

Attention :
Grad-CAM est une visualisation explicative.
Ce n'est ni une segmentation de tumeur ni une preuve médicale.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from config.model_config import (
    CLASSES,
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD
)

from neurotrace.ai.model_loader import ModelLoader


@dataclass
class GradCAMResult:
    predicted_class: str
    confidence: float
    heatmap_path: str
    overlay_path: str
    metadata_path: str


class GradCAM:

    def __init__(self):

        # Charger le modèle NeuroTrace
        loader = ModelLoader()
        self.model, self.device = loader.load_model()

        # Même transformation que le prédicteur
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD
            )
        ])

        # Stockage des activations et gradients
        self.activations = None
        self.gradients = None

        # Dernière couche convolutionnelle de ResNet50
        self.target_layer = self.model.layer4[-1]

        # Installer le hook
        self.hook = self.target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):

        # Conserver les activations
        self.activations = output.detach()

        # Capturer les gradients pendant la rétropropagation
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):

        self.gradients = gradient.detach()

    def generate(
        self,
        image_path: str,
        output_dir: str,
        target_class: str | None = None
    ) -> GradCAMResult:

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(
                f"Image introuvable : {image_path}"
            )

        if target_class is not None and target_class not in CLASSES:
            raise ValueError(
                f"Classe inconnue : {target_class}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Charger l'image originale
        with Image.open(image_path) as image:
            original_image = image.convert("RGB")
            original_image = original_image.resize(IMAGE_SIZE)

        # 2. Transformer l'image pour ResNet50
        input_tensor = self.transform(original_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # 3. Réinitialiser les gradients
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        # 4. Effectuer le passage avant
        # Ne pas utiliser torch.no_grad() : Grad-CAM a besoin
        # des gradients pour construire la carte thermique.
        outputs = self.model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        predicted_index = torch.argmax(probabilities).item()
        predicted_class = CLASSES[predicted_index]

        # Choisir la classe à expliquer
        if target_class is None:
            target_index = predicted_index
        else:
            target_index = CLASSES.index(target_class)

        confidence = probabilities[target_index].item() * 100

        # 5. Calculer les gradients de la classe cible
        score = outputs[0, target_index]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Impossible de récupérer les gradients ou activations."
            )

        # 6. Calculer les poids des canaux
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # 7. Construire la carte Grad-CAM
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        # 8. Redimensionner la carte à 224 x 224
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()

        # 9. Normaliser la carte entre 0 et 1
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # 10. Transformer la carte en image couleur
        heatmap = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # 11. Préparer l'image originale
        original_array = np.array(original_image)

        # 12. Superposer la heatmap à l'image originale
        overlay = cv2.addWeighted(
            original_array,
            0.6,
            heatmap,
            0.4,
            0
        )

        # 13. Construire les chemins de sortie
        prefix = image_path.stem

        heatmap_path = output_dir / f"{prefix}_heatmap.png"
        overlay_path = output_dir / f"{prefix}_gradcam.png"
        metadata_path = output_dir / f"{prefix}_gradcam.json"

        # 14. Enregistrer les images
        Image.fromarray(heatmap).save(heatmap_path)
        Image.fromarray(overlay).save(overlay_path)

        # 15. Enregistrer les métadonnées
        metadata = {
            "source_image": image_path.name,
            "predicted_class": predicted_class,
            "explained_class": CLASSES[target_index],
            "confidence": round(confidence, 2),
            "target_layer": "layer4[-1]",
            "method": "Grad-CAM",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "heatmap_file": heatmap_path.name,
            "overlay_file": overlay_path.name,
            "interpretation_limit": (
                "Visualisation explicative expérimentale, "
                "ne constituant pas une segmentation ni un diagnostic médical."
            )
        }

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)

        return GradCAMResult(
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            heatmap_path=str(heatmap_path),
            overlay_path=str(overlay_path),
            metadata_path=str(metadata_path)
        )

    def close(self):
        """Retirer le hook lorsque l'objet n'est plus utilisé."""
        self.hook.remove()
