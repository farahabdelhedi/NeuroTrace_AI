
"""
NeuroTrace - MRI Image Preprocessing

Responsabilités :
- Vérifier qu'une image est lisible et valide.
- Vérifier son format.
- Convertir l'image en RGB.
- Redimensionner l'image à 224 x 224.
- Enregistrer l'image préparée.
- Générer un fichier JSON contenant les métadonnées
  et les opérations effectuées.

Le traitement est compatible avec le prétraitement utilisé
lors de l'évaluation actuelle du modèle ResNet50.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# ==================================================
# 1. CONFIGURATION
# ==================================================

IMAGE_SIZE = (224, 224)

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ==================================================
# 2. RESULTAT DU PRETRAITEMENT
# ==================================================

@dataclass
class PreprocessingResult:
    """
    Contient le résultat du prétraitement d'une image.
    """

    success: bool
    input_path: str
    output_path: str | None
    operations: list[str]
    metadata: dict
    error: str | None = None
    metadata_path: str | None = None


# ==================================================
# 3. CLASSE PRINCIPALE
# ==================================================

class MRIImageProcessor:
    """
    Classe responsable du prétraitement des images IRM.
    """

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size

    # --------------------------------------------------
    # 3.1 VALIDATION DE L'IMAGE
    # --------------------------------------------------

    def validate_image(self, image_path: Path):
        """
        Vérifie :
        - L'existence du fichier.
        - La validité de son extension.
        - La lisibilité de l'image.
        - La validité de ses dimensions.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image introuvable : {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                "Le chemin fourni ne correspond pas à un fichier."
            )

        if image_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Format non supporté : {image_path.suffix}"
            )

        # Vérification de l'intégrité du fichier
        with Image.open(image_path) as image:
            image.verify()

        # Réouverture pour récupérer les dimensions
        with Image.open(image_path) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Dimensions de l'image invalides."
                )

        return width, height

    # --------------------------------------------------
    # 3.2 PRETRAITEMENT
    # --------------------------------------------------

    def process_image(
        self,
        input_path: str,
        output_path: str
    ) -> PreprocessingResult:

        """
        Prétraite une image IRM.

        Étapes :
        1. Validation.
        2. Chargement.
        3. Conversion RGB.
        4. Redimensionnement.
        5. Enregistrement.
        6. Génération des métadonnées JSON.
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        operations = []

        try:

            # ------------------------------------------
            # ETAPE 1 : VALIDATION
            # ------------------------------------------

            original_width, original_height = (
                self.validate_image(input_path)
            )

            operations.append(
                "Validation de l'image"
            )

            # ------------------------------------------
            # ETAPE 2 : CHARGEMENT
            # ------------------------------------------

            with Image.open(input_path) as image:

                original_format = image.format
                original_mode = image.mode

                # --------------------------------------
                # ETAPE 3 : CONVERSION RGB
                # --------------------------------------

                image = image.convert("RGB")

                operations.append(
                    "Conversion du format couleur en RGB"
                )

                # --------------------------------------
                # ETAPE 4 : REDIMENSIONNEMENT
                # --------------------------------------

                image = image.resize(
                    self.image_size,
                    Image.Resampling.BILINEAR
                )

                operations.append(
                    f"Redimensionnement en "
                    f"{self.image_size[0]}x{self.image_size[1]}"
                )

                # --------------------------------------
                # ETAPE 5 : ENREGISTREMENT
                # --------------------------------------

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                image.save(
                    output_path,
                    format="PNG"
                )

                operations.append(
                    "Enregistrement de l'image prétraitée en PNG"
                )

            # ------------------------------------------
            # ETAPE 6 : METADONNEES
            # ------------------------------------------

            metadata = {
                "original_filename": input_path.name,
                "original_format": original_format,
                "original_color_mode": original_mode,
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": self.image_size[0],
                "processed_height": self.image_size[1],
                "processed_color_mode": "RGB",
                "operations": operations,
                "processing_date": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            # Chemin du fichier JSON
            metadata_path = output_path.with_suffix(".json")

            # Création du dossier si nécessaire
            metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Enregistrement JSON
            with open(
                metadata_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            # ------------------------------------------
            # ETAPE 7 : RESULTAT
            # ------------------------------------------

            return PreprocessingResult(
                success=True,
                input_path=str(input_path),
                output_path=str(output_path),
                operations=operations,
                metadata=metadata,
                error=None,
                metadata_path=str(metadata_path)
            )

        # ----------------------------------------------
        # GESTION DES ERREURS
        # ----------------------------------------------

        except (
            FileNotFoundError,
            ValueError,
            UnidentifiedImageError,
            OSError
        ) as error:

            return PreprocessingResult(
                success=False,
                input_path=str(input_path),
                output_path=None,
                operations=operations,
                metadata={},
                error=str(error),
                metadata_path=None
            )
