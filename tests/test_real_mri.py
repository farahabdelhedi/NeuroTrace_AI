
from pathlib import Path

from src.neurotrace.preprocessing.processor import (
    MRIImageProcessor
)


# Chemin du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossier contenant les vraies images IRM
DATASET_DIR = BASE_DIR / "datasets" / "Testing"

# Dossier de sortie pour le test
OUTPUT_DIR = BASE_DIR / "artifacts" / "preprocessing_test"


def test_real_mri_image():

    # Récupérer une vraie image du dataset
    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(DATASET_DIR.rglob(extension))

    assert image_files, "Aucune image trouvée dans le dataset."

    # Choisir une seule image
    input_image = image_files[0]

    print("\nImage IRM sélectionnée :", input_image)

    # Définir le chemin de sortie
    output_image = OUTPUT_DIR / f"{input_image.stem}.png"

    # Lancer le prétraitement
    processor = MRIImageProcessor()

    result = processor.process_image(
        str(input_image),
        str(output_image)
    )

    # Vérifier le résultat
    assert result.success is True, result.error
    assert output_image.exists()
    assert Path(result.metadata_path).exists()

    print("\nPrétraitement réussi !")
    print("Image originale :", input_image)
    print("Image prétraitée :", output_image)
    print("Métadonnées JSON :", result.metadata_path)
    print("Opérations :", result.operations)
