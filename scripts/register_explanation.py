
"""
NeuroTrace - Enregistrement et traçabilité Grad-CAM
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.security.hashing import (
    create_artifact_metadata,
    create_data_artifact_metadata,
    save_hash_metadata,
    verify_artifact_metadata,
    verify_data_integrity
)

CASE_ID = "CASE_001"

EXPLANATION_DIR = ROOT_DIR / "artifacts" / "explanations"
SECURITY_DIR = ROOT_DIR / "artifacts" / "security" / CASE_ID

EXPLANATION_JSON = EXPLANATION_DIR / "Te-gl_1_gradcam.json"
HEATMAP_IMAGE = EXPLANATION_DIR / "Te-gl_1_heatmap.png"
GRADCAM_IMAGE = EXPLANATION_DIR / "Te-gl_1_gradcam.png"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("===== ENREGISTREMENT DE L'EXPLICATION GRAD-CAM =====")

    # Vérifier que les fichiers existent
    required_files = [
        EXPLANATION_JSON,
        HEATMAP_IMAGE,
        GRADCAM_IMAGE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier Grad-CAM introuvable : {path}"
            )

    SECURITY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Récupérer l'identifiant de la validation
    validation_metadata = load_json(
        SECURITY_DIR / "validation.json"
    )

    validation_id = validation_metadata["artifact_id"]

    # 2. Charger les données de l'explication
    explanation_data = load_json(EXPLANATION_JSON)

    # 3. Enregistrer les données JSON de Grad-CAM
    explanation_metadata = create_data_artifact_metadata(
        data=explanation_data,
        case_id=CASE_ID,
        artifact_type="gradcam_explanation",
        storage_reference=str(EXPLANATION_JSON.relative_to(ROOT_DIR)),
        model_version="ResNet50",
        parent_artifact_id=validation_id
    )

    save_hash_metadata(
        explanation_metadata,
        SECURITY_DIR / "explanation.json"
    )

    print("\nEXPLICATION JSON")
    print("Artifact ID :", explanation_metadata["artifact_id"])
    print("SHA-256 :", explanation_metadata["sha256"])
    print("Parent ID :", explanation_metadata["parent_artifact_id"])

    # 4. Enregistrer la heatmap
    heatmap_metadata = create_artifact_metadata(
        file_path=HEATMAP_IMAGE,
        case_id=CASE_ID,
        artifact_type="gradcam_heatmap",
        storage_reference=str(HEATMAP_IMAGE.relative_to(ROOT_DIR)),
        model_version="ResNet50",
        parent_artifact_id=explanation_metadata["artifact_id"]
    )

    save_hash_metadata(
        heatmap_metadata,
        SECURITY_DIR / "gradcam_heatmap.json"
    )

    print("\nHEATMAP")
    print("Artifact ID :", heatmap_metadata["artifact_id"])
    print("SHA-256 :", heatmap_metadata["sha256"])
    print("Parent ID :", heatmap_metadata["parent_artifact_id"])

    # 5. Enregistrer l'image avec superposition Grad-CAM
    overlay_metadata = create_artifact_metadata(
        file_path=GRADCAM_IMAGE,
        case_id=CASE_ID,
        artifact_type="gradcam_overlay",
        storage_reference=str(GRADCAM_IMAGE.relative_to(ROOT_DIR)),
        model_version="ResNet50",
        parent_artifact_id=explanation_metadata["artifact_id"]
    )

    save_hash_metadata(
        overlay_metadata,
        SECURITY_DIR / "gradcam_overlay.json"
    )

    print("\nIMAGE GRAD-CAM")
    print("Artifact ID :", overlay_metadata["artifact_id"])
    print("SHA-256 :", overlay_metadata["sha256"])
    print("Parent ID :", overlay_metadata["parent_artifact_id"])

    # 6. Vérifier l'intégrité des artefacts
    print("\n===== VÉRIFICATION DE L'INTÉGRITÉ =====")

    explanation_ok = verify_data_integrity(
        explanation_data,
        explanation_metadata["sha256"]
    )

    heatmap_ok = verify_artifact_metadata(
        HEATMAP_IMAGE,
        heatmap_metadata
    )

    overlay_ok = verify_artifact_metadata(
        GRADCAM_IMAGE,
        overlay_metadata
    )

    print("Explication JSON :", explanation_ok)
    print("Heatmap :", heatmap_ok)
    print("Image Grad-CAM :", overlay_ok)

    assert explanation_ok
    assert heatmap_ok
    assert overlay_ok

    print("\nTous les artefacts Grad-CAM sont enregistrés et vérifiés !")


if __name__ == "__main__":
    main()
