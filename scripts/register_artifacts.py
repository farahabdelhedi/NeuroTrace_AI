
"""
NeuroTrace - Enregistrement des artefacts

Ce script calcule les empreintes SHA-256 de l'image
originale et de l'image prétraitée, puis crée leurs
métadonnées de traçabilité.
"""

"""
NeuroTrace - Enregistrement des artefacts
"""

import sys
from pathlib import Path

# Ajouter le dossier src au chemin de recherche de Python
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from neurotrace.security.hashing import (
    create_artifact_metadata,
    save_hash_metadata,
    verify_artifact_metadata
)


def main():

    # Identifiant pseudonyme du cas
    case_id = "CASE_001"

    # Chemins des deux images
    original_path = Path(
        "datasets/Testing/glioma/Te-gl_1.jpg"
    )

    processed_path = Path(
        "artifacts/preprocessing_test/Te-gl_1.png"
    )

    # Vérifier que les deux images existent
    if not original_path.exists():
        raise FileNotFoundError(
            f"Image originale introuvable : {original_path}"
        )

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Image prétraitée introuvable : {processed_path}"
        )

    # Dossier de stockage des métadonnées
    metadata_dir = Path("artifacts/security") / case_id

    # 1. Enregistrer l'image originale
    original_metadata = create_artifact_metadata(
        file_path=original_path,
        case_id=case_id,
        artifact_type="original_image",
        storage_reference=str(original_path)
    )

    save_hash_metadata(
        original_metadata,
        metadata_dir / "original_image.json"
    )

    # 2. Enregistrer l'image prétraitée
    processed_metadata = create_artifact_metadata(
        file_path=processed_path,
        case_id=case_id,
        artifact_type="preprocessed_image",
        storage_reference=str(processed_path),
        parent_artifact_id=original_metadata["artifact_id"]
    )

    save_hash_metadata(
        processed_metadata,
        metadata_dir / "preprocessed_image.json"
    )

    # 3. Vérifier l'intégrité des deux images
    original_valid = verify_artifact_metadata(
        original_path,
        original_metadata
    )

    processed_valid = verify_artifact_metadata(
        processed_path,
        processed_metadata
    )

    # 4. Afficher les résultats
    print("\n===== NEUROTRACE : TRAÇABILITÉ =====")

    print("\nIMAGE ORIGINALE")
    print("Artifact ID :", original_metadata["artifact_id"])
    print("SHA-256 :", original_metadata["sha256"])
    print("Intégrité :", original_valid)

    print("\nIMAGE PRÉTRAITÉE")
    print("Artifact ID :", processed_metadata["artifact_id"])
    print("SHA-256 :", processed_metadata["sha256"])
    print("Parent ID :", processed_metadata["parent_artifact_id"])
    print("Intégrité :", processed_valid)

    print("\nMétadonnées sauvegardées dans :", metadata_dir)


if __name__ == "__main__":
    main()
