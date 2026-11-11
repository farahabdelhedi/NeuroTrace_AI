
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.storage.artifact_service import (
    ArtifactStorageService
)

from neurotrace.storage.database import engine


def main():

    service = ArtifactStorageService()

    original_image = (
        ROOT_DIR / "datasets" / "Testing"
        / "glioma" / "Te-gl_1.jpg"
    )

    processed_image = (
        ROOT_DIR / "artifacts" / "preprocessing_test"
        / "Te-gl_1.png"
    )

    print("===== INTEGRATION STOCKAGE + POSTGRESQL =====")

    # 1. Enregistrer l'image originale
    original = service.register_artifact(
        case_id="CASE_001",
        source_path=str(original_image),
        artifact_type="original_image",
        category="original"
    )

    print("\nIMAGE ORIGINALE")
    print("Artifact ID :", original.artifact_id)
    print("SHA-256 :", original.sha256)
    print("Chemin :", original.storage_reference)

    # 2. Enregistrer l'image prétraitée
    processed = service.register_artifact(
        case_id="CASE_001",
        source_path=str(processed_image),
        artifact_type="preprocessed_image",
        category="processed",
        parent_artifact_id=original.artifact_id
    )

    print("\nIMAGE PRÉTRAITÉE")
    print("Artifact ID :", processed.artifact_id)
    print("SHA-256 :", processed.sha256)
    print("Parent ID :", processed.parent_artifact_id)
    print("Chemin :", processed.storage_reference)

    print("\nIntégration réussie !")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Erreur :", error)
        raise
    finally:
        engine.dispose()
