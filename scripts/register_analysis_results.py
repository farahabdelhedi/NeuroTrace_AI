
"""
NeuroTrace - Enregistrement des résultats IA dans PostgreSQL

Enregistre :
- la prédiction ;
- la validation ;
- l'explication Grad-CAM ;
- la heatmap ;
- l'image avec superposition Grad-CAM.

Les fichiers restent hors chaîne.
PostgreSQL conserve leurs hashes et leurs métadonnées.
"""

import sys
import shutil
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.storage.database import SessionLocal, engine
from neurotrace.storage.models import Artifact
from neurotrace.security.hashing import calculate_file_hash


CASE_ID = "CASE_001"
MODEL_VERSION = "Abuzaid01/brain-tumor-resnet50-classifier"

PREDICTION_FILE = (
    ROOT_DIR / "artifacts" / "predictions"
    / CASE_ID / "prediction.json"
)

VALIDATION_FILE = (
    ROOT_DIR / "artifacts" / "predictions"
    / CASE_ID / "validation.json"
)

EXPLANATION_FILE = (
    ROOT_DIR / "artifacts" / "explanations"
    / "Te-gl_1_gradcam.json"
)

HEATMAP_FILE = (
    ROOT_DIR / "artifacts" / "explanations"
    / "Te-gl_1_heatmap.png"
)

OVERLAY_FILE = (
    ROOT_DIR / "artifacts" / "explanations"
    / "Te-gl_1_gradcam.png"
)


def get_artifact(session, artifact_type):
    """Récupère un artefact déjà enregistré pour ce cas."""

    return session.query(Artifact).filter_by(
        case_id=CASE_ID,
        artifact_type=artifact_type
    ).first()


def register_result(
    session,
    source_path,
    artifact_type,
    destination_category,
    parent_artifact_id=None,
    model_version=None
):
    """Copie un fichier et enregistre ses métadonnées."""

    source_path = Path(source_path)

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(
            f"Fichier source introuvable : {source_path}"
        )

    # Éviter de créer des doublons lors d'une nouvelle exécution.
    existing = get_artifact(session, artifact_type)

    if existing:
        print(f"\nDéjà enregistré : {artifact_type}")
        return existing

    destination_dir = (
        ROOT_DIR / "data" / "storage" / CASE_ID
        / destination_category
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / source_path.name
    shutil.copy2(source_path, destination)

    file_hash = calculate_file_hash(destination)

    artifact = Artifact(
        artifact_id=str(uuid4()),
        case_id=CASE_ID,
        artifact_type=artifact_type,
        sha256=file_hash,
        parent_artifact_id=parent_artifact_id,
        model_version=model_version,
        storage_reference=str(destination)
    )

    session.add(artifact)
    session.flush()

    print(f"\nArtefact : {artifact_type}")
    print("ID :", artifact.artifact_id)
    print("SHA-256 :", artifact.sha256)
    print("Parent ID :", artifact.parent_artifact_id)
    print("Stockage :", artifact.storage_reference)

    return artifact


def main():
    print("===== ENREGISTREMENT DES RÉSULTATS =====")

    with SessionLocal() as session:
        try:
            # Vérifier les artefacts précédents.
            processed = get_artifact(
                session, "preprocessed_image"
            )
            if processed is None:
                raise ValueError(
                    "Image prétraitée absente de PostgreSQL. "
                    "Exécute d'abord le test d'intégration précédent."
                )

            # 1. Prédiction : enfant de l'image prétraitée.
            prediction = register_result(
                session,
                PREDICTION_FILE,
                "prediction",
                "results",
                parent_artifact_id=processed.artifact_id,
                model_version=MODEL_VERSION
            )

            # 2. Validation : enfant de la prédiction.
            validation = register_result(
                session,
                VALIDATION_FILE,
                "validation",
                "results",
                parent_artifact_id=prediction.artifact_id,
                model_version=MODEL_VERSION
            )

            # 3. Explication JSON : enfant de la validation.
            explanation = register_result(
                session,
                EXPLANATION_FILE,
                "explanation",
                "explanations",
                parent_artifact_id=validation.artifact_id,
                model_version=MODEL_VERSION
            )

            # 4. Heatmap : enfant de l'explication.
            heatmap = register_result(
                session,
                HEATMAP_FILE,
                "gradcam_heatmap",
                "explanations",
                parent_artifact_id=explanation.artifact_id,
                model_version=MODEL_VERSION
            )

            # 5. Image Grad-CAM : enfant de l'explication.
            overlay = register_result(
                session,
                OVERLAY_FILE,
                "gradcam_overlay",
                "explanations",
                parent_artifact_id=explanation.artifact_id,
                model_version=MODEL_VERSION
            )

            session.commit()

            print("\n===== ENREGISTREMENT TERMINÉ =====")
            print("Les cinq artefacts sont enregistrés dans PostgreSQL.")

        except Exception:
            session.rollback()
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERREUR :", error)
        raise
    finally:
        engine.dispose()
