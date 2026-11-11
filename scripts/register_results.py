"""
NeuroTrace - Enregistrement des résultats IA

Responsabilités :
- Exécuter la prédiction IA.
- Valider la prédiction.
- Sauvegarder les résultats en JSON.
- Calculer les empreintes SHA-256.
- Relier les artefacts avec leurs parents.
- Vérifier l'intégrité des résultats.
"""

import sys
import json
from pathlib import Path
from dataclasses import asdict

# Ajouter les chemins du projet
ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.ai.predictor import MRIPredictor
from neurotrace.validation.validator import PredictionValidator

from neurotrace.security.hashing import (
    create_data_artifact_metadata,
    save_hash_metadata,
    verify_data_integrity
)


def save_json(data: dict, output_path: Path):
    """Sauvegarde des données structurées en JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def main():

    case_id = "CASE_001"

    # 1. Chemins
    processed_path = (
        ROOT_DIR / "artifacts/preprocessing_test/Te-gl_1.png"
    )

    security_dir = ROOT_DIR / "artifacts/security" / case_id

    prediction_path = (
        ROOT_DIR / "artifacts/predictions" /
        case_id / "prediction.json"
    )

    validation_path = (
        ROOT_DIR / "artifacts/predictions" /
        case_id / "validation.json"
    )

    # 2. Vérifier les fichiers nécessaires
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Image prétraitée introuvable : {processed_path}"
        )

    preprocessed_metadata_path = (
        security_dir / "preprocessed_image.json"
    )

    if not preprocessed_metadata_path.exists():
        raise FileNotFoundError(
            "Métadonnées de l'image prétraitée introuvables."
        )

    with preprocessed_metadata_path.open(
        "r", encoding="utf-8"
    ) as file:
        preprocessed_metadata = json.load(file)

    # 3. Exécuter la prédiction IA
    print("\n===== ANALYSE IA =====")

    predictor = MRIPredictor()
    prediction = predictor.predict(str(processed_path))

    prediction_data = asdict(prediction)

    print("Classe prédite :", prediction.predicted_class)
    print("Confiance :", prediction.confidence, "%")
    print("Scores :", prediction.class_scores)
    print("Modèle :", prediction.model_id)

    # 4. Valider la prédiction
    print("\n===== VALIDATION =====")

    validator = PredictionValidator()
    validation = validator.validate(prediction)

    validation_data = asdict(validation)

    print("Statut :", validation.status)
    print("Valide :", validation.is_valid)
    print("Raisons :", validation.reasons)

    # 5. Sauvegarder les résultats
    save_json(prediction_data, prediction_path)
    save_json(validation_data, validation_path)

    # 6. Créer les métadonnées de la prédiction
    prediction_metadata = create_data_artifact_metadata(
        data=prediction_data,
        case_id=case_id,
        artifact_type="prediction",
        storage_reference=str(prediction_path.relative_to(ROOT_DIR)),
        model_version=prediction.model_id,
        parent_artifact_id=preprocessed_metadata["artifact_id"]
    )

    save_hash_metadata(
        prediction_metadata,
        security_dir / "prediction.json"
    )

    # 7. Créer les métadonnées de la validation
    validation_metadata = create_data_artifact_metadata(
        data=validation_data,
        case_id=case_id,
        artifact_type="validation",
        storage_reference=str(validation_path.relative_to(ROOT_DIR)),
        model_version=prediction.model_id,
        parent_artifact_id=prediction_metadata["artifact_id"]
    )

    save_hash_metadata(
        validation_metadata,
        security_dir / "validation.json"
    )

    # 8. Vérifier l'intégrité des données sauvegardées
    with prediction_path.open("r", encoding="utf-8") as file:
        saved_prediction = json.load(file)

    with validation_path.open("r", encoding="utf-8") as file:
        saved_validation = json.load(file)

    prediction_valid = verify_data_integrity(
        saved_prediction,
        prediction_metadata["sha256"]
    )

    validation_valid = verify_data_integrity(
        saved_validation,
        validation_metadata["sha256"]
    )

    # 9. Afficher les résultats
    print("\n===== TRAÇABILITÉ DES RÉSULTATS =====")

    print("\nPRÉDICTION")
    print("Artifact ID :", prediction_metadata["artifact_id"])
    print("SHA-256 :", prediction_metadata["sha256"])
    print("Parent ID :", prediction_metadata["parent_artifact_id"])
    print("Intégrité :", prediction_valid)

    print("\nVALIDATION")
    print("Artifact ID :", validation_metadata["artifact_id"])
    print("SHA-256 :", validation_metadata["sha256"])
    print("Parent ID :", validation_metadata["parent_artifact_id"])
    print("Intégrité :", validation_valid)

    print("\nRésultats sauvegardés dans :")
    print(prediction_path)
    print(validation_path)

    print("\nMétadonnées sauvegardées dans :")
    print(security_dir)


if __name__ == "__main__":
    main()