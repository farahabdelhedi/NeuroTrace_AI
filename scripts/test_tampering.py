"""
NeuroTrace - Test de détection des modifications

Vérifie qu'une modification volontaire d'un artefact
est détectée par son empreinte SHA-256.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.security.hashing import verify_data_integrity


CASE_ID = "CASE_001"

DATA_DIR = ROOT_DIR / "artifacts/predictions" / CASE_ID
METADATA_DIR = ROOT_DIR / "artifacts/security" / CASE_ID


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def test_modification(data_path, metadata_path, field, new_value):

    original_data = load_json(data_path)
    metadata = load_json(metadata_path)

    expected_hash = metadata["sha256"]

    # 1. Vérification avant modification
    initial_integrity = verify_data_integrity(
        original_data,
        expected_hash
    )

    print("\nIntégrité initiale :", initial_integrity)

    assert initial_integrity is True

    try:
        # 2. Modification volontaire
        modified_data = original_data.copy()
        modified_data[field] = new_value

        save_json(modified_data, data_path)

        # 3. Vérification après modification
        modified_integrity = verify_data_integrity(
            load_json(data_path),
            expected_hash
        )

        print("Intégrité après modification :", modified_integrity)

        assert modified_integrity is False

        print("Modification détectée avec succès !")

    finally:
        # 4. Restaurer le fichier original
        save_json(original_data, data_path)

    # 5. Vérifier que le fichier est restauré
    restored_integrity = verify_data_integrity(
        load_json(data_path),
        expected_hash
    )

    print("Intégrité après restauration :", restored_integrity)

    assert restored_integrity is True


def main():

    print("===== TEST DE MODIFICATION : PRÉDICTION =====")

    test_modification(
        DATA_DIR / "prediction.json",
        METADATA_DIR / "prediction.json",
        "confidence",
        0.0
    )

    print("\n===== TEST DE MODIFICATION : VALIDATION =====")

    test_modification(
        DATA_DIR / "validation.json",
        METADATA_DIR / "validation.json",
        "status",
        "Modification volontaire"
    )

    print("\nTous les tests de modification ont réussi !")


if __name__ == "__main__":
    main()