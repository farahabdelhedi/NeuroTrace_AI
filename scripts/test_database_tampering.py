
"""
NeuroTrace - Test de détection de modification

1. Sélectionne un artefact enregistré dans PostgreSQL.
2. Vérifie son intégrité initiale.
3. Modifie volontairement son contenu.
4. Vérifie que la modification est détectée.
5. Restaure exactement le fichier original.
6. Vérifie à nouveau son intégrité.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.storage.database import SessionLocal, engine
from neurotrace.storage.models import Artifact
from neurotrace.security.integrity_service import IntegrityService


def main():

    case_id = "CASE_001"
    artifact_type = "prediction"

    service = IntegrityService()

    original_content = None
    file_path = None

    print("===== TEST DE MODIFICATION =====")

    try:
        # 1. Récupérer l'artefact depuis PostgreSQL.
        with SessionLocal() as session:
            artifact = session.query(Artifact).filter_by(
                case_id=case_id,
                artifact_type=artifact_type
            ).order_by(Artifact.timestamp.desc()).first()

            if artifact is None:
                raise ValueError("Artefact prediction introuvable.")

            artifact_id = artifact.artifact_id
            file_path = Path(artifact.storage_reference)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        # 2. Sauvegarder exactement le contenu original.
        original_content = file_path.read_bytes()

        initial = service.verify_artifact(artifact_id)

        if not initial["is_valid"]:
            raise RuntimeError(
                "L'artefact n'est pas intègre avant le test. "
                "Arrêt pour éviter de masquer une anomalie."
            )

        print("\n1. Vérification initiale")
        print("Intégrité :", initial["is_valid"])

        # 3. Modifier volontairement le fichier.
        file_path.write_bytes(
            original_content + b"\nTEST_MODIFICATION_NEUROTRACE"
        )

        print("\n2. Modification volontaire effectuée.")

        # 4. Vérifier que la modification est détectée.
        modified = service.verify_artifact(artifact_id)

        print("\n3. Vérification après modification")
        print("Intégrité :", modified["is_valid"])

        if modified["is_valid"]:
            raise AssertionError(
                "ERREUR : la modification n'a pas été détectée."
            )

        print("MODIFICATION DÉTECTÉE !")

    finally:
        # 5. Restaurer le fichier même si une erreur survient.
        if original_content is not None and file_path is not None:
            file_path.write_bytes(original_content)

            print("\n4. Fichier original restauré.")

            # 6. Vérifier l'intégrité après restauration.
            restored = service.verify_artifact(artifact_id)

            print("\n5. Vérification après restauration")
            print("Intégrité :", restored["is_valid"])

            if not restored["is_valid"]:
                raise RuntimeError(
                    "La restauration a échoué : "
                    "l'intégrité n'est pas rétablie."
                )

            print("INTÉGRITÉ RÉTABLIE !")

        engine.dispose()


if __name__ == "__main__":
    main()
