
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.security.integrity_service import IntegrityService
from neurotrace.storage.database import engine


def main():

    service = IntegrityService()

    print("===== VÉRIFICATION D'INTÉGRITÉ =====")
    print("Dossier :", "CASE_001")

    results = service.verify_case("CASE_001")

    valid_count = 0
    invalid_count = 0

    for result in results:

        print("\nArtefact :", result["artifact_type"])
        print("ID :", result["artifact_id"])

        if result["error"]:
            print("Erreur :", result["error"])

        elif result["is_valid"]:
            print("SHA-256 :", result["actual_hash"])
            print("Statut : INTÉGRITÉ VÉRIFIÉE")
            valid_count += 1

        else:
            print("Hash enregistré :", result["expected_hash"])
            print("Hash recalculé :", result["actual_hash"])
            print("Statut : MODIFICATION DÉTECTÉE")
            invalid_count += 1

    print("\n===== BILAN =====")
    print("Artefacts vérifiés :", len(results))
    print("Intègres :", valid_count)
    print("Modifiés ou introuvables :", invalid_count)

    if invalid_count == 0:
        print("\nTous les artefacts sont intègres.")
    else:
        print("\nAttention : certains artefacts nécessitent une vérification.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Erreur :", error)
        raise
    finally:
        engine.dispose()
