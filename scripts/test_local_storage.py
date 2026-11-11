
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.storage.local_storage import LocalStorage


def test_local_storage():

    storage = LocalStorage()

    original_image = (
        ROOT_DIR / "datasets" / "Testing"
        / "glioma" / "Te-gl_1.jpg"
    )

    processed_image = (
        ROOT_DIR / "artifacts" / "preprocessing_test"
        / "Te-gl_1.png"
    )

    print("===== TEST DU STOCKAGE LOCAL =====")

    # Vérifier les fichiers sources
    assert original_image.exists(), "Image originale introuvable."
    assert processed_image.exists(), "Image prétraitée introuvable."

    # Créer les dossiers du cas
    directories = storage.create_case_directories("CASE_001")

    print("Dossiers créés :")
    for name, path in directories.items():
        print(f"{name} : {path}")

    # Copier les images
    original_path = storage.store_original_image(
        str(original_image), "CASE_001"
    )

    processed_path = storage.store_processed_image(
        str(processed_image), "CASE_001"
    )

    # Vérifier les copies
    assert Path(original_path).exists()
    assert Path(processed_path).exists()

    # Vérifier que les fichiers sources existent toujours
    assert original_image.exists()
    assert processed_image.exists()

    print("\nImage originale stockée :", original_path)
    print("Image prétraitée stockée :", processed_path)

    print("\nTous les tests du stockage local ont réussi !")


if __name__ == "__main__":
    test_local_storage()
