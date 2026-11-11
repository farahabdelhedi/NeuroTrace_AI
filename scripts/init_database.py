
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from neurotrace.storage.database import Base, engine
from neurotrace.storage.models import Case, Artifact


def initialize_database():
    print("Initialisation de la base NeuroTrace...")

    Base.metadata.create_all(bind=engine)

    print("Tables créées avec succès !")
    print(" - cases")
    print(" - artifacts")


if __name__ == "__main__":
    try:
        initialize_database()
    except Exception as error:
        print("Erreur :", error)
    finally:
        engine.dispose()
