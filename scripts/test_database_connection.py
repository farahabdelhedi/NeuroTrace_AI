
import sys
from pathlib import Path

# Ajouter la racine du projet au chemin de recherche Python
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import text
from config.database_config import engine


try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()"))
        version = result.scalar()

        print("Connexion PostgreSQL réussie !")
        print("Base de données : neurotrace_db")
        print("Version :", version)

except Exception as error:
    print("Échec de connexion à PostgreSQL.")
    print("Erreur :", error)

finally:
    engine.dispose()
