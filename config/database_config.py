
import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine

# Charger les variables du fichier .env
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Récupérer les paramètres PostgreSQL
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "neurotrace_db")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not DB_USER or not DB_PASSWORD:
    raise ValueError(
        "DB_USER ou DB_PASSWORD manquant dans le fichier .env"
    )

# Construire l'URL de connexion sans problème
# avec les caractères spéciaux du mot de passe
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

# Créer le moteur SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)
