
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config.database_config import engine


class Base(DeclarativeBase):
    """Classe de base des modèles SQLAlchemy."""
    pass


# Fabrique de sessions pour communiquer avec PostgreSQL
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)
