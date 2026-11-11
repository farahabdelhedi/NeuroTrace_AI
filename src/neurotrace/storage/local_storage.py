
"""
NeuroTrace - Local File Storage

Gestion du stockage hors chaîne :
- Images originales.
- Images prétraitées.
- Résultats de prédiction.
- Explications Grad-CAM.
"""

import re
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
STORAGE_DIR = ROOT_DIR / "data" / "storage"


class LocalStorage:

    def __init__(self, storage_dir: Path = STORAGE_DIR):
        self.storage_dir = Path(storage_dir)

    def _validate_case_id(self, case_id: str):
        if not isinstance(case_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", case_id
        ):
            raise ValueError("Identifiant CASE_ID invalide.")

    def create_case_directories(self, case_id: str) -> dict:
        self._validate_case_id(case_id)

        case_dir = self.storage_dir / case_id

        directories = {
            "case": case_dir,
            "original": case_dir / "original",
            "processed": case_dir / "processed",
            "results": case_dir / "results",
            "explanations": case_dir / "explanations"
        }

        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return {
            name: str(path)
            for name, path in directories.items()
        }

    def _store_file(
        self,
        source_path: str,
        case_id: str,
        category: str
    ) -> str:

        source = Path(source_path)

        if not source.exists() or not source.is_file():
            raise FileNotFoundError(
                f"Fichier source introuvable : {source}"
            )

        directories = self.create_case_directories(case_id)

        if category not in (
            "original", "processed", "results", "explanations"
        ):
            raise ValueError("Catégorie de stockage invalide.")

        destination_dir = Path(directories[category])
        destination = destination_dir / source.name

        if source.resolve() == destination.resolve():
            return str(destination)

        shutil.copy2(source, destination)

        return str(destination)

    def store_original_image(self, source_path: str, case_id: str) -> str:
        return self._store_file(source_path, case_id, "original")

    def store_processed_image(self, source_path: str, case_id: str) -> str:
        return self._store_file(source_path, case_id, "processed")

    def store_result(self, source_path: str, case_id: str) -> str:
        return self._store_file(source_path, case_id, "results")

    def store_explanation(self, source_path: str, case_id: str) -> str:
        return self._store_file(source_path, case_id, "explanations")
