
"""
NeuroTrace - Integrity Verification Service

Vérifie l'intégrité des artefacts enregistrés :
1. Récupère les métadonnées depuis PostgreSQL.
2. Accède au fichier dans le stockage local.
3. Recalcule son SHA-256.
4. Compare le hash recalculé au hash enregistré.
"""

from pathlib import Path
from neurotrace.storage.database import SessionLocal
from neurotrace.storage.models import Artifact
from neurotrace.security.hashing import calculate_file_hash


class IntegrityService:

    def verify_artifact(self, artifact_id: str) -> dict:
        with SessionLocal() as session:

            artifact = session.get(Artifact, artifact_id)

            if artifact is None:
                raise ValueError(
                    f"Artefact introuvable : {artifact_id}"
                )

            # Récupérer les informations depuis PostgreSQL.
            expected_hash = artifact.sha256
            storage_reference = artifact.storage_reference
            artifact_type = artifact.artifact_type

            if not storage_reference:
                return {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "is_valid": False,
                    "error": "Référence de stockage absente."
                }

            file_path = Path(storage_reference)

            if not file_path.exists() or not file_path.is_file():
                return {
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "is_valid": False,
                    "error": "Fichier introuvable dans le stockage local."
                }

            # Recalculer le SHA-256.
            actual_hash = calculate_file_hash(file_path)

            # Comparer avec le hash enregistré.
            is_valid = actual_hash == expected_hash

            return {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
                "is_valid": is_valid,
                "error": None
            }

    def verify_case(self, case_id: str) -> list[dict]:
        """Vérifie tous les artefacts d'un dossier."""

        with SessionLocal() as session:
            artifacts = session.query(Artifact).filter_by(
                case_id=case_id
            ).order_by(Artifact.timestamp).all()

            artifact_ids = [
                artifact.artifact_id for artifact in artifacts
            ]

        if not artifact_ids:
            raise ValueError(
                f"Aucun artefact trouvé pour {case_id}"
            )

        results = []

        for artifact_id in artifact_ids:
            result = self.verify_artifact(artifact_id)
            results.append(result)

        return results
