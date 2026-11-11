
"""
NeuroTrace - Artifact Storage Service

Coordonne :
- Le stockage physique des fichiers.
- Le calcul des empreintes SHA-256.
- L'enregistrement des métadonnées dans PostgreSQL.
"""

from pathlib import Path
from uuid import uuid4

from neurotrace.storage.database import SessionLocal
from neurotrace.storage.models import Case, Artifact
from neurotrace.storage.local_storage import LocalStorage
from neurotrace.security.hashing import calculate_file_hash
from neurotrace.blockchain.fabric_client import fabric_client


class ArtifactStorageService:

    def __init__(self):
        self.storage = LocalStorage()

    def register_artifact(
        self,
        case_id: str,
        source_path: str,
        artifact_type: str,
        category: str,
        parent_artifact_id: str | None = None,
        model_version: str | None = None
    ) -> Artifact:

        source = Path(source_path)

        if not source.exists() or not source.is_file():
            raise FileNotFoundError(
                f"Fichier introuvable : {source}"
            )

        if category == "original":
            stored_path = self.storage.store_original_image(
                str(source), case_id
            )

        elif category == "processed":
            stored_path = self.storage.store_processed_image(
                str(source), case_id
            )

        elif category == "results":
            stored_path = self.storage.store_result(
                str(source), case_id
            )

        elif category == "explanations":
            stored_path = self.storage.store_explanation(
                str(source), case_id
            )

        else:
            raise ValueError("Catégorie de stockage inconnue.")

        file_hash = calculate_file_hash(stored_path)

        with SessionLocal() as session:
            try:
                case = session.get(Case, case_id)

                if case is None:
                    case = Case(case_id=case_id)
                    session.add(case)
                    session.flush()

                artifact = Artifact(
                    artifact_id=str(uuid4()),
                    case_id=case_id,
                    artifact_type=artifact_type,
                    sha256=file_hash,
                    model_version=model_version,
                    parent_artifact_id=parent_artifact_id,
                    storage_reference=stored_path
                )


                session.add(artifact)
                session.commit()
                session.refresh(artifact)

                # Enregistrement dans Hyperledger Fabric
                try:
                    blockchain_result = fabric_client.register_artifact(artifact)

                    # Mise à jour du statut blockchain
                    artifact.details = {
                        "blockchain": {
                            "status": "registered",
                            "message": blockchain_result.get(
                                "message",
                                "Artefact enregistré sur Fabric."
                            )
                        }
                    }

                    session.commit()
                    session.refresh(artifact)

                except Exception as blockchain_error:
                    # L'artefact reste enregistré dans PostgreSQL.
                    artifact.details = {
                        "blockchain": {
                            "status": "pending",
                            "message": str(blockchain_error)
                        }
                    }

                    session.commit()
                    print(
                        f"[Fabric] Échec de synchronisation de "
                        f"{artifact.artifact_id} : {blockchain_error}"
                    )

                return artifact


            except Exception:
                session.rollback()
                raise
