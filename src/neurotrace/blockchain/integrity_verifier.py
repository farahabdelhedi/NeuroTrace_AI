
"""
NeuroTrace - Integrity Verifier

Vérifie l'intégrité d'un artefact stocké hors blockchain
en comparant son SHA-256 actuel avec celui enregistré
dans Hyperledger Fabric.
"""

from pathlib import Path

from neurotrace.blockchain.fabric_client import fabric_client
from neurotrace.security.hashing import calculate_file_hash


class IntegrityVerifier:

    def __init__(self):
        self.fabric = fabric_client

    def verify_artifact(self, artifact):
        """
        Vérifie l'intégrité d'un artefact provenant de PostgreSQL.
        """

        # 1. Vérifier la référence du fichier
        if not artifact.storage_reference:
            return {
                "success": False,
                "status": "REFERENCE_MANQUANTE",
                "message": "Aucune référence de stockage disponible.",
                "artifact_id": artifact.artifact_id
            }

        file_path = Path(artifact.storage_reference)

        if not file_path.exists() or not file_path.is_file():
            return {
                "success": False,
                "status": "FICHIER_INTROUVABLE",
                "message": "Le fichier stocké est introuvable.",
                "artifact_id": artifact.artifact_id
            }

        # 2. Recalculer le hash du fichier
        current_hash = calculate_file_hash(file_path)

        # 3. Comparer avec le hash enregistré dans Fabric
        try:
            blockchain_result = self.fabric.verify_hash(
                artifact.artifact_id,
                current_hash
            )
        except Exception as error:
            return {
                "success": False,
                "status": "ERREUR_BLOCKCHAIN",
                "message": f"Impossible de vérifier dans Fabric : {error}",
                "artifact_id": artifact.artifact_id
            }

        # 4. Interpréter le résultat
        integrity_valid = blockchain_result.get(
            "integrityValid", False
        )

        return {
            "success": True,
            "artifact_id": artifact.artifact_id,
            "case_id": artifact.case_id,
            "original_hash": blockchain_result.get("expectedHash"),
            "current_hash": current_hash,
            "integrity_valid": integrity_valid,
            "status": (
                "INTEGRITE_VERIFIEE"
                if integrity_valid
                else "MODIFICATION_DETECTEE"
            ),
            "message": (
                "Intégrité vérifiée : le fichier n'a pas été modifié."
                if integrity_valid
                else "Modification détectée : le hash du fichier est différent."
            )
        }


integrity_verifier = IntegrityVerifier()
