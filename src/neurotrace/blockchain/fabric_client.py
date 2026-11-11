
import os
from datetime import datetime
from typing import Any

import requests


class FabricClient:
    """
    Client Python permettant de communiquer avec
    le serveur Node.js Fabric Gateway.
    """

    def __init__(self, gateway_url: str | None = None):
        self.gateway_url = (
            gateway_url
            or os.getenv(
                "FABRIC_GATEWAY_URL",
                "http://127.0.0.1:3001"
            )
        ).rstrip("/")

        self.timeout = 30

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Effectue une requête HTTP vers Fabric Gateway.
        """

        url = f"{self.gateway_url}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Impossible de contacter Fabric Gateway. "
                "Vérifiez que le serveur Node.js est démarré."
            ) from exc

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Le serveur Fabric Gateway ne répond pas dans le délai prévu."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            try:
                details = response.json()
            except ValueError:
                details = response.text

            raise RuntimeError(
                f"Erreur Fabric Gateway ({response.status_code}) : {details}"
            ) from exc

        except ValueError as exc:
            raise RuntimeError(
                "La réponse de Fabric Gateway n'est pas un JSON valide."
            ) from exc

    def health(self) -> dict:
        """
        Vérifie la disponibilité de Fabric Gateway.
        """
        return self._request("GET", "/health")

    def register_artifact(self, artifact) -> dict:
        """
        Enregistre un artefact dans Hyperledger Fabric.

        Accepte un objet Artifact SQLAlchemy ou un dictionnaire.
        """

        def get_value(name, default=None):
            if isinstance(artifact, dict):
                return artifact.get(name, default)
            return getattr(artifact, name, default)

        timestamp = get_value("timestamp")

        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif timestamp is not None:
            timestamp = str(timestamp)

        # Le chaincode exige un modelVersion non vide.
        model_version = get_value("model_version") or "N/A"

        payload = {
            "artifactId": get_value("artifact_id"),
            "caseId": get_value("case_id"),
            "artifactType": get_value("artifact_type"),
            "sha256": get_value("sha256"),
            "timestamp": timestamp,
            "modelVersion": model_version,
            "storageReference": get_value("storage_reference"),
        }

        parent_id = get_value("parent_artifact_id")

        if parent_id:
            payload["parentArtifactId"] = parent_id

        # Vérification des champs obligatoires avant l'envoi.
        required_fields = [
            "artifactId",
            "caseId",
            "artifactType",
            "sha256",
            "timestamp",
            "modelVersion",
            "storageReference",
        ]

        for field in required_fields:
            if payload.get(field) is None or payload.get(field) == "":
                raise ValueError(
                    f"Champ obligatoire manquant pour Fabric : {field}"
                )

        if len(payload["sha256"]) != 64:
            raise ValueError(
                "Le hash SHA-256 doit contenir 64 caractères."
            )

        return self._request(
            "POST",
            "/artifacts",
            json=payload
        )

    def get_artifact(self, artifact_id: str) -> dict:
        """
        Récupère un artefact depuis la blockchain.
        """
        return self._request(
            "GET",
            f"/artifacts/{artifact_id}"
        )

    def verify_hash(
        self,
        artifact_id: str,
        sha256: str
    ) -> dict:
        """
        Vérifie l'intégrité d'un artefact en comparant
        son hash avec celui enregistré sur Fabric.
        """
        return self._request(
            "POST",
            f"/artifacts/{artifact_id}/verify",
            json={"sha256": sha256}
        )

    def get_artifact_history(self, artifact_id: str) -> list:
        """
        Consulte l'historique blockchain d'un artefact.
        """
        return self._request(
            "GET",
            f"/artifacts/{artifact_id}/history"
        )


# Instance réutilisable dans le backend NeuroTrace.
fabric_client = FabricClient()
