
"""
NeuroTrace - Module de sécurité

Responsabilités :
- Calculer les empreintes SHA-256 des fichiers.
- Calculer les empreintes SHA-256 des données structurées.
- Vérifier l'intégrité des fichiers et des données.
"""

import hashlib
import json

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4


def calculate_file_hash(file_path: str | Path) -> str:
    """
    Calcule le hash SHA-256 d'un fichier.

    Lecture par blocs pour éviter de charger les gros fichiers
    entièrement en mémoire.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    if not file_path.is_file():
        raise ValueError("Le chemin fourni ne correspond pas à un fichier.")

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_data_hash(data: Any) -> str:
    """
    Calcule le hash SHA-256 de données structurées.

    Les données sont converties en JSON de manière déterministe
    afin d'obtenir la même empreinte pour un même contenu.
    """
    try:
        serialized_data = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Impossible de convertir les données en JSON : {error}"
        ) from error

    return hashlib.sha256(
        serialized_data.encode("utf-8")
    ).hexdigest()


def verify_file_integrity(
    file_path: str | Path,
    expected_hash: str
) -> bool:
    """
    Vérifie si le hash actuel d'un fichier correspond
    au hash enregistré précédemment.
    """
    actual_hash = calculate_file_hash(file_path)
    return actual_hash == expected_hash


def verify_data_integrity(
    data: Any,
    expected_hash: str
) -> bool:
    """
    Vérifie l'intégrité de données structurées.
    """
    actual_hash = calculate_data_hash(data)
    return actual_hash == expected_hash


def create_artifact_metadata(
    file_path: str | Path,
    case_id: str,
    artifact_type: str,
    storage_reference: str | None = None,
    model_version: str | None = None,
    parent_artifact_id: str | None = None
) -> dict:
    """
    Crée les métadonnées de traçabilité d'un artefact NeuroTrace.
    """

    if not case_id.strip():
        raise ValueError("Le case_id ne peut pas être vide.")

    if not artifact_type.strip():
        raise ValueError("Le type d'artefact ne peut pas être vide.")

    file_path = Path(file_path)
    file_hash = calculate_file_hash(file_path)

    return {
        "artifact_id": str(uuid4()),
        "case_id": case_id,
        "artifact_type": artifact_type,
        "sha256": file_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "parent_artifact_id": parent_artifact_id,
        "storage_reference": storage_reference or file_path.name
    }


def save_hash_metadata(
    metadata: dict,
    output_path: str | Path
) -> str:
    """
    Enregistre les métadonnées dans un fichier JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    return str(output_path)


def verify_artifact_metadata(
    file_path: str | Path,
    metadata: dict
) -> bool:
    """
    Vérifie si un fichier correspond au hash enregistré
    dans ses métadonnées.
    """

    expected_hash = metadata.get("sha256")

    if not expected_hash:
        raise ValueError("Les métadonnées ne contiennent pas de hash SHA-256.")

    return verify_file_integrity(file_path, expected_hash)
def create_data_artifact_metadata(
    data: dict,
    case_id: str,
    artifact_type: str,
    storage_reference: str | None = None,
    model_version: str | None = None,
    parent_artifact_id: str | None = None
) -> dict:
    """
    Crée les métadonnées d'un artefact de données structurées.
    """

    from uuid import uuid4
    from datetime import datetime, timezone

    return {
        "artifact_id": str(uuid4()),
        "case_id": case_id,
        "artifact_type": artifact_type,
        "sha256": calculate_data_hash(data),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "parent_artifact_id": parent_artifact_id,
        "storage_reference": storage_reference
    }
