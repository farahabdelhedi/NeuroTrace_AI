
import sys
import re
import json
import tempfile

from pathlib import Path
from uuid import uuid4
from dataclasses import asdict
from datetime import datetime, timezone

# ============================================================
# 1. CONFIGURATION DU CHEMIN DU PROJET
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 2. IMPORTATION DES BIBLIOTHEQUES
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import desc

from neurotrace.ai.predictor import MRIPredictor
from neurotrace.preprocessing.processor import MRIImageProcessor
from neurotrace.storage.artifact_service import ArtifactStorageService
from neurotrace.storage.database import SessionLocal
from neurotrace.storage.models import Case, Artifact
from neurotrace.validation.validator import PredictionValidator
from neurotrace.ai.predictor import PredictionResult
from neurotrace.explainability.gradcam import GradCAM
from neurotrace.blockchain.integrity_verifier import integrity_verifier
from neurotrace.blockchain.fabric_client import fabric_client


# ============================================================
# 3. INITIALISATION DE FASTAPI ET DES SERVICES
# ============================================================

app = FastAPI(
    title="NeuroTrace API",
    description="API de détection et de traçabilité des tumeurs cérébrales",
    version="1.0.0"
)

image_processor = MRIImageProcessor()
artifact_service = ArtifactStorageService()

# Le modèle sera chargé une seule fois, à la première prédiction.
predictor = None
prediction_validator = PredictionValidator()
gradcam_generator = None


def get_predictor():
    global predictor

    if predictor is None:
        print("Initialisation du modèle IA ResNet50...")
        predictor = MRIPredictor()

    return predictor


# ============================================================
# 4. ROUTES PRINCIPALES
# ============================================================

@app.get("/")
def accueil():
    return {
        "message": "Bienvenue dans l'API NeuroTrace",
        "status": "running"
    }


@app.get("/health")
def verifier_etat():
    return {
        "status": "healthy",
        "application": "NeuroTrace"
    }


# ============================================================
# 5. GENERATION AUTOMATIQUE DE L'IDENTIFIANT DU CAS
# ============================================================

def generate_case_id():
    """Génère un identifiant CASE unique à partir des cas existants."""

    with SessionLocal() as session:
        existing_ids = session.query(Case.case_id).all()

    numbers = []

    for row in existing_ids:
        match = re.fullmatch(r"CASE_(\d+)", row[0])

        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers, default=0) + 1

    return f"CASE_{next_number:03d}"


# ============================================================
# 6. UPLOAD ET PRETRAITEMENT DE L'IMAGE IRM
# ============================================================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Reçoit une image IRM, la prétraite et enregistre ses artefacts."""

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Format non autorisé. Utilisez JPG, PNG, BMP ou TIFF."
        )

    # Lecture et vérification du fichier
    content = await file.read(20 * 1024 * 1024 + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé est vide."
        )

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Le fichier dépasse la limite de 20 Mo."
        )

    case_id = generate_case_id()

    try:
        with tempfile.TemporaryDirectory(
            prefix="neurotrace_"
        ) as temp_dir:

            original_path = Path(temp_dir) / f"original{extension}"
            processed_path = Path(temp_dir) / "processed.png"

            original_path.write_bytes(content)

            # 1. Prétraitement
            result = image_processor.process_image(
                str(original_path),
                str(processed_path)
            )

            if not result.success:
                raise HTTPException(
                    status_code=400,
                    detail=getattr(
                        result, "error", "Image IRM invalide."
                    )
                )

            # 2. Enregistrement de l'image originale
            original_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=str(original_path),
                artifact_type="image_originale",
                category="original"
            )

            # 3. Enregistrement de l'image prétraitée
            processed_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=str(processed_path),
                artifact_type="image_pretraitee",
                category="processed",
                parent_artifact_id=original_artifact.artifact_id
            )

        return {
            "success": True,
            "message": "Image IRM chargée et prétraitée avec succès.",
            "case_id": case_id,
            "preprocessing": {
                "status": "completed",
                "operations": [
                    "Validation de l'image",
                    "Conversion RGB",
                    "Redimensionnement 224x224",
                    "Export au format PNG"
                ]
            },
            "artifacts": {
                "original": {
                    "artifact_id": original_artifact.artifact_id,
                    "sha256": original_artifact.sha256,
                    "storage_reference": original_artifact.storage_reference
                },
                "processed": {
                    "artifact_id": processed_artifact.artifact_id,
                    "sha256": processed_artifact.sha256,
                    "storage_reference": processed_artifact.storage_reference
                }
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Erreur pendant le prétraitement : {error}")

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue pendant le traitement de l'image."
        ) from error


# ============================================================
# 7. CLASSIFICATION IA AVEC RESNET50
# ============================================================

@app.post("/cases/{case_id}/analyze")
def analyze_case(case_id: str):
    """Classe une image IRM prétraitée avec ResNet50."""

    # 1. Récupérer le dernier artefact prétraité du cas
    with SessionLocal() as session:

        processed_artifact = (
            session.query(Artifact)
            .filter(
                Artifact.case_id == case_id,
                Artifact.artifact_type == "image_pretraitee"
            )
            .order_by(desc(Artifact.timestamp))
            .first()
        )

        # Copier les informations avant la fermeture de la session
        if processed_artifact is not None:
            processed_artifact_id = processed_artifact.artifact_id
            processed_storage_reference = processed_artifact.storage_reference
        else:
            processed_artifact_id = None
            processed_storage_reference = None

    # 2. Vérifier l'existence de l'artefact
    if processed_artifact_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune image prétraitée trouvée pour {case_id}."
        )

    image_path = Path(processed_storage_reference)

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Le fichier de l'image prétraitée est introuvable."
        )

    try:
        # 3. Charger le prédicteur
        model_predictor = get_predictor()

        # 4. Effectuer la classification
        prediction = model_predictor.predict(str(image_path))

        # 5. Construire les données de la prédiction
        prediction_data = {
            **asdict(prediction),
            "case_id": case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_artifact_id": processed_artifact_id
        }

        # 6. Créer un fichier JSON temporaire
        with tempfile.TemporaryDirectory(
            prefix="neurotrace_prediction_"
        ) as temp_dir:

            prediction_path = (
                Path(temp_dir) / f"prediction_{uuid4().hex}.json"
            )

            with open(
                prediction_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    prediction_data,
                    file,
                    ensure_ascii=False,
                    indent=4
                )

            # 7. Enregistrer la prédiction dans le stockage local
            # et ses métadonnées dans PostgreSQL
            prediction_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=str(prediction_path),
                artifact_type="prediction_ia",
                category="results",
                parent_artifact_id=processed_artifact_id,
                model_version=prediction.model_id
            )

        # 8. Retourner le résultat
        return {
            "success": True,
            "message": "Classification IA effectuée avec succès.",
            "case_id": case_id,

            "prediction": {
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "class_scores": prediction.class_scores,
                "model_id": prediction.model_id
            },

            "artifact": {
                "artifact_id": prediction_artifact.artifact_id,
                "artifact_type": prediction_artifact.artifact_type,
                "sha256": prediction_artifact.sha256,
                "parent_artifact_id": prediction_artifact.parent_artifact_id,
                "storage_reference": prediction_artifact.storage_reference
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Erreur pendant la classification : {error}")

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue pendant la classification IA."
        ) from error

@app.post("/cases/{case_id}/validate")
def validate_prediction(case_id: str):

    # 1. Récupérer la dernière prédiction enregistrée
    with SessionLocal() as session:

        prediction_artifact = (
            session.query(Artifact)
            .filter(
                Artifact.case_id == case_id,
                Artifact.artifact_type == "prediction_ia"
            )
            .order_by(Artifact.timestamp.desc())
            .first()
        )
        if prediction_artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Aucune prédiction trouvée pour ce cas."
            )

        prediction_artifact_id = prediction_artifact.artifact_id
        prediction_path = prediction_artifact.storage_reference

    # 2. Lire le fichier JSON de prédiction
    prediction_path = Path(prediction_path)

    if not prediction_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Le fichier de prédiction est introuvable."
        )

    with open(prediction_path, "r", encoding="utf-8") as file:
        prediction_data = json.load(file)

    # 3. Reconstituer l'objet PredictionResult
    prediction = PredictionResult(
        predicted_class=prediction_data["predicted_class"],
        confidence=prediction_data["confidence"],
        class_scores=prediction_data["class_scores"],
        model_id=prediction_data["model_id"]
    )

    # 4. Exécuter la validation
    validation = prediction_validator.validate(prediction)

    # 5. Préparer le résultat de validation
    validation_data = {
        "case_id": case_id,
        "status": validation.status,
        "is_valid": validation.is_valid,
        "confidence": validation.confidence,
        "reasons": validation.reasons,
        "parent_artifact_id": prediction_artifact_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # 6. Enregistrer le résultat dans un fichier JSON temporaire
    temp_dir = Path(tempfile.gettempdir())
    validation_file = temp_dir / f"validation_{uuid4().hex}.json"

    with open(validation_file, "w", encoding="utf-8") as file:
        json.dump(validation_data, file, ensure_ascii=False, indent=4)

    # 7. Enregistrer le résultat dans PostgreSQL et le stockage local
    validation_artifact = artifact_service.register_artifact(
    case_id=case_id,
    source_path=str(validation_file),
    artifact_type="validation_ia",
    category="results",
    parent_artifact_id=prediction_artifact_id,
    model_version=prediction.model_id
)

    # 8. Retourner le résultat à l'utilisateur
    return {
        "success": True,
        "message": "Validation effectuée avec succès.",
        "case_id": case_id,
        "validation": validation_data,
        "artifact": {
            "artifact_id": str(validation_artifact.artifact_id),
            "artifact_type": validation_artifact.artifact_type,
            "sha256": validation_artifact.sha256,
            "parent_artifact_id": str(
                validation_artifact.parent_artifact_id
            ),
            "storage_reference": validation_artifact.storage_reference
        }
    }

def get_gradcam():
    global gradcam_generator

    if gradcam_generator is None:
        gradcam_generator = GradCAM()

    return gradcam_generator

# ============================================================
# 8. GENERATION DE L'EXPLICATION GRAD-CAM
# ============================================================

@app.post("/cases/{case_id}/explain")
def explain_case(case_id: str):
    """Génère une explication Grad-CAM pour une prédiction validée."""

    # 1. Récupérer le dernier artefact de validation
    with SessionLocal() as session:

        validation_artifact = (
            session.query(Artifact)
            .filter(
                Artifact.case_id == case_id,
                Artifact.artifact_type == "validation_ia"
            )
            .order_by(Artifact.timestamp.desc())
            .first()
        )

        if validation_artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Aucune validation trouvée pour ce cas."
            )

        validation_artifact_id = validation_artifact.artifact_id
        prediction_parent_id = validation_artifact.parent_artifact_id

        # 2. Récupérer la prédiction associée à la validation
        prediction_artifact = (
            session.query(Artifact)
            .filter(Artifact.artifact_id == prediction_parent_id)
            .first()
        )

        if prediction_artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Artefact de prédiction introuvable."
            )

        processed_parent_id = prediction_artifact.parent_artifact_id
        model_version = prediction_artifact.model_version

        # 3. Récupérer l'image prétraitée associée à la prédiction
        processed_artifact = (
            session.query(Artifact)
            .filter(Artifact.artifact_id == processed_parent_id)
            .first()
        )

        if processed_artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Image prétraitée introuvable."
            )

        processed_image_path = processed_artifact.storage_reference

    # 4. Vérifier que le fichier existe
    image_path = Path(processed_image_path)

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Le fichier de l'image prétraitée est introuvable."
        )

    try:
        # 5. Générer les explications Grad-CAM
        with tempfile.TemporaryDirectory(
            prefix="neurotrace_gradcam_"
        ) as temp_dir:

            output_dir = Path(temp_dir) / "gradcam"
            output_dir.mkdir(parents=True, exist_ok=True)

            gradcam = get_gradcam()

            result = gradcam.generate(
                image_path=str(image_path),
                output_dir=str(output_dir)
            )

            # 6. Enregistrer la carte thermique
            heatmap_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=result.heatmap_path,
                artifact_type="heatmap_gradcam",
                category="explanations",
                parent_artifact_id=validation_artifact_id,
                model_version=model_version
            )

            # 7. Enregistrer l'image superposée
            overlay_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=result.overlay_path,
                artifact_type="explication_gradcam",
                category="explanations",
                parent_artifact_id=validation_artifact_id,
                model_version=model_version
            )

            # 8. Enregistrer les métadonnées JSON
            metadata_artifact = artifact_service.register_artifact(
                case_id=case_id,
                source_path=result.metadata_path,
                artifact_type="metadonnees_gradcam",
                category="explanations",
                parent_artifact_id=validation_artifact_id,
                model_version=model_version
            )

        # 9. Retourner les résultats
        return {
            "success": True,
            "message": "Explication Grad-CAM générée avec succès.",
            "case_id": case_id,

            "explanation": {
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "heatmap_path": heatmap_artifact.storage_reference,
                "overlay_path": overlay_artifact.storage_reference,
                "metadata_path": metadata_artifact.storage_reference
            },

            "artifacts": {
                "heatmap": {
                    "artifact_id": str(heatmap_artifact.artifact_id),
                    "sha256": heatmap_artifact.sha256,
                    "parent_artifact_id": str(
                        heatmap_artifact.parent_artifact_id
                    )
                },
                "overlay": {
                    "artifact_id": str(overlay_artifact.artifact_id),
                    "sha256": overlay_artifact.sha256,
                    "parent_artifact_id": str(
                        overlay_artifact.parent_artifact_id
                    )
                },
                "metadata": {
                    "artifact_id": str(metadata_artifact.artifact_id),
                    "sha256": metadata_artifact.sha256,
                    "parent_artifact_id": str(
                        metadata_artifact.parent_artifact_id
                    )
                }
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Erreur pendant la génération Grad-CAM : {error}")

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue pendant la génération Grad-CAM."
        ) from error

# ============================================================
# 9. VERIFICATION DE L'INTEGRITE AVEC HYPERLEDGER FABRIC
# ============================================================

@app.post("/cases/{case_id}/artifacts/{artifact_id}/verify")
def verify_artifact_integrity(case_id: str, artifact_id: str):
    """
    Vérifie l'intégrité d'un artefact :
    fichier local contre empreinte enregistrée dans Fabric.
    """

    # 1. Rechercher l'artefact dans PostgreSQL
    with SessionLocal() as session:

        artifact = (
            session.query(Artifact)
            .filter(
                Artifact.artifact_id == artifact_id,
                Artifact.case_id == case_id
            )
            .first()
        )

        if artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Artefact introuvable pour ce cas."
            )

        # Copier les informations nécessaires avant fermeture
        # de la session PostgreSQL.
        artifact_data = {
            "artifact_id": artifact.artifact_id,
            "case_id": artifact.case_id,
            "storage_reference": artifact.storage_reference
        }

    # 2. Vérifier que l'artefact existe dans Fabric
    try:
        # 3. Recalculer le hash et comparer avec Fabric
        from types import SimpleNamespace

        artifact_object = SimpleNamespace(**artifact_data)

        result = integrity_verifier.verify_artifact(
            artifact_object
        )

    except Exception as error:
        print(f"Erreur de vérification : {error}")

        raise HTTPException(
            status_code=502,
            detail="Impossible de vérifier l'intégrité de l'artefact."
        ) from error

    # 4. Retourner le résultat
    if not result["success"]:
        raise HTTPException(
            status_code=502,
            detail=result
        )

    return result
# ============================================================
# 10. HISTORIQUE D'UN ARTEFACT SUR HYPERLEDGER FABRIC
# ============================================================

@app.get("/cases/{case_id}/artifacts/{artifact_id}/history")
def get_artifact_history(case_id: str, artifact_id: str):

    # 1. Vérifier que l'artefact existe dans PostgreSQL
    with SessionLocal() as session:
        artifact = (
            session.query(Artifact)
            .filter(
                Artifact.artifact_id == artifact_id,
                Artifact.case_id == case_id
            )
            .first()
        )

        if artifact is None:
            raise HTTPException(
                status_code=404,
                detail="Artefact introuvable pour ce cas."
            )

    # 2. Récupérer son historique depuis Hyperledger Fabric
    try:
        history = fabric_client.get_artifact_history(artifact_id)

        return {
            "success": True,
            "case_id": case_id,
            "artifact_id": artifact_id,
            "history": history
        }

    except Exception as error:
        print(f"Erreur récupération historique Fabric : {error}")

        raise HTTPException(
            status_code=502,
            detail="Impossible de récupérer l'historique depuis Hyperledger Fabric."
        ) from error

# ============================================================
# 11. PROVENANCE COMPLETE D'UN CAS
# ============================================================

@app.get("/cases/{case_id}/provenance")
def get_case_provenance(case_id: str):

    with SessionLocal() as session:

        # 1. Récupérer tous les artefacts du cas
        artifacts = (
            session.query(Artifact)
            .filter(Artifact.case_id == case_id)
            .order_by(Artifact.timestamp.asc())
            .all()
        )

        if not artifacts:
            raise HTTPException(
                status_code=404,
                detail="Aucun artefact trouvé pour ce cas."
            )

        # 2. Construire les nœuds de provenance
        nodes = {}

        for artifact in artifacts:
            blockchain_info = (artifact.details or {}).get(
                "blockchain", {}
            )

            nodes[artifact.artifact_id] = {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "sha256": artifact.sha256,
                "timestamp": (
                    artifact.timestamp.isoformat()
                    if artifact.timestamp else None
                ),
                "model_version": artifact.model_version,
                "parent_artifact_id": artifact.parent_artifact_id,
                "blockchain_status": blockchain_info.get(
                    "status", "inconnu"
                ),
                "children": []
            }

        # 3. Reconstituer les relations parent-enfant
        roots = []

        for node in nodes.values():
            parent_id = node["parent_artifact_id"]

            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)

        # 4. Retourner l'arbre de provenance
        return {
            "success": True,
            "case_id": case_id,
            "artifact_count": len(artifacts),
            "provenance": roots
        }


