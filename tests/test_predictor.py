
from pathlib import Path

from src.neurotrace.ai.predictor import MRIPredictor


def test_real_mri_prediction():

    base_dir = Path(__file__).resolve().parent.parent

    # Image déjà prétraitée à l'étape précédente
    image_path = (
        base_dir
        / "artifacts"
        / "preprocessing_test"
        / "Te-gl_1.png"
    )

    assert image_path.exists(), (
        "Image prétraitée introuvable. "
        "Exécuter d'abord le test de prétraitement réel."
    )

    # Initialiser le prédicteur
    predictor = MRIPredictor()

    # Effectuer la prédiction
    result = predictor.predict(str(image_path))

    # Vérifier le résultat
    assert result.predicted_class in [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    ]

    assert 0 <= result.confidence <= 100

    assert len(result.class_scores) == 4

    assert abs(sum(result.class_scores.values()) - 100) < 0.1

    assert result.model_id == (
        "Abuzaid01/brain-tumor-resnet50-classifier"
    )

    print("\n========== RESULTAT DE LA PREDICTION ==========")

    print("Classe prédite :", result.predicted_class)

    print("Confiance :", result.confidence, "%")

    print("\nScores par classe :")

    for class_name, score in result.class_scores.items():
        print(f"{class_name} : {score}%")

    print("\nModèle utilisé :", result.model_id)
