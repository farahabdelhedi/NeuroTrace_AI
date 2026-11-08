
from pathlib import Path

from src.neurotrace.explainability.gradcam import GradCAM


def test_gradcam_real_mri():

    base_dir = Path(__file__).resolve().parent.parent

    image_path = (
        base_dir
        / "artifacts"
        / "preprocessing_test"
        / "Te-gl_1.png"
    )

    output_dir = (
        base_dir
        / "artifacts"
        / "explanations"
    )

    assert image_path.exists(), (
        "Image prétraitée introuvable. "
        "Exécuter d'abord le test de prétraitement."
    )

    gradcam = GradCAM()

    try:
        result = gradcam.generate(
            image_path=str(image_path),
            output_dir=str(output_dir)
        )

        # Vérifier la classe prédite
        assert result.predicted_class in [
            "glioma",
            "meningioma",
            "notumor",
            "pituitary"
        ]

        # Vérifier que les fichiers ont été créés
        assert Path(result.heatmap_path).exists()
        assert Path(result.overlay_path).exists()
        assert Path(result.metadata_path).exists()

        print("\n========== RESULTAT GRAD-CAM ==========")

        print("Classe prédite :", result.predicted_class)
        print("Confiance :", result.confidence, "%")
        print("Heatmap :", result.heatmap_path)
        print("Superposition :", result.overlay_path)
        print("Métadonnées :", result.metadata_path)

    finally:
        gradcam.close()
