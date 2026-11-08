
from pathlib import Path
from PIL import Image

from src.neurotrace.preprocessing.processor import (
    MRIImageProcessor
)


def test_valid_image(tmp_path):

    input_path = tmp_path / "brain.jpg"
    output_path = tmp_path / "processed" / "brain.png"

    # Création d'une image de test
    image = Image.new("RGB", (512, 512), color="gray")
    image.save(input_path)

    processor = MRIImageProcessor()

    result = processor.process_image(
        str(input_path),
        str(output_path)
    )

    assert result.success is True
    assert output_path.exists()
    assert result.metadata["processed_width"] == 224
    assert result.metadata["processed_height"] == 224
    assert result.metadata_path is not None
    assert Path(result.metadata_path).exists()


def test_invalid_image(tmp_path):

    input_path = tmp_path / "invalid.jpg"
    output_path = tmp_path / "processed.png"

    # Création d'un fichier qui n'est pas une image
    input_path.write_text("ceci n'est pas une image")

    processor = MRIImageProcessor()

    result = processor.process_image(
        str(input_path),
        str(output_path)
    )

    assert result.success is False
    assert result.error is not None
    assert not output_path.exists()


def test_missing_image(tmp_path):

    processor = MRIImageProcessor()

    result = processor.process_image(
        str(tmp_path / "missing.jpg"),
        str(tmp_path / "output.png")
    )

    assert result.success is False
    assert "introuvable" in result.error.lower()
    
