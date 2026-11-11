
import pytest

from neurotrace.security.hashing import (
    calculate_file_hash,
    calculate_data_hash,
    verify_file_integrity,
    verify_data_integrity,
    create_artifact_metadata,
    save_hash_metadata,
    verify_artifact_metadata
)


def test_calculate_file_hash(tmp_path):
    file_path = tmp_path / "image.txt"
    file_path.write_text("NeuroTrace", encoding="utf-8")

    result = calculate_file_hash(file_path)

    assert len(result) == 64


def test_file_hash_is_deterministic(tmp_path):
    file_path = tmp_path / "image.txt"
    file_path.write_text("NeuroTrace", encoding="utf-8")

    hash1 = calculate_file_hash(file_path)
    hash2 = calculate_file_hash(file_path)

    assert hash1 == hash2


def test_file_integrity_valid(tmp_path):
    file_path = tmp_path / "image.txt"
    file_path.write_text("NeuroTrace", encoding="utf-8")

    expected_hash = calculate_file_hash(file_path)

    assert verify_file_integrity(file_path, expected_hash)


def test_file_integrity_detects_modification(tmp_path):
    file_path = tmp_path / "image.txt"
    file_path.write_text("NeuroTrace", encoding="utf-8")

    expected_hash = calculate_file_hash(file_path)

    file_path.write_text("NeuroTrace modified", encoding="utf-8")

    assert not verify_file_integrity(file_path, expected_hash)


def test_calculate_data_hash():
    data = {"class": "glioma", "confidence": 0.90}

    result = calculate_data_hash(data)

    assert len(result) == 64


def test_data_hash_is_deterministic():
    data1 = {"class": "glioma", "confidence": 0.90}
    data2 = {"confidence": 0.90, "class": "glioma"}

    assert calculate_data_hash(data1) == calculate_data_hash(data2)


def test_data_integrity_detects_modification():
    original = {"class": "glioma", "confidence": 0.90}
    modified = {"class": "meningioma", "confidence": 0.90}

    expected_hash = calculate_data_hash(original)

    assert not verify_data_integrity(modified, expected_hash)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        calculate_file_hash("fichier_inexistant.jpg")




def test_create_artifact_metadata(tmp_path):
    file_path = tmp_path / "original.jpg"
    file_path.write_text("MRI data", encoding="utf-8")

    metadata = create_artifact_metadata(
        file_path=file_path,
        case_id="CASE_001",
        artifact_type="original_image"
    )

    assert metadata["case_id"] == "CASE_001"
    assert metadata["artifact_type"] == "original_image"
    assert len(metadata["sha256"]) == 64
    assert metadata["parent_artifact_id"] is None


def test_save_hash_metadata(tmp_path):
    file_path = tmp_path / "original.jpg"
    file_path.write_text("MRI data", encoding="utf-8")

    metadata = create_artifact_metadata(
        file_path, "CASE_001", "original_image"
    )

    output_path = tmp_path / "metadata" / "original.json"

    save_hash_metadata(metadata, output_path)

    assert output_path.exists()


def test_verify_artifact_metadata(tmp_path):
    file_path = tmp_path / "original.jpg"
    file_path.write_text("MRI data", encoding="utf-8")

    metadata = create_artifact_metadata(
        file_path, "CASE_001", "original_image"
    )

    assert verify_artifact_metadata(file_path, metadata)

    file_path.write_text("Modified MRI data", encoding="utf-8")

    assert not verify_artifact_metadata(file_path, metadata)
