
from types import SimpleNamespace

from src.neurotrace.validation.validator import (
    PredictionValidator
)


def test_prediction_accepted():

    prediction = SimpleNamespace(confidence=90.63)

    validator = PredictionValidator()
    result = validator.validate(prediction)

    assert result.status == "Accepté"
    assert result.is_valid is True


def test_prediction_requires_review():

    prediction = SimpleNamespace(confidence=65.0)

    validator = PredictionValidator()
    result = validator.validate(prediction)

    assert result.status == "Révision requise"
    assert result.is_valid is False


def test_prediction_rejected():

    prediction = SimpleNamespace(confidence=35.0)

    validator = PredictionValidator()
    result = validator.validate(prediction)

    assert result.status == "Rejeté"
    assert result.is_valid is False
