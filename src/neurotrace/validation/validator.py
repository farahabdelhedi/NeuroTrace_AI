
"""
NeuroTrace - Prediction Validator

Responsabilités :
- Vérifier le niveau de confiance.
- Attribuer un statut à la prédiction.
- Expliquer la décision de validation.
"""

from dataclasses import dataclass


# Seuils expérimentaux du prototype
ACCEPTANCE_THRESHOLD = 80.0
REVIEW_THRESHOLD = 50.0


@dataclass
class ValidationResult:

    status: str
    is_valid: bool
    confidence: float
    reasons: list[str]


class PredictionValidator:

    def validate(self, prediction):

        confidence = prediction.confidence
        reasons = []

        # Cas 1 : confiance élevée
        if confidence >= ACCEPTANCE_THRESHOLD:

            status = "Accepté"
            is_valid = True

            reasons.append(
                "Le niveau de confiance est supérieur "
                "ou égal à 80 %."
            )

        # Cas 2 : confiance intermédiaire
        elif confidence >= REVIEW_THRESHOLD:

            status = "Révision requise"
            is_valid = False

            reasons.append(
                "Le niveau de confiance est compris "
                "entre 50 % et 80 %."
            )

        # Cas 3 : confiance faible
        else:

            status = "Rejeté"
            is_valid = False

            reasons.append(
                "Le niveau de confiance est inférieur à 50 %."
            )

        return ValidationResult(
            status=status,
            is_valid=is_valid,
            confidence=confidence,
            reasons=reasons
        )
