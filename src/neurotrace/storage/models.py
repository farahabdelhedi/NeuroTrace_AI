
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    DateTime,
    ForeignKey,
    JSON
)
from sqlalchemy.orm import Mapped, mapped_column

from neurotrace.storage.database import Base


class Case(Base):
    """Représente un cas d'analyse IRM."""

    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class Artifact(Base):
    """Représente un artefact produit pendant l'analyse."""

    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True
    )

    case_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("cases.case_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    model_version: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True
    )

    parent_artifact_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("artifacts.artifact_id"),
        nullable=True
    )

    storage_reference: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    details: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True
    )
