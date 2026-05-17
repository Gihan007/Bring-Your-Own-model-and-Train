from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.models import ValidationJobRequest, ValidationMetrics


class ValidationAdapter(Protocol):
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        ...

