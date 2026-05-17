from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.adapters.utils import build_metrics, load_dataset, maybe_extract_model_archive, write_result
from app.models import ValidationJobRequest, ValidationMetrics


class HuggingFaceAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        pipeline = self._lazy_import()
        dataset = load_dataset(request, dataset_path)

        text_column = request.dimensions.get("text_column")
        if not isinstance(text_column, str) or text_column not in dataset.columns:
            raise ValueError("dimensions.text_column is required for huggingface validation")

        hf_task = request.dimensions.get("hf_task") or request.task_type or "text-classification"
        if hf_task == "classification":
            hf_task = "text-classification"
        resolved_model_path = maybe_extract_model_archive(model_path, artifacts_dir)
        pipe = pipeline(hf_task, model=str(resolved_model_path), tokenizer=str(resolved_model_path))

        outputs = pipe(dataset[text_column].astype(str).tolist())
        predictions = self._extract_labels(outputs)
        y_true = dataset[request.target_column]

        result = build_metrics(
            request,
            y_true,
            predictions,
            row_count=int(len(dataset)),
            feature_count=1,
            extra_metrics={"hf_task": hf_task},
        )
        write_result(artifacts_dir, result)
        return result

    def _extract_labels(self, outputs: Any) -> List[Any]:
        labels: List[Any] = []
        for output in outputs:
            if isinstance(output, list):
                output = max(output, key=lambda item: item.get("score", 0))
            if isinstance(output, dict):
                labels.append(output.get("label"))
            else:
                labels.append(output)
        return labels

    def _lazy_import(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("huggingface adapter dependency is missing: transformers") from exc
        return pipeline
