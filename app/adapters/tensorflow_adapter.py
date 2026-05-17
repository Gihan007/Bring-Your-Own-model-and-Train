from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.utils import (
    build_metrics,
    decode_predictions,
    load_dataset,
    maybe_extract_model_archive,
    prepare_tabular_features,
    select_feature_columns,
    write_result,
)
from app.models import ValidationJobRequest, ValidationMetrics


class TensorFlowAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        tf = self._lazy_import()
        dataset = load_dataset(request, dataset_path)
        feature_columns = select_feature_columns(request, dataset)
        x_values = prepare_tabular_features(request, dataset, feature_columns)
        y_true = dataset[request.target_column]

        resolved_model_path = maybe_extract_model_archive(model_path, artifacts_dir)
        model = tf.keras.models.load_model(str(resolved_model_path))
        raw_predictions = model.predict(x_values.to_numpy(), verbose=0)
        predictions = decode_predictions(request, raw_predictions)

        result = build_metrics(
            request,
            y_true,
            predictions,
            row_count=int(len(dataset)),
            feature_count=int(x_values.shape[1]),
        )
        write_result(artifacts_dir, result)
        return result

    def _lazy_import(self) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("tensorflow adapter dependency is missing: tensorflow") from exc
        return tf

