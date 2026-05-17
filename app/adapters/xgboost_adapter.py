from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.utils import (
    build_metrics,
    decode_predictions,
    load_dataset,
    prepare_tabular_features,
    select_feature_columns,
    write_result,
)
from app.models import ValidationJobRequest, ValidationMetrics


class XGBoostAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        xgb = self._lazy_import()
        dataset = load_dataset(request, dataset_path)
        feature_columns = select_feature_columns(request, dataset)
        x_values = prepare_tabular_features(request, dataset, feature_columns)
        y_true = dataset[request.target_column]

        model = xgb.XGBRegressor() if request.task_type == "regression" else xgb.XGBClassifier()
        model.load_model(str(model_path))

        raw_predictions = model.predict(x_values)
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
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost adapter dependency is missing: xgboost") from exc
        return xgb

