from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.models import ValidationJobRequest, ValidationMetrics


class SklearnJoblibAdapter:
    def validate(
        self,
        request: ValidationJobRequest,
        model_path: Path,
        dataset_path: Path,
        artifacts_dir: Path,
    ) -> ValidationMetrics:
        pd, joblib, metrics = self._lazy_imports()

        if not request.target_column:
            raise ValueError("target_column is required for sklearn validation")

        dataset = pd.read_csv(dataset_path)
        if request.target_column not in dataset.columns:
            raise ValueError(f"target_column not found in dataset: {request.target_column}")

        model = joblib.load(model_path)
        y_true = dataset[request.target_column]
        feature_columns = self._select_feature_columns(request, dataset, model)
        x_values = dataset[feature_columns]

        y_pred = model.predict(x_values)
        if request.task_type == "regression":
            metric_values = self._regression_metrics(metrics, y_true, y_pred)
        else:
            metric_values = self._classification_metrics(metrics, y_true, y_pred)

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(x_values)
                metric_values["predict_proba_available"] = True
                metric_values["probability_columns"] = int(probabilities.shape[1])
            except Exception as exc:
                metric_values["predict_proba_available"] = False
                metric_values["predict_proba_error"] = str(exc)

        preview = list(y_pred[:10])
        classes = list(getattr(model, "classes_", []))
        output = {
            "metrics": metric_values,
            "predictions_preview": self._jsonable(preview),
            "classes": self._jsonable(classes),
            "row_count": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
        }

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "validation_result.json").write_text(
            json.dumps(output, indent=2),
            encoding="utf-8",
        )

        return ValidationMetrics(**output)

    def _select_feature_columns(self, request: ValidationJobRequest, dataset: Any, model: Any) -> List[str]:
        supplied = [*request.numeric_features, *request.categorical_features]
        if supplied:
            missing = [column for column in supplied if column not in dataset.columns]
            if missing:
                raise ValueError(f"feature columns missing from dataset: {missing}")
            return supplied

        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            return [column for column in list(feature_names) if column in dataset.columns]

        return [column for column in dataset.columns if column != request.target_column]

    def _classification_metrics(self, metrics: Any, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        return {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "precision_macro": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def _regression_metrics(self, metrics: Any, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        mse = float(metrics.mean_squared_error(y_true, y_pred))
        return {
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": mse ** 0.5,
            "r2": float(metrics.r2_score(y_true, y_pred)),
        }

    def _jsonable(self, values: List[Any]) -> List[Any]:
        result: List[Any] = []
        for value in values:
            if hasattr(value, "item"):
                result.append(value.item())
            else:
                result.append(value)
        return result

    def _lazy_imports(self) -> Any:
        try:
            import joblib
            import pandas as pd
            from sklearn import metrics
        except ImportError as exc:
            raise RuntimeError(
                "sklearn adapter dependencies are missing. Install pandas, joblib, and scikit-learn "
                "inside the validation runtime."
            ) from exc
        return pd, joblib, metrics
