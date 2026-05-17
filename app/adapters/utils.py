from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import ValidationJobRequest, ValidationMetrics


def load_dataset(request: ValidationJobRequest, dataset_path: Path) -> Any:
    pd = import_pandas()
    if not request.target_column:
        raise ValueError("target_column is required for validation")

    dataset = pd.read_csv(dataset_path)
    if request.target_column not in dataset.columns:
        raise ValueError(f"target_column not found in dataset: {request.target_column}")
    return dataset


def select_feature_columns(request: ValidationJobRequest, dataset: Any, model: Any = None) -> List[str]:
    supplied = [*request.numeric_features, *request.categorical_features]
    if supplied:
        missing = [column for column in supplied if column not in dataset.columns]
        if missing:
            raise ValueError(f"feature columns missing from dataset: {missing}")
        return supplied

    feature_names = getattr(model, "feature_names_in_", None) if model is not None else None
    if feature_names is not None:
        return [column for column in list(feature_names) if column in dataset.columns]

    feature_columns = request.dimensions.get("feature_columns")
    if isinstance(feature_columns, list) and all(isinstance(column, str) for column in feature_columns):
        missing = [column for column in feature_columns if column not in dataset.columns]
        if missing:
            raise ValueError(f"feature columns missing from dataset: {missing}")
        return feature_columns

    return [column for column in dataset.columns if column != request.target_column]


def prepare_tabular_features(request: ValidationJobRequest, dataset: Any, feature_columns: List[str]) -> Any:
    pd = import_pandas()
    x_values = dataset[feature_columns]
    if request.categorical_features:
        x_values = pd.get_dummies(x_values, columns=request.categorical_features)

    expected_features = request.dimensions.get("encoded_feature_columns")
    if isinstance(expected_features, list) and all(isinstance(column, str) for column in expected_features):
        x_values = x_values.reindex(columns=expected_features, fill_value=0)

    return x_values


def decode_predictions(request: ValidationJobRequest, raw_predictions: Any) -> List[Any]:
    np = import_numpy()
    predictions = np.asarray(raw_predictions)

    if predictions.ndim == 0:
        predictions = predictions.reshape(1)
    if predictions.ndim > 1:
        if predictions.shape[1] == 1:
            values = predictions.reshape(-1)
            if request.task_type == "classification":
                predictions = (values >= 0.5).astype(int)
            else:
                predictions = values
        elif request.task_type == "classification":
            predictions = predictions.argmax(axis=1)
        else:
            predictions = predictions[:, 0]

    output = predictions.tolist()
    class_names = request.dimensions.get("class_names")
    if request.task_type == "classification" and isinstance(class_names, list):
        mapped = []
        for value in output:
            if isinstance(value, (int, float)) and int(value) == value and 0 <= int(value) < len(class_names):
                mapped.append(class_names[int(value)])
            else:
                mapped.append(value)
        output = mapped

    return jsonable(output)


def build_metrics(
    request: ValidationJobRequest,
    y_true: Any,
    y_pred: List[Any],
    row_count: int,
    feature_count: int,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> ValidationMetrics:
    metrics = import_sklearn_metrics()
    metric_values: Dict[str, Any]
    if request.task_type == "regression":
        mse = float(metrics.mean_squared_error(y_true, y_pred))
        metric_values = {
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": mse ** 0.5,
            "r2": float(metrics.r2_score(y_true, y_pred)),
        }
    else:
        metric_values = {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "precision_macro": float(metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    if extra_metrics:
        metric_values.update(extra_metrics)

    return ValidationMetrics(
        metrics=metric_values,
        predictions_preview=jsonable(list(y_pred[:10])),
        classes=jsonable(sorted(set(y_true))) if request.task_type != "regression" else [],
        row_count=row_count,
        feature_count=feature_count,
    )


def write_result(artifacts_dir: Path, result: ValidationMetrics) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    (artifacts_dir / "validation_result.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def maybe_extract_model_archive(model_path: Path, destination_dir: Path) -> Path:
    if model_path.suffix.lower() != ".zip":
        return model_path

    extract_dir = destination_dir / f"{model_path.stem}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(model_path) as archive:
        archive.extractall(extract_dir)

    children = [path for path in extract_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def jsonable(values: List[Any]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        if hasattr(value, "item"):
            result.append(value.item())
        else:
            result.append(value)
    return result


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("adapter dependency is missing: numpy") from exc
    return np


def import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("adapter dependency is missing: pandas") from exc
    return pd


def import_sklearn_metrics() -> Any:
    try:
        from sklearn import metrics
    except ImportError as exc:
        raise RuntimeError("adapter dependency is missing: scikit-learn") from exc
    return metrics
