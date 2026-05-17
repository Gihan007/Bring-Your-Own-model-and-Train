from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    root = Path(__file__).resolve().parents[1]
    cases_dir = root / "samples" / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    classification_case = cases_dir / "sklearn_classification_pipeline"
    regression_case = cases_dir / "sklearn_regression_pipeline"
    inferred_case = cases_dir / "sklearn_inferred_framework"
    legacy_case = cases_dir / "sklearn_legacy_runtime_plan"
    unsupported_case = cases_dir / "unsupported_xgboost"
    missing_target_case = cases_dir / "failure_missing_target"

    _write_classification_case(
        classification_case,
        joblib,
        pd,
        ColumnTransformer,
        SimpleImputer,
        LogisticRegression,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    )
    _write_regression_case(
        regression_case,
        joblib,
        pd,
        ColumnTransformer,
        SimpleImputer,
        LinearRegression,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    )

    _clone_case(classification_case, inferred_case)
    _patch_payload(inferred_case / "payload.json", {"job_id": "case-sklearn-inferred-framework", "framework": None})

    _clone_case(classification_case, legacy_case)
    (legacy_case / "requirements.txt").write_text(
        "pandas==1.3.5\njoblib==1.1.1\nscikit-learn==0.24.2\n",
        encoding="utf-8",
    )
    _patch_payload(
        legacy_case / "payload.json",
        {
            "job_id": "case-sklearn-legacy-runtime-plan",
            "requirements_gcs_path": "samples/cases/sklearn_legacy_runtime_plan/requirements.txt",
        },
    )

    _write_unsupported_xgboost_case(unsupported_case)

    _clone_case(classification_case, missing_target_case)
    _patch_payload(
        missing_target_case / "payload.json",
        {
            "job_id": "case-failure-missing-target",
            "target_column": "MissingRisk",
            "model_gcs_path": "samples/cases/failure_missing_target/model.joblib",
            "dataset_gcs_path": "samples/cases/failure_missing_target/validation.csv",
            "requirements_gcs_path": "samples/cases/failure_missing_target/requirements.txt",
        },
    )

    # Keep backward-compatible sample names from the first prototype.
    shutil.copy2(classification_case / "model.joblib", root / "samples" / "model.joblib")
    shutil.copy2(classification_case / "validation.csv", root / "samples" / "validation.csv")

    print(f"wrote sample cases under {cases_dir}")


def _write_classification_case(
    case_dir: Path,
    joblib: Any,
    pd: Any,
    ColumnTransformer: Any,
    SimpleImputer: Any,
    LogisticRegression: Any,
    Pipeline: Any,
    OneHotEncoder: Any,
    StandardScaler: Any,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    numeric_features = ["Age", "Job", "Credit amount", "Duration"]
    categorical_features = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    data = pd.DataFrame(
        [
            [22, 2, 1200, 12, "male", "rent", "little", "little", "radio/TV", "bad"],
            [35, 1, 2400, 24, "female", "own", "moderate", "moderate", "car", "good"],
            [48, 3, 8000, 36, "male", "own", "rich", "none", "business", "good"],
            [29, 2, 1600, 18, "female", "rent", "little", "little", "education", "bad"],
            [41, 1, 4300, 30, "male", "free", "moderate", "none", "furniture", "good"],
            [31, 2, 1900, 15, "female", "own", "little", "moderate", "car", "bad"],
            [55, 3, 6100, 42, "male", "own", "moderate", "rich", "business", "good"],
            [26, 1, 900, 9, "female", "rent", "little", "little", "radio/TV", "bad"],
        ],
        columns=[*numeric_features, *categorical_features, "Risk"],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(data[numeric_features + categorical_features], data["Risk"])

    data.to_csv(case_dir / "validation.csv", index=False)
    joblib.dump(model, case_dir / "model.joblib")
    (case_dir / "requirements.txt").write_text(
        "pandas==2.2.2\njoblib==1.4.2\nscikit-learn==1.5.1\n",
        encoding="utf-8",
    )
    _write_payload(
        case_dir / "payload.json",
        {
            "job_id": "case-sklearn-classification",
            "tenant_id": "tenant-demo",
            "model_id": "credit-risk-classifier",
            "model_gcs_path": "samples/cases/sklearn_classification_pipeline/model.joblib",
            "dataset_gcs_path": "samples/cases/sklearn_classification_pipeline/validation.csv",
            "requirements_gcs_path": "samples/cases/sklearn_classification_pipeline/requirements.txt",
            "framework": "sklearn",
            "task_type": "classification",
            "target_column": "Risk",
            "model_type": "LogisticRegression",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "model_params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
            "dimensions": {"purpose": "happy path sklearn classification"},
        },
    )


def _write_regression_case(
    case_dir: Path,
    joblib: Any,
    pd: Any,
    ColumnTransformer: Any,
    SimpleImputer: Any,
    LinearRegression: Any,
    Pipeline: Any,
    OneHotEncoder: Any,
    StandardScaler: Any,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    numeric_features = ["rooms", "area_sqft", "age_years"]
    categorical_features = ["location", "condition"]
    data = pd.DataFrame(
        [
            [2, 850, 20, "suburb", "fair", 210000],
            [3, 1200, 12, "suburb", "good", 310000],
            [4, 1800, 8, "city", "good", 520000],
            [1, 650, 30, "city", "fair", 260000],
            [5, 2400, 4, "city", "excellent", 760000],
            [3, 1400, 18, "rural", "good", 230000],
            [4, 2100, 6, "suburb", "excellent", 610000],
            [2, 900, 25, "rural", "fair", 160000],
        ],
        columns=[*numeric_features, *categorical_features, "price"],
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(data[numeric_features + categorical_features], data["price"])

    data.to_csv(case_dir / "validation.csv", index=False)
    joblib.dump(model, case_dir / "model.joblib")
    (case_dir / "requirements.txt").write_text(
        "pandas==2.2.2\njoblib==1.4.2\nscikit-learn==1.5.1\n",
        encoding="utf-8",
    )
    _write_payload(
        case_dir / "payload.json",
        {
            "job_id": "case-sklearn-regression",
            "tenant_id": "tenant-demo",
            "model_id": "house-price-regressor",
            "model_gcs_path": "samples/cases/sklearn_regression_pipeline/model.joblib",
            "dataset_gcs_path": "samples/cases/sklearn_regression_pipeline/validation.csv",
            "requirements_gcs_path": "samples/cases/sklearn_regression_pipeline/requirements.txt",
            "framework": "sklearn",
            "task_type": "regression",
            "target_column": "price",
            "model_type": "LinearRegression",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "model_params": {},
            "dimensions": {"purpose": "happy path sklearn regression"},
        },
    )


def _write_unsupported_xgboost_case(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "model.xgb").write_text("placeholder only; prototype should fail before loading", encoding="utf-8")
    (case_dir / "validation.csv").write_text("x1,target\n1,1\n2,0\n", encoding="utf-8")
    (case_dir / "requirements.txt").write_text("xgboost==2.1.1\npandas==2.2.2\n", encoding="utf-8")
    _write_payload(
        case_dir / "payload.json",
        {
            "job_id": "case-unsupported-xgboost",
            "tenant_id": "tenant-demo",
            "model_id": "unsupported-xgboost-model",
            "model_gcs_path": "samples/cases/unsupported_xgboost/model.xgb",
            "dataset_gcs_path": "samples/cases/unsupported_xgboost/validation.csv",
            "requirements_gcs_path": "samples/cases/unsupported_xgboost/requirements.txt",
            "framework": "xgboost",
            "task_type": "classification",
            "target_column": "target",
            "numeric_features": ["x1"],
            "categorical_features": [],
            "model_params": {},
            "dimensions": {"purpose": "expected unsupported framework failure"},
        },
    )


def _clone_case(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    _patch_payload(
        target_dir / "payload.json",
        {
            "model_gcs_path": f"samples/cases/{target_dir.name}/model.joblib",
            "dataset_gcs_path": f"samples/cases/{target_dir.name}/validation.csv",
            "requirements_gcs_path": f"samples/cases/{target_dir.name}/requirements.txt",
        },
    )


def _patch_payload(payload_path: Path, updates: Dict[str, Any]) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.update(updates)
    _write_payload(payload_path, payload)


def _write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
