# Sample Validation Cases

These cases let you test the prototype across successful, inferred, legacy-runtime, and failure paths.

| Case | Payload | Expected Result | Purpose |
| --- | --- | --- | --- |
| sklearn classification pipeline | `samples/cases/sklearn_classification_pipeline/payload.json` | `completed` | Happy path for sklearn `.joblib` classification with preprocessing inside the pipeline. |
| sklearn regression pipeline | `samples/cases/sklearn_regression_pipeline/payload.json` | `completed` | Happy path for sklearn `.joblib` regression metrics. |
| Kaggle logistic regression | `samples/cases/kaggle_logistic_regression/payload.json` | `completed` | Real KaggleHub sklearn `.pkl` model with fake matching validation data and sklearn `1.2.2` runtime requirements. |
| sklearn inferred framework | `samples/cases/sklearn_inferred_framework/payload.json` | `completed` | Payload omits `framework`; resolver infers sklearn from `.joblib`. |
| sklearn legacy runtime plan | `samples/cases/sklearn_legacy_runtime_plan/payload.json` | `completed` | Uses old `scikit-learn==0.24.2` requirements to show Python `3.8` runtime inference. |
| xgboost placeholder failure | `samples/cases/unsupported_xgboost/payload.json` | `failed` | Shows clean failure when the runtime lacks `xgboost` or the uploaded model file is not a real XGBoost model. |
| missing target column | `samples/cases/failure_missing_target/payload.json` | `failed` | Shows validation failure when payload metadata does not match the dataset. |

The legacy-runtime case proves runtime planning only. This prototype still executes in the local Python environment; the next production step is building/running the dependency-specific Docker image from the runtime plan.
