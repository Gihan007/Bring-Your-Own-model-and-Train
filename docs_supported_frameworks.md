# Supported Framework Adapters

The prototype now has adapters for:

- `sklearn`: `.joblib`, `.pkl`, `.pickle`
- `xgboost`: native XGBoost model files such as `.xgb`, `.ubj`, `.json`
- `onnx`: `.onnx`
- `tensorflow`: `.keras`, `.h5`, or zipped Keras/SavedModel directories
- `pytorch`: TorchScript `.pt` / `.pth`
- `huggingface`: zipped Transformers model directory

## Important Runtime Rule

The API can accept all these jobs, but the actual validation runtime must have the required library installed from the uploaded `requirements.txt`.

For example:

```text
framework=xgboost      -> xgboost must be installed
framework=onnx         -> onnxruntime must be installed
framework=tensorflow   -> tensorflow must be installed
framework=pytorch      -> torch must be installed
framework=huggingface  -> transformers and torch must be installed
```

When `BYOM_EXECUTOR=docker`, those dependencies are installed inside a per-job Docker image built from the uploaded `requirements.txt`.

When `BYOM_RUNTIME_AGENT=true`, the platform can retry obvious runtime mismatches, such as a package requiring a newer Python version than the first inferred plan.

## Upload Endpoint

```text
POST /validation-jobs/upload
```

Use multipart form data:

```text
model_file
dataset_file
requirements_file
job_id
tenant_id
model_id
framework
task_type
target_column
numeric_features
categorical_features
dimensions
```

## Framework Notes

### XGBoost

Expected model: file saved by `XGBClassifier.save_model()` or `XGBRegressor.save_model()`.

If the model file extension is `.json`, set `framework=xgboost` explicitly because `.json` is too generic to infer safely.

Use `dimensions.encoded_feature_columns` if your training used one-hot encoding and you need exact encoded column order.

### ONNX

Expected model: `.onnx`.

Use `dimensions.onnx_input_name` when the model input is not the first ONNX input or has a specific name.

### TensorFlow

Expected model: `.keras`, `.h5`, or a zipped model directory.

The adapter uses `tf.keras.models.load_model()`.

### PyTorch

Expected model: TorchScript, not a raw `state_dict`.

Save with:

```python
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

Raw PyTorch checkpoints need the original model class, so they are not portable enough for generic BYOM validation.

### HuggingFace

Expected model: zipped local Transformers model directory containing model config, weights, tokenizer files.

The validation CSV must have:

```text
text column
label/target column
```

Set:

```json
{
  "dimensions": {
    "hf_task": "text-classification",
    "text_column": "text"
  }
}
```

## Optional Local Full-Adapter Install

To try installing all adapter libraries in the local prototype venv:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-adapters-full.txt
```

This can take a long time because PyTorch, TensorFlow, and Transformers are large packages. The production design should install these per job from the uploaded `requirements.txt`, not permanently in the API service.
