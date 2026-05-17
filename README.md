# BYOM Model Validation Prototype

This is a small prototype for a **containerized bring-your-own-model validation platform**.

The goal is to show the correct flow when a customer uploads:

- model file, for example `model.joblib`
- `requirements.txt`
- validation dataset
- supporting files and metadata in the job payload

The platform creates a validation job, resolves a runtime plan from the payload and requirements, picks a framework adapter, and runs validation in an isolated worker flow.

## Prototype Scope

Implemented now:

- FastAPI job API
- in-memory job store
- runtime resolver with dependency hash and Python-version inference
- sklearn/joblib classification validation adapter
- local path support for prototype testing
- optional `gs://` download hook using `google-cloud-storage`
- clear failure state when metadata or dependencies are missing

Not implemented yet:

- real queue such as Redis, Kafka, RabbitMQ, or Cloud Tasks
- real Docker image build per job
- container registry cache
- Kubernetes Jobs / Cloud Run Jobs execution
- production sandbox limits

## API Flow

```text
POST /validation-jobs
  -> queued
  -> resolving_environment
  -> running_validation
  -> completed / failed

GET /validation-jobs/{job_id}
  -> current status, runtime plan, metrics, or error
```

## Runtime Strategy

The prototype reads `requirements.txt` and builds a runtime plan:

```text
python_version + requirements.txt -> dependency_hash
dependency_hash -> reusable runtime image key
framework -> validation adapter
```

In production, this plan should map to a cached Docker image:

```text
python:3.10-slim + requirements hash abc123 -> registry/runtime:abc123
```

## Run Locally

From this folder:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8090
```

Then open:

```text
http://localhost:8090/docs
```

## Run With Per-Job Docker Isolation

Local mode is useful for quick prototype checks. Docker mode is the real BYOM version-isolation path.

Start the API in Docker executor mode:

```powershell
$env:BYOM_EXECUTOR="docker"
$env:BYOM_RUNTIME_AGENT="true"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8090
```

When a job runs, the worker will:

```text
1. read requirements.txt
2. infer Python version and framework adapter
3. build/reuse image tag by dependency_hash
4. run docker container with model/dataset mounted
5. read validation_result.json from mounted artifacts
```

Images are cached with tags like:

```text
byom-runtime:<dependency_hash>
```

Intermediate build/run files are stored under:

```text
.runtime_cache/
```

You need Docker Desktop running for this mode.

## Runtime Orchestration Agent

Docker mode includes a small rules-based runtime agent when:

```powershell
$env:BYOM_RUNTIME_AGENT="true"
```

It is not an LLM. It does deterministic retries for known runtime failures.

Current behavior:

```text
1. Try initial runtime plan
2. If Docker/pip says package requires a higher Python version, retry with that Python version
3. If unpickling fails because a custom class is missing, fail with a clear missing supporting code diagnosis
4. Store attempt history under .runtime_cache/orchestration_history/
5. Store successful runtime images under .runtime_cache/runtime_knowledge.json
```

This is the prototype version of the "Codex-like" background reasoning we discussed, but implemented as safe rules instead of free-form code editing.

## Run The Sample Jobs

Create trained sklearn sample models and validation CSVs:

```bash
python scripts/create_sample_assets.py
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8090
```

Submit the sample job:

```bash
curl -X POST http://127.0.0.1:8090/validation-jobs \
  -H "Content-Type: application/json" \
  --data @samples/job_payload_sklearn_joblib.json
```

Check the result:

```bash
curl http://127.0.0.1:8090/validation-jobs/demo-sklearn-joblib-001
```

More cases are available in:

```text
samples/TEST_CASES.md
samples/cases/
```

The Kaggle logistic regression sample can be refreshed with:

```powershell
.\.venv\Scripts\python.exe scripts\download_kaggle_logistic_regression.py
```

Supported framework adapter details are here:

```text
docs_supported_frameworks.md
samples/framework_templates/
```

## Upload Your Own Model And Dataset

Use this endpoint when you want to inject your own files directly:

```text
POST /validation-jobs/upload
```

Required multipart fields:

```text
model_file
dataset_file
job_id
tenant_id
model_id
target_column
```

Optional multipart fields:

```text
requirements_file
framework
task_type
model_type
numeric_features       Comma-separated or JSON array, for example Age,Duration
categorical_features   Comma-separated or JSON array, for example Sex,Housing
model_params           JSON object string
dimensions             JSON object string
```

PowerShell example:

```powershell
curl.exe -X POST http://127.0.0.1:8090/validation-jobs/upload `
  -F "job_id=my-upload-job-001" `
  -F "tenant_id=tenant-demo" `
  -F "model_id=my-model" `
  -F "framework=sklearn" `
  -F "task_type=classification" `
  -F "target_column=Risk" `
  -F "numeric_features=Age,Job,Credit amount,Duration" `
  -F "categorical_features=Sex,Housing,Saving accounts,Checking account,Purpose" `
  -F "model_file=@samples/cases/sklearn_classification_pipeline/model.joblib" `
  -F "dataset_file=@samples/cases/sklearn_classification_pipeline/validation.csv" `
  -F "requirements_file=@samples/cases/sklearn_classification_pipeline/requirements.txt"
```

Then check:

```powershell
curl.exe http://127.0.0.1:8090/validation-jobs/my-upload-job-001
```

## Important Security Note

Loading `.joblib`/pickle files can execute Python code. In production, validation must run inside an isolated container with:

- read-only model/data mounts where possible
- no host filesystem access
- no platform secrets
- CPU/memory/time limits
- controlled network access
