from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


class MaterializedPath:
    def __init__(self, source: str, local_path: Path) -> None:
        self.source = source
        self.local_path = local_path


def materialize_path(source: str, destination_dir: Path) -> MaterializedPath:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source.startswith("gs://"):
        return download_gcs_path(source, destination_dir)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local path does not exist: {source}")

    destination = destination_dir / source_path.name
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return MaterializedPath(source=source, local_path=destination)


def read_text_path(source: Optional[str]) -> str:
    if not source:
        return ""
    if source.startswith("gs://"):
        raise RuntimeError("GCS requirements download needs google-cloud-storage credentials in this prototype")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"requirements file does not exist: {source}")
    return path.read_text(encoding="utf-8")


def download_gcs_path(source: str, destination_dir: Path) -> MaterializedPath:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError("install google-cloud-storage to download gs:// paths") from exc

    without_scheme = source.removeprefix("gs://")
    bucket_name, blob_name = without_scheme.split("/", 1)
    destination = destination_dir / Path(blob_name).name

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(destination))
    return MaterializedPath(source=source, local_path=destination)

