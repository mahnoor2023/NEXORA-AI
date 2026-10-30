from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Protocol


class ObjectStorage(Protocol):
    local: bool

    def put(self, key: str, body: bytes) -> None: ...
    def get(self, key: str, max_bytes: int | None = None) -> bytes: ...
    def delete(self, key: str) -> None: ...


def _safe_key(key: str) -> str:
    try:
        return uuid.UUID(key).hex
    except (ValueError, TypeError, AttributeError):
        raise KeyError("Invalid object key") from None


class LocalStorage:
    local = True

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / f"{_safe_key(key)}.csv").resolve()
        if path.parent != self.root:
            raise KeyError("Invalid object key")
        return path

    def put(self, key: str, body: bytes) -> None:
        import os
        import tempfile
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(key)
        fd, temporary = tempfile.mkstemp(prefix="upload-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def get(self, key: str, max_bytes: int | None = None) -> bytes:
        path = self._path(key)
        with path.open("rb") as body:
            return body.read() if max_bytes is None else body.read(max_bytes + 1)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage:
    local = False

    def __init__(self, bucket: str, region: str, endpoint_url: str | None):
        import boto3
        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region or None, endpoint_url=endpoint_url or None)

    @staticmethod
    def _object_key(key: str) -> str:
        return f"datasets/{_safe_key(key)}.csv"

    def put(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._object_key(key), Body=body, ContentType="text/csv")

    def get(self, key: str, max_bytes: int | None = None) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._object_key(key))
            body = response["Body"]
            try:
                return body.read() if max_bytes is None else body.read(max_bytes + 1)
            finally:
                body.close()
        except self.client.exceptions.NoSuchKey:
            raise KeyError("Object not found") from None

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._object_key(key))


_adapter: ObjectStorage | None = None
_config: tuple[str, ...] | None = None


def get_storage(local_root: Path | None = None) -> ObjectStorage:
    global _adapter, _config
    mode = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    selected_root = str(local_root) if local_root is not None else os.getenv("LOCAL_STORAGE_PATH", "")
    config = (mode, selected_root, os.getenv("OBJECT_STORAGE_BUCKET", ""), os.getenv("OBJECT_STORAGE_REGION", ""), os.getenv("OBJECT_STORAGE_ENDPOINT_URL", ""))
    if _adapter is not None and _config == config:
        return _adapter
    if mode == "local":
        root = Path(config[1] or (Path(__file__).resolve().parents[2] / "runtime" / "datasets"))
        _adapter = LocalStorage(root)
    elif mode == "s3":
        if not config[2]:
            raise RuntimeError("OBJECT_STORAGE_BUCKET is required when STORAGE_BACKEND=s3")
        _adapter = S3Storage(config[2], config[3], config[4] or None)
    else:
        raise RuntimeError("STORAGE_BACKEND must be local or s3")
    _config = config
    return _adapter
