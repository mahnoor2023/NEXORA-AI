from __future__ import annotations

import hashlib
import json
import copy
import os
import secrets
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("DATABASE_PATH", "runtime/nexora.sqlite3"))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH
COOKIE_NAME = "nexora_session"
SESSION_DAYS = 7
STORAGE_ROOT = ROOT / "runtime" / "datasets"
MAX_DATASET_STORAGE_BYTES = int(os.getenv("MAX_USER_DATASET_STORAGE_MB", "250")) * 1024 * 1024
MAX_TOTAL_DATASET_STORAGE_BYTES = int(os.getenv("MAX_TOTAL_DATASET_STORAGE_MB", "2048")) * 1024 * 1024
MAX_DATASETS_PER_USER = int(os.getenv("MAX_DATASETS_PER_USER", "100"))
MAX_SOURCE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


class DatasetUnavailableError(Exception):
    pass


class DatasetQuotaError(Exception):
    pass


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=10000")
    return db


def initialize() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL, password_hash TEXT NOT NULL, password_salt TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, expires_at TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
        CREATE TABLE IF NOT EXISTS preferences(user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, preferences_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, content_hash TEXT NOT NULL, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL, row_count INTEGER NOT NULL, column_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, storage_key TEXT, available INTEGER NOT NULL DEFAULT 0, last_opened_at TEXT, UNIQUE(owner_id, content_hash), UNIQUE(id, owner_id));
        CREATE TABLE IF NOT EXISTS analysis_history(id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, owner_id TEXT NOT NULL, kind TEXT NOT NULL, summary_json TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(dataset_id, owner_id) REFERENCES datasets(id, owner_id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS history_owner ON analysis_history(owner_id, created_at DESC);
        """)
        # Older installations have metadata only. Keep those rows and mark them
        # unavailable until the owner reuploads the source CSV.
        columns = {row["name"] for row in db.execute("PRAGMA table_info(datasets)")}
        if "storage_key" not in columns:
            db.execute("ALTER TABLE datasets ADD COLUMN storage_key TEXT")
        if "available" not in columns:
            db.execute("ALTER TABLE datasets ADD COLUMN available INTEGER NOT NULL DEFAULT 0")
        if "last_opened_at" not in columns:
            db.execute("ALTER TABLE datasets ADD COLUMN last_opened_at TEXT")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000).hex()
    return digest, salt.hex()


def verify_password(password: str, digest: str, salt_hex: str) -> bool:
    import hmac
    candidate, _ = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, digest)


def create_session(db: sqlite3.Connection, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    created = now()
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
    db.execute("INSERT INTO sessions VALUES(?,?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user_id, expires, created))
    return token


def public_user(row: sqlite3.Row) -> dict[str, str]:
    return {"id": row["id"], "email": row["email"], "display_name": row["display_name"], "created_at": row["created_at"]}


def _storage_path(storage_key: str) -> Path:
    try:
        parsed = uuid.UUID(storage_key)
    except (ValueError, TypeError, AttributeError):
        raise DatasetUnavailableError("The stored dataset file is unavailable. Reupload the CSV to restore it.") from None
    path = (STORAGE_ROOT / f"{parsed.hex}.csv").resolve()
    if path.parent != STORAGE_ROOT.resolve():
        raise DatasetUnavailableError("The stored dataset file is unavailable. Reupload the CSV to restore it.")
    return path


def _write_source(raw: bytes, storage_key: str) -> Path:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = _storage_path(storage_key)
    fd, temporary = tempfile.mkstemp(prefix="upload-", suffix=".tmp", dir=STORAGE_ROOT)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def save_analysis(user_id: str, raw: bytes, filename: str, size_bytes: int, analysis: dict[str, Any], kind: str, source_row_count: int | None = None, source_column_count: int | None = None) -> tuple[str, str]:
    fingerprint = hashlib.sha256(raw).hexdigest()
    timestamp = now()
    wrote_path: Path | None = None
    try:
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id,storage_key,available FROM datasets WHERE owner_id=? AND content_hash=?", (user_id, fingerprint)).fetchone()
            is_new = row is None
            dataset_id = row["id"] if row else str(uuid.uuid4())
            storage_key = row["storage_key"] if row else dataset_id
            available_path = None
            stale_path: Path | None = None
            if row and row["available"] and storage_key:
                try:
                    candidate = _storage_path(storage_key)
                    stale_path = candidate
                    if candidate.is_file() and candidate.stat().st_size == size_bytes and size_bytes <= MAX_SOURCE_BYTES:
                        with candidate.open("rb") as existing_file:
                            existing_raw = existing_file.read(MAX_SOURCE_BYTES + 1)
                        if len(existing_raw) == size_bytes and hashlib.sha256(existing_raw).hexdigest() == fingerprint:
                            available_path = candidate
                except DatasetUnavailableError:
                    available_path = None
            if available_path is None:
                if stale_path:
                    stale_path.unlink(missing_ok=True)
                stored = db.execute("SELECT COALESCE(SUM(size_bytes),0) total FROM datasets WHERE owner_id=? AND available=1 AND id<>?", (user_id, dataset_id)).fetchone()["total"]
                if stored + size_bytes > MAX_DATASET_STORAGE_BYTES:
                    raise DatasetQuotaError(f"Account storage quota is {MAX_DATASET_STORAGE_BYTES // (1024 * 1024)} MB. Delete an existing dataset or upload a smaller CSV.")
                total_stored = db.execute("SELECT COALESCE(SUM(size_bytes),0) total FROM datasets WHERE available=1 AND id<>?", (dataset_id,)).fetchone()["total"]
                if total_stored + size_bytes > MAX_TOTAL_DATASET_STORAGE_BYTES:
                    raise DatasetQuotaError("The local dataset storage limit has been reached. Delete an unused dataset or contact the administrator.")
                if is_new:
                    count = db.execute("SELECT COUNT(*) total FROM datasets WHERE owner_id=?", (user_id,)).fetchone()["total"]
                    if count >= MAX_DATASETS_PER_USER:
                        raise DatasetQuotaError(f"An account can store at most {MAX_DATASETS_PER_USER} datasets.")
                storage_key = str(uuid.uuid4())
                wrote_path = _write_source(raw, storage_key)
            if row:
                db.execute("UPDATE datasets SET filename=?,size_bytes=?,row_count=?,column_count=?,updated_at=?,storage_key=?,available=1 WHERE id=? AND owner_id=?", (filename, size_bytes, source_row_count if source_row_count is not None else analysis["row_count"], source_column_count if source_column_count is not None else analysis["column_count"], timestamp, storage_key, dataset_id, user_id))
            else:
                db.execute("INSERT INTO datasets(id,owner_id,content_hash,filename,size_bytes,row_count,column_count,created_at,updated_at,storage_key,available,last_opened_at) VALUES(?,?,?,?,?,?,?,?,?,?,1,NULL)", (dataset_id, user_id, fingerprint, filename, size_bytes, source_row_count if source_row_count is not None else analysis["row_count"], source_column_count if source_column_count is not None else analysis["column_count"], timestamp, timestamp, storage_key))
            history_id = str(uuid.uuid4())
            summary = {k: analysis[k] for k in ("row_count", "column_count", "total_missing", "duplicate_rows") if k in analysis}
            if kind == "training":
                summary.update({k: copy.deepcopy(analysis[k]) for k in ("task", "target", "best_model", "primary_metric", "models") if k in analysis})
                for model in summary.get("models", []):
                    model.pop("predictions", None)
            db.execute("INSERT INTO analysis_history VALUES(?,?,?,?,?,?)", (history_id, dataset_id, user_id, kind, json.dumps(summary), timestamp))
            return dataset_id, history_id
    except Exception:
        if wrote_path:
            try:
                wrote_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def dataset_source(user_id: str, dataset_id: str) -> tuple[bytes, str, dict[str, Any]]:
    with connect() as db:
        row = db.execute("SELECT * FROM datasets WHERE id=? AND owner_id=?", (dataset_id, user_id)).fetchone()
    if not row:
        raise KeyError("Dataset not found.")
    if not row["available"] or not row["storage_key"]:
        raise DatasetUnavailableError("This older dataset has metadata only. Reupload its CSV to restore preview, ML, and AI access.")
    try:
        path = _storage_path(row["storage_key"])
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise DatasetUnavailableError("The saved CSV exceeds the configured upload limit. Reupload an allowed file to restore access.")
        with path.open("rb") as source:
            raw = source.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise DatasetUnavailableError("The saved CSV exceeds the configured upload limit. Reupload an allowed file to restore access.")
    except (OSError, DatasetUnavailableError):
        with connect() as db:
            db.execute("UPDATE datasets SET available=0,storage_key=NULL WHERE id=? AND owner_id=?", (dataset_id, user_id))
        try:
            if "path" in locals():
                path.unlink(missing_ok=True)
        except OSError:
            pass
        raise DatasetUnavailableError("The saved CSV is missing, invalid, or exceeds the configured limit. Reupload it to restore preview, ML, and AI access.") from None
    if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["content_hash"]:
        with connect() as db:
            db.execute("UPDATE datasets SET available=0,storage_key=NULL WHERE id=? AND owner_id=?", (dataset_id, user_id))
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise DatasetUnavailableError("The saved CSV failed its integrity check. Reupload it to restore access.")
    with connect() as db:
        db.execute("UPDATE datasets SET last_opened_at=? WHERE id=? AND owner_id=?", (now(), dataset_id, user_id))
    return raw, row["filename"], dict(row)


def dataset_public_record(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in ("id", "filename", "size_bytes", "row_count", "column_count", "created_at", "updated_at", "last_opened_at", "available")}


def active_dataset_id(user_id: str) -> str | None:
    with connect() as db:
        row = db.execute("SELECT id FROM datasets WHERE owner_id=? ORDER BY COALESCE(last_opened_at,updated_at) DESC LIMIT 1", (user_id,)).fetchone()
    return row["id"] if row else None


def delete_dataset_record(user_id: str, dataset_id: str) -> bool:
    with connect() as db:
        row = db.execute("SELECT storage_key FROM datasets WHERE id=? AND owner_id=?", (dataset_id, user_id)).fetchone()
        if not row:
            return False
        storage_key = row["storage_key"]
        db.execute("DELETE FROM datasets WHERE id=? AND owner_id=?", (dataset_id, user_id))
    if storage_key:
        try:
            _storage_path(storage_key).unlink(missing_ok=True)
        except (OSError, DatasetUnavailableError):
            pass
    return True


def delete_user_datasets(user_id: str) -> None:
    with connect() as db:
        keys = [row["storage_key"] for row in db.execute("SELECT storage_key FROM datasets WHERE owner_id=?", (user_id,)) if row["storage_key"]]
        db.execute("DELETE FROM users WHERE id=?", (user_id,))
    for storage_key in keys:
        try:
            _storage_path(storage_key).unlink(missing_ok=True)
        except (OSError, DatasetUnavailableError):
            pass
