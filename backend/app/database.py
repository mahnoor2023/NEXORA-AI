from __future__ import annotations

import hashlib
import json
import copy
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine, Row, URL
from sqlalchemy.exc import IntegrityError

from app.storage import get_storage

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("DATABASE_PATH", "runtime/nexora.sqlite3"))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT / DB_PATH
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
COOKIE_NAME = "nexora_session"
SESSION_DAYS = 7
STORAGE_ROOT = Path(os.getenv("LOCAL_STORAGE_PATH", str(ROOT / "runtime" / "datasets")))
if not STORAGE_ROOT.is_absolute():
    STORAGE_ROOT = ROOT / STORAGE_ROOT
MAX_DATASET_STORAGE_BYTES = int(os.getenv("MAX_USER_DATASET_STORAGE_MB", "250")) * 1024 * 1024
MAX_TOTAL_DATASET_STORAGE_BYTES = int(os.getenv("MAX_TOTAL_DATASET_STORAGE_MB", "2048")) * 1024 * 1024
MAX_DATASETS_PER_USER = int(os.getenv("MAX_DATASETS_PER_USER", "100"))
MAX_SOURCE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

_engine: Engine | None = None
_engine_url: str | None = None


class DatasetUnavailableError(Exception):
    pass


class DatasetQuotaError(Exception):
    pass


def integrity_error_types() -> tuple[type[BaseException], ...]:
    return (IntegrityError, sqlite3.IntegrityError)


def _storage():
    return get_storage(STORAGE_ROOT)


def _storage_path(storage_key: str) -> Path:
    from app.storage import LocalStorage
    adapter = _storage()
    if not isinstance(adapter, LocalStorage):
        raise DatasetUnavailableError("Local file paths are unavailable for cloud object storage.")
    try:
        return adapter._path(storage_key)
    except KeyError:
        raise DatasetUnavailableError("The stored dataset file is unavailable. Reupload the CSV to restore it.") from None


def _database_url() -> str:
    configured = DATABASE_URL or os.getenv("DATABASE_URL", "").strip()
    if configured:
        if configured.startswith("postgres://"):
            configured = "postgresql+psycopg://" + configured[len("postgres://"):]
        elif configured.startswith("postgresql://"):
            configured = "postgresql+psycopg://" + configured[len("postgresql://"):]
        return configured
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return URL.create("sqlite", database=str(path))


def _get_engine() -> Engine:
    global _engine, _engine_url
    url = _database_url()
    url_key = str(url)
    if _engine is None or _engine_url != url_key:
        if _engine is not None:
            _engine.dispose()
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
        if _engine.dialect.name == "sqlite":
            @event.listens_for(_engine, "connect")
            def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=10000")
                cursor.close()
        _engine_url = url_key
    return _engine


class DBConnection:
    """Small qmark-compatible adapter used by the existing parameterized SQL."""

    def __init__(self, connection: Connection):
        self.connection = connection

    @staticmethod
    def _statement(query: str, parameters: tuple | list | dict | None):
        if parameters is None:
            return text(query), {}
        if isinstance(parameters, dict):
            return text(query), parameters
        pieces = query.split("?")
        if len(pieces) - 1 != len(parameters):
            raise ValueError("SQL parameter count does not match placeholders")
        statement = pieces[0]
        values: dict[str, Any] = {}
        for index, value in enumerate(parameters):
            name = f"p{index}"
            statement += f":{name}" + pieces[index + 1]
            values[name] = value
        return text(statement), values

    def execute(self, query: str, parameters: tuple | list | dict | None = None):
        statement, values = self._statement(query, parameters)
        return self.connection.execute(statement, values).mappings()

    def lock_user(self, user_id: str) -> None:
        if self.connection.dialect.name == "sqlite":
            # A no-op update obtains SQLite's writer lock before quota reads.
            self.execute("UPDATE users SET updated_at=updated_at WHERE id=?", (user_id,))
        else:
            self.execute("SELECT id FROM users WHERE id=? FOR UPDATE", (user_id,)).fetchone()

    def __enter__(self) -> "DBConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.connection.close()


@contextmanager
def connect() -> Iterator[DBConnection]:
    connection = _get_engine().connect()
    try:
        yield DBConnection(connection)
        if connection.in_transaction():
            connection.commit()
    except Exception:
        if connection.in_transaction():
            connection.rollback()
        raise
    finally:
        connection.close()


def _table_columns(db: DBConnection, table: str) -> set[str]:
    if db.connection.dialect.name == "sqlite":
        return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    rows = db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=?", (table,)).fetchall()
    return {row["column_name"] for row in rows}


def initialize() -> None:
    """Idempotent, additive schema migration; existing rows are never rewritten."""
    statements = [
        "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL, password_hash TEXT NOT NULL, password_salt TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, expires_at TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id)",
        "CREATE TABLE IF NOT EXISTS preferences(user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, preferences_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, content_hash TEXT NOT NULL, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL, row_count INTEGER NOT NULL, column_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, storage_key TEXT, available INTEGER NOT NULL DEFAULT 0, last_opened_at TEXT, UNIQUE(owner_id, content_hash), UNIQUE(id, owner_id))",
        "CREATE TABLE IF NOT EXISTS analysis_history(id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, owner_id TEXT NOT NULL, kind TEXT NOT NULL, summary_json TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(dataset_id, owner_id) REFERENCES datasets(id, owner_id) ON DELETE CASCADE)",
        "CREATE INDEX IF NOT EXISTS history_owner ON analysis_history(owner_id, created_at DESC)",
        "CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
    ]
    with connect() as db:
        for statement in statements:
            db.execute(statement)
        # Legacy SQLite installations get additive columns. PostgreSQL follows
        # the same guarded migration path and preserves all existing records.
        columns = _table_columns(db, "datasets")
        for name, declaration in (("storage_key", "TEXT"), ("available", "INTEGER NOT NULL DEFAULT 0"), ("last_opened_at", "TEXT")):
            if name not in columns:
                db.execute(f"ALTER TABLE datasets ADD COLUMN {name} {declaration}")
        db.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(1,?) ON CONFLICT(version) DO NOTHING", (now(),))


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


def create_session(db: DBConnection, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    created = now()
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat()
    db.execute("INSERT INTO sessions VALUES(?,?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user_id, expires, created))
    return token


def public_user(row: Row) -> dict[str, str]:
    return {"id": row["id"], "email": row["email"], "display_name": row["display_name"], "created_at": row["created_at"]}


def _storage_key(storage_key: str) -> str:
    try:
        return uuid.UUID(storage_key).hex
    except (ValueError, TypeError, AttributeError):
        raise DatasetUnavailableError("The stored dataset file is unavailable. Reupload the CSV to restore it.") from None


def save_analysis(user_id: str, raw: bytes, filename: str, size_bytes: int, analysis: dict[str, Any], kind: str, source_row_count: int | None = None, source_column_count: int | None = None) -> tuple[str, str]:
    fingerprint = hashlib.sha256(raw).hexdigest()
    timestamp = now()
    wrote_key: str | None = None
    storage = _storage()
    try:
        with connect() as db:
            db.lock_user(user_id)
            row = db.execute("SELECT id,storage_key,available FROM datasets WHERE owner_id=? AND content_hash=?", (user_id, fingerprint)).fetchone()
            is_new = row is None
            dataset_id = row["id"] if row else str(uuid.uuid4())
            storage_key = row["storage_key"] if row else uuid.uuid4().hex
            already_stored = False
            if row and row["available"] and storage_key:
                try:
                    old = storage.get(_storage_key(storage_key), MAX_SOURCE_BYTES)
                    already_stored = len(old) == size_bytes and hashlib.sha256(old).hexdigest() == fingerprint
                except (DatasetUnavailableError, OSError, KeyError):
                    already_stored = False
            if not already_stored:
                stored = db.execute("SELECT COALESCE(SUM(size_bytes),0) total FROM datasets WHERE owner_id=? AND available=1 AND id<>?", (user_id, dataset_id)).fetchone()["total"]
                if stored + size_bytes > MAX_DATASET_STORAGE_BYTES:
                    raise DatasetQuotaError(f"Account storage quota is {MAX_DATASET_STORAGE_BYTES // (1024 * 1024)} MB. Delete an existing dataset or upload a smaller CSV.")
                if storage.local:
                    total_stored = db.execute("SELECT COALESCE(SUM(size_bytes),0) total FROM datasets WHERE available=1 AND id<>?", (dataset_id,)).fetchone()["total"]
                    if total_stored + size_bytes > MAX_TOTAL_DATASET_STORAGE_BYTES:
                        raise DatasetQuotaError("The local dataset storage limit has been reached. Delete an unused dataset or contact the administrator.")
                if is_new:
                    count = db.execute("SELECT COUNT(*) total FROM datasets WHERE owner_id=?", (user_id,)).fetchone()["total"]
                    if count >= MAX_DATASETS_PER_USER:
                        raise DatasetQuotaError(f"An account can store at most {MAX_DATASETS_PER_USER} datasets.")
                storage_key = uuid.uuid4().hex
                storage.put(storage_key, raw)
                wrote_key = storage_key
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
        if wrote_key:
            try:
                storage.delete(wrote_key)
            except Exception:
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
        raw = _storage().get(_storage_key(row["storage_key"]), MAX_SOURCE_BYTES)
        if len(raw) > MAX_SOURCE_BYTES:
            raise DatasetUnavailableError("The saved CSV exceeds the configured upload limit. Reupload an allowed file to restore access.")
    except (OSError, KeyError, DatasetUnavailableError):
        with connect() as db:
            db.execute("UPDATE datasets SET available=0,storage_key=NULL WHERE id=? AND owner_id=?", (dataset_id, user_id))
        raise DatasetUnavailableError("The saved CSV is missing, invalid, or exceeds the configured limit. Reupload it to restore preview, ML, and AI access.") from None
    if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["content_hash"]:
        with connect() as db:
            db.execute("UPDATE datasets SET available=0,storage_key=NULL WHERE id=? AND owner_id=?", (dataset_id, user_id))
        raise DatasetUnavailableError("The saved CSV failed its integrity check. Reupload it to restore access.")
    with connect() as db:
        db.execute("UPDATE datasets SET last_opened_at=? WHERE id=? AND owner_id=?", (now(), dataset_id, user_id))
    return raw, row["filename"], dict(row)


def dataset_public_record(row: Row) -> dict[str, Any]:
    return {key: row[key] for key in ("id", "filename", "size_bytes", "row_count", "column_count", "created_at", "updated_at", "last_opened_at", "available")}


def active_dataset_id(user_id: str) -> str | None:
    with connect() as db:
        row = db.execute("SELECT id FROM datasets WHERE owner_id=? ORDER BY COALESCE(last_opened_at,updated_at) DESC LIMIT 1", (user_id,)).fetchone()
    return row["id"] if row else None


def delete_dataset_record(user_id: str, dataset_id: str) -> bool:
    storage = _storage()
    with connect() as db:
        db.lock_user(user_id)
        row = db.execute("SELECT storage_key FROM datasets WHERE id=? AND owner_id=?", (dataset_id, user_id)).fetchone()
        if not row:
            return False
        if row["storage_key"]:
            storage.delete(_storage_key(row["storage_key"]))
        db.execute("DELETE FROM datasets WHERE id=? AND owner_id=?", (dataset_id, user_id))
    return True


def delete_user_datasets(user_id: str) -> None:
    storage = _storage()
    with connect() as db:
        db.lock_user(user_id)
        keys = [row["storage_key"] for row in db.execute("SELECT storage_key FROM datasets WHERE owner_id=?", (user_id,)) if row["storage_key"]]
        for key in keys:
            storage.delete(_storage_key(key))
        db.execute("DELETE FROM users WHERE id=?", (user_id,))
