from __future__ import annotations

import io
import json
import os
import asyncio
import threading
import hashlib
import json as json_module
import re
import sqlite3
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent import AgentError, MAX_AGENT_SECONDS, MAX_ML_COLUMNS, MAX_ML_ROWS, ollama_base_url, ollama_status, run_data_analyst_agent
from app.ml import MAX_TRAIN_SECONDS, MLValidationError, train_and_evaluate
from app.state import forget_user_training_results, remember_training_result, retrieve_training_result
from app import database
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_COLUMNS = 200
MAX_ROWS = 500_000
PREVIEW_ROWS = 10
TRAINING_LOCK = threading.Lock()
AGENT_LOCK = threading.Lock()

app = FastAPI(title="NEXORA AI API", version="0.1.0", description="Local-first data exploration API")
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"], allow_headers=["*"])


@app.middleware("http")
async def enforce_browser_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method in {"POST", "PATCH", "DELETE", "PUT"} and origin and origin not in origins:
        return JSONResponse(status_code=403, content={"detail": "Request origin is not allowed."})
    return await call_next(request)


@app.on_event("startup")
def startup() -> None:
    database.initialize()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Signup(StrictModel):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("display_name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name is required.")
        return value


class Login(StrictModel):
    email: str
    password: str = Field(min_length=1, max_length=128)


class ProfileUpdate(StrictModel):
    display_name: str = Field(min_length=1, max_length=80)


class PasswordUpdate(StrictModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=128)


class PreferencesUpdate(StrictModel):
    default_view: str = Field(pattern="^(overview|datasets|exploration)$")
    compact_tables: bool


class DeleteAccount(StrictModel):
    password: str
    confirmation: str


def current_user(request: Request) -> dict[str, str]:
    token = request.cookies.get(database.COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    hashed = hashlib.sha256(token.encode()).hexdigest()
    with database.connect() as db:
        row = db.execute("SELECT users.* FROM sessions JOIN users ON users.id=sessions.user_id WHERE sessions.token_hash=? AND sessions.expires_at>?", (hashed, database.now())).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Your session has expired. Please log in again.")
    return database.public_user(row)


User = Annotated[dict[str, str], Depends(current_user)]


def issue_cookie(response: Response, token: str) -> None:
    secure_default = os.getenv("APP_ENV", "development").lower() == "production"
    secure = os.getenv("COOKIE_SECURE", str(secure_default)).lower() == "true"
    response.set_cookie(database.COOKIE_NAME, token, max_age=database.SESSION_DAYS * 86400, httponly=True, secure=secure, samesite="lax", path="/")


def clear_cookie(response: Response) -> None:
    secure_default = os.getenv("APP_ENV", "development").lower() == "production"
    secure = os.getenv("COOKIE_SECURE", str(secure_default)).lower() == "true"
    response.delete_cookie(database.COOKIE_NAME, path="/", httponly=True, secure=secure, samesite="lax")


@app.post("/api/v1/auth/signup")
def signup(payload: Signup, response: Response) -> dict[str, Any]:
    password_hash, salt = database.hash_password(payload.password)
    user_id = str(__import__("uuid").uuid4())
    timestamp = database.now()
    try:
        with database.connect() as db:
            db.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?)", (user_id, payload.email, payload.display_name, password_hash, salt, timestamp, timestamp))
            db.execute("INSERT INTO preferences VALUES(?,?,?)", (user_id, json_module.dumps({"default_view": "overview", "compact_tables": False}), timestamp))
            token = database.create_session(db, user_id)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from None
    issue_cookie(response, token)
    return {"user": {"id": user_id, "email": payload.email, "display_name": payload.display_name, "created_at": timestamp}}


@app.post("/api/v1/auth/login")
def login(payload: Login, response: Response) -> dict[str, Any]:
    with database.connect() as db:
        row = db.execute("SELECT * FROM users WHERE email=?", (payload.email.strip().lower(),)).fetchone()
        if not row or not database.verify_password(payload.password, row["password_hash"], row["password_salt"]):
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        token = database.create_session(db, row["id"])
    issue_cookie(response, token)
    return {"user": database.public_user(row)}


@app.post("/api/v1/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    token = request.cookies.get(database.COOKIE_NAME)
    if token:
        with database.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
    clear_cookie(response)
    return {"ok": True}


@app.get("/api/v1/auth/me")
def me(user: User) -> dict[str, Any]:
    return {"user": user}


@app.patch("/api/v1/auth/profile")
def update_profile(payload: ProfileUpdate, user: User) -> dict[str, Any]:
    name = payload.display_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    with database.connect() as db:
        db.execute("UPDATE users SET display_name=?,updated_at=? WHERE id=?", (name, database.now(), user["id"]))
        row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    return {"user": database.public_user(row)}


@app.post("/api/v1/auth/password")
def update_password(payload: PasswordUpdate, request: Request, response: Response, user: User) -> dict[str, bool]:
    with database.connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        if not database.verify_password(payload.current_password, row["password_hash"], row["password_salt"]):
            raise HTTPException(status_code=422, detail="Current password is incorrect.")
        digest, salt = database.hash_password(payload.new_password)
        db.execute("UPDATE users SET password_hash=?,password_salt=?,updated_at=? WHERE id=?", (digest, salt, database.now(), user["id"]))
        db.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
        token = database.create_session(db, user["id"])
    # A new session is issued on the response via request.state; route response is set below.
    issue_cookie(response, token)
    return {"ok": True}


@app.patch("/api/v1/auth/preferences")
def update_preferences(payload: PreferencesUpdate, user: User) -> dict[str, Any]:
    value = {"default_view": payload.default_view, "compact_tables": payload.compact_tables}
    with database.connect() as db:
        db.execute("INSERT INTO preferences VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET preferences_json=excluded.preferences_json,updated_at=excluded.updated_at", (user["id"], json_module.dumps(value), database.now()))
    return {"preferences": value}


@app.get("/api/v1/auth/preferences")
def get_preferences(user: User) -> dict[str, Any]:
    with database.connect() as db:
        row = db.execute("SELECT preferences_json FROM preferences WHERE user_id=?", (user["id"],)).fetchone()
    return {"preferences": json_module.loads(row["preferences_json"]) if row else {"default_view": "overview", "compact_tables": False}}


@app.delete("/api/v1/auth/account")
def delete_account(payload: DeleteAccount, user: User, response: Response) -> dict[str, bool]:
    if payload.confirmation != "DELETE":
        raise HTTPException(status_code=422, detail="Type DELETE to confirm account removal.")
    with database.connect() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
        if not database.verify_password(payload.password, row["password_hash"], row["password_salt"]):
            raise HTTPException(status_code=422, detail="Password is incorrect.")
    database.delete_user_datasets(user["id"])
    forget_user_training_results(user["id"])
    clear_cookie(response)
    return {"ok": True}


@app.get("/api/v1/datasets")
def list_datasets(user: User) -> dict[str, Any]:
    with database.connect() as db:
        rows = db.execute("SELECT d.*, (SELECT kind FROM analysis_history h WHERE h.dataset_id=d.id AND h.owner_id=d.owner_id ORDER BY h.created_at DESC LIMIT 1) last_kind FROM datasets d WHERE owner_id=? ORDER BY COALESCE(last_opened_at,updated_at) DESC", (user["id"],)).fetchall()
    return {"datasets": [{**database.dataset_public_record(row), "last_kind": row["last_kind"]} for row in rows]}


@app.get("/api/v1/datasets/active")
def get_active_dataset(user: User) -> dict[str, str | None]:
    return {"dataset_id": database.active_dataset_id(user["id"])}


@app.get("/api/v1/datasets/history")
def list_history(user: User) -> dict[str, Any]:
    with database.connect() as db:
        rows = db.execute("SELECT h.id,h.kind,h.summary_json,h.created_at,d.id dataset_id,d.filename FROM analysis_history h JOIN datasets d ON d.id=h.dataset_id AND d.owner_id=h.owner_id WHERE h.owner_id=? ORDER BY h.created_at DESC LIMIT 200", (user["id"],)).fetchall()
    return {"history": [{**dict(row), "summary": json_module.loads(row["summary_json"])} for row in rows]}


@app.get("/api/v1/datasets/{dataset_id}/analysis")
async def reopen_dataset(dataset_id: str, user: User) -> JSONResponse:
    try:
        raw, filename, metadata = database.dataset_source(user["id"], dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except database.DatasetUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "DATASET_REUPLOAD_REQUIRED", "message": str(exc)}) from None
    from starlette.datastructures import Headers
    from io import BytesIO
    upload = UploadFile(filename=filename, file=BytesIO(raw), headers=Headers({"content-type": "text/csv"}))
    validated_raw, frame = await read_csv_upload(upload)
    result = analyze_dataframe(frame, filename, len(validated_raw))
    result.update({"dataset_id": dataset_id, "dataset_available": bool(metadata["available"])})
    return JSONResponse(result)


@app.get("/api/v1/datasets/{dataset_id}/source")
def download_dataset_source(dataset_id: str, user: User) -> StreamingResponse:
    try:
        raw, _, _ = database.dataset_source(user["id"], dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except database.DatasetUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "DATASET_REUPLOAD_REQUIRED", "message": str(exc)}) from None
    return StreamingResponse(io.BytesIO(raw), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=dataset.csv"})


@app.delete("/api/v1/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user: User) -> dict[str, bool]:
    if not database.delete_dataset_record(user["id"], dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"ok": True}


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def read_csv_upload(file: UploadFile, row_limit: int = MAX_ROWS) -> tuple[bytes, pd.DataFrame]:
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Please upload a CSV file.")
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="The uploaded file must be a CSV.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", nrows=row_limit + 1, on_bad_lines="error")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise HTTPException(status_code=422, detail="Could not parse this file as a valid UTF-8 CSV.") from None
    if len(df) > row_limit:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {row_limit:,} row limit.")
    if len(df.columns) == 0 or (len(df.columns) == 1 and str(df.columns[0]).startswith("Unnamed:")):
        raise HTTPException(status_code=422, detail="The CSV must contain a header row and at least one column.")
    if len(df.columns) > MAX_COLUMNS:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {MAX_COLUMNS} column limit.")
    df.columns = [str(column).strip() for column in df.columns]
    if any(not column for column in df.columns):
        raise HTTPException(status_code=422, detail="Column names cannot be blank.")
    if df.columns.duplicated().any():
        raise HTTPException(status_code=422, detail="Column names must be unique.")
    return raw, df


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def analyze_dataframe(df: pd.DataFrame, filename: str, size_bytes: int) -> dict[str, Any]:
    row_count = len(df)
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        boolean = pd.api.types.is_bool_dtype(series)
        datetime = pd.api.types.is_datetime64_any_dtype(series)
        kind = "numeric" if numeric else "boolean" if boolean else "datetime" if datetime else "categorical"
        summary = None
        histogram: list[dict[str, Any]] = []
        if numeric:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                quantiles = values.quantile([0.25, 0.5, 0.75])
                summary = {"count": int(values.count()), "mean": _json_value(values.mean()), "std": _json_value(values.std()), "min": _json_value(values.min()), "q25": _json_value(quantiles.loc[0.25]), "median": _json_value(quantiles.loc[0.5]), "q75": _json_value(quantiles.loc[0.75]), "max": _json_value(values.max())}
                if values.nunique() == 1:
                    histogram = [{"bin": str(values.iloc[0]), "count": int(values.count())}]
                else:
                    counts = pd.cut(values, bins=min(10, values.nunique()), include_lowest=True)
                    histogram = [{"bin": str(interval), "count": int(count)} for interval, count in counts.value_counts(sort=False).items()]
        frequencies = series.dropna().astype(str).value_counts().head(8)
        columns.append({"name": str(name), "kind": kind, "dtype": str(series.dtype), "missing": int(series.isna().sum()), "missing_pct": round(float(series.isna().mean() * 100), 2) if row_count else 0, "unique": int(series.nunique(dropna=True)), "numeric": summary, "top_values": [{"value": str(value), "count": int(count)} for value, count in frequencies.items()], "histogram": histogram})
    preview = [{str(key): _json_value(value) for key, value in row.items()} for row in df.head(PREVIEW_ROWS).to_dict(orient="records")]
    return {"filename": filename, "size_bytes": size_bytes, "row_count": row_count, "column_count": len(df.columns), "total_missing": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()), "columns": columns, "preview": preview}


@app.post("/api/v1/datasets/analyze")
async def analyze_dataset(file: Annotated[UploadFile, File()], user: User, cleaning: Annotated[str | None, Form()] = None) -> JSONResponse:
    filename = file.filename or "upload.csv"
    options = {"drop_duplicates": False, "fill_numeric": False, "fill_categorical": False}
    if cleaning:
        try:
            parsed = json.loads(cleaning)
            if not isinstance(parsed, dict) or any(key not in options or not isinstance(value, bool) for key, value in parsed.items()):
                raise ValueError
            options.update(parsed)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=422, detail="Cleaning options must be valid boolean settings.") from None
    raw, df = await read_csv_upload(file)
    original_df = df
    # Work on a fresh frame; the uploaded bytes and browser file are never modified.
    if options["drop_duplicates"]:
        df = df.drop_duplicates().copy()
    if options["fill_numeric"] or options["fill_categorical"]:
        df = df.copy()
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]) and options["fill_numeric"]:
                median = df[column].median()
                if pd.notna(median):
                    df[column] = df[column].fillna(median)
            elif options["fill_categorical"]:
                mode = df[column].mode(dropna=True)
                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])
    result = analyze_dataframe(df, filename, len(raw))
    try:
        dataset_id, history_id = database.save_analysis(user["id"], raw, filename, len(raw), result, "analysis", len(original_df), len(original_df.columns))
    except database.DatasetQuotaError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    result["dataset_id"] = dataset_id
    result["history_id"] = history_id
    return JSONResponse(result)


@app.post("/api/v1/datasets/train")
async def train_dataset(
    file: Annotated[UploadFile, File()],
    target: Annotated[str, Form()],
    user: User,
    task: Annotated[str, Form()] = "auto",
) -> JSONResponse:
    filename = file.filename or "upload.csv"
    if task not in {"auto", "classification", "regression"}:
        raise HTTPException(status_code=422, detail="Task must be auto, classification, or regression.")
    raw, df = await read_csv_upload(file, MAX_ML_ROWS)
    if not TRAINING_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another training run is in progress. Retry when it finishes.")
    worker = asyncio.create_task(asyncio.to_thread(train_and_evaluate, df, target, task))
    try:
        result = await asyncio.wait_for(
            asyncio.shield(worker),
            timeout=MAX_TRAIN_SECONDS + 5,
        )
    except asyncio.TimeoutError:
        worker.add_done_callback(lambda _: TRAINING_LOCK.release())
        raise HTTPException(status_code=408, detail="Training exceeded the 25-second request limit. Try a smaller dataset.") from None
    except MLValidationError as exc:
        TRAINING_LOCK.release()
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (ValueError, TypeError) as exc:
        TRAINING_LOCK.release()
        raise HTTPException(status_code=422, detail=f"This dataset cannot be trained with the selected options: {exc}") from None
    except asyncio.CancelledError:
        worker.add_done_callback(lambda _: TRAINING_LOCK.release())
        raise
    except Exception:
        TRAINING_LOCK.release()
        raise
    else:
        TRAINING_LOCK.release()
    result["filename"] = filename
    try:
        dataset_id, history_id = database.save_analysis(user["id"], raw, filename, len(raw), {**result, "row_count": len(df), "column_count": len(df.columns)}, "training")
    except database.DatasetQuotaError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    result["dataset_id"] = dataset_id
    result["history_id"] = history_id
    remember_training_result(raw, {**result, "owner_id": user["id"]})
    return JSONResponse(result)


@app.get("/api/v1/agent/status")
async def analyst_agent_status(user: User) -> dict[str, Any]:
    return ollama_status()


@app.post("/api/v1/agent/chat")
async def analyst_agent_chat(
    file: Annotated[UploadFile, File()],
    question: Annotated[str, Form()],
    user: User,
) -> JSONResponse:
    if not question.strip() or len(question) > 1_000:
        raise HTTPException(status_code=422, detail={"code": "INVALID_QUESTION", "message": "Question must contain 1 to 1,000 characters."})
    status = ollama_status()
    if status["state"] != "ready":
        status_codes = {"ollama_unavailable": "OLLAMA_UNAVAILABLE", "model_unavailable": "OLLAMA_MODEL_UNAVAILABLE", "configuration_error": "OLLAMA_CONFIGURATION_ERROR"}
        code = status_codes.get(status["state"], "OLLAMA_UNAVAILABLE")
        raise HTTPException(status_code=503, detail={"code": code, "message": status["message"]})
    filename = file.filename or "upload.csv"
    raw, df = await read_csv_upload(file, MAX_ML_ROWS)
    if len(df.columns) > MAX_ML_COLUMNS:
        raise HTTPException(status_code=413, detail={"code": "ANALYSIS_ERROR", "message": f"Agent analysis is limited to {MAX_ML_COLUMNS} columns."})
    if not AGENT_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail={"code": "AGENT_BUSY", "message": "An analyst question is already being processed. Retry when it finishes."})
    worker = asyncio.create_task(
        asyncio.to_thread(
            run_data_analyst_agent,
            question,
            df,
            filename,
            _training_result_for_user(user["id"], raw),
            status["model"],
            ollama_base_url(),
        )
    )
    try:
        result = await asyncio.wait_for(asyncio.shield(worker), timeout=MAX_AGENT_SECONDS + 5)
    except asyncio.TimeoutError:
        worker.add_done_callback(lambda _: AGENT_LOCK.release())
        raise HTTPException(status_code=408, detail={"code": "AGENT_TIMEOUT", "message": "The analyst agent exceeded its execution time limit."}) from None
    except AgentError as exc:
        AGENT_LOCK.release()
        status_code = 503 if exc.code.startswith("OLLAMA_") else 408 if exc.code == "AGENT_TIMEOUT" else 422 if exc.code in {"INVALID_QUESTION", "INVALID_TOOL_ARGUMENTS", "TOOL_CALL_LIMIT", "ANALYSIS_ERROR", "TOOL_RESULT_LIMIT"} else 502
        raise HTTPException(status_code=status_code, detail={"code": exc.code, "message": str(exc)}) from None
    except asyncio.CancelledError:
        worker.add_done_callback(lambda _: AGENT_LOCK.release())
        raise
    except Exception:
        AGENT_LOCK.release()
        raise HTTPException(status_code=502, detail={"code": "AGENT_ANALYSIS_ERROR", "message": "The analyst agent could not complete this request."}) from None
    else:
        AGENT_LOCK.release()
    return JSONResponse(result)


def _training_result_for_user(user_id: str, raw: bytes) -> dict[str, Any] | None:
    result = retrieve_training_result(raw)
    if result and result.get("owner_id") == user_id:
        return result
    fingerprint = hashlib.sha256(raw).hexdigest()
    with database.connect() as db:
        row = db.execute("SELECT summary_json FROM analysis_history h JOIN datasets d ON d.id=h.dataset_id AND d.owner_id=h.owner_id WHERE h.owner_id=? AND d.content_hash=? AND h.kind='training' ORDER BY h.created_at DESC LIMIT 1", (user_id, fingerprint)).fetchone()
    return json_module.loads(row["summary_json"]) if row else None
