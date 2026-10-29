# NEXORA AI

NEXORA AI is a local-first data analysis workspace built with Next.js, TypeScript, Tailwind CSS, FastAPI, Pandas, scikit-learn, and SQLite. It supports account signup/login, CSV profiling and safe cleaning, supervised baseline ML, an optional local Ollama analyst, and per-user analysis history.

## What persists

User accounts, salted password hashes, session-token hashes, preferences, dataset metadata, and aggregate analysis/training history are stored in `runtime/nexora.sqlite3` by default. Validated original CSV files are stored immutably under `runtime/datasets/` with generated identifiers; paths are never returned to clients. Storage is limited to 250 MB and 100 datasets per account plus a 2 GB local total by default, configurable with `MAX_USER_DATASET_STORAGE_MB`, `MAX_DATASETS_PER_USER`, and `MAX_TOTAL_DATASET_STORAGE_MB`. Row previews, chat content, model artifacts, and ML test-set predictions are not persisted. Reopened datasets rebuild their analytics from the validated source file.

Passwords use PBKDF2-HMAC-SHA256 with per-account random salts. Sessions use random opaque tokens in HttpOnly, SameSite=Lax cookies; only token hashes are stored. Browser writes enforce the configured Origin allowlist. In production set `APP_ENV=production`, `COOKIE_SECURE=true`, HTTPS, and exact trusted `CORS_ORIGINS` values. Use an HTTPS reverse proxy and add rate limiting and monitoring before exposing the API publicly.

## Requirements

- Node.js 20.9+
- `uv` and Python 3.12
- Ollama and an installed tool-calling model are optional. NEXORA does not download models and reports provider/model errors when unavailable.

## Install on H:

Run these commands from `H:\NEXORA AI`. They keep temporary files and caches on H:.

```powershell
$project = (Resolve-Path .).Path
New-Item -ItemType Directory -Force -Path "$project\runtime\tmp", "$project\runtime\uv-cache", "$project\runtime\pip-cache", "$project\runtime\npm-cache", "$project\runtime\pytest-cache", "$project\runtime\ollama-models" | Out-Null
$env:TEMP = "$project\runtime\tmp"
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = "$project\runtime\uv-cache"
$env:PIP_CACHE_DIR = "$project\runtime\pip-cache"
$env:npm_config_cache = "$project\runtime\npm-cache"
uv venv --python 3.12 --allow-existing backend/.venv
uv pip install --python backend/.venv/Scripts/python.exe -r backend/requirements.txt
npm.cmd ci --no-audit --no-fund
```

Copy environment examples once:

```powershell
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
```

`backend/.env` defaults the SQLite database to `runtime/nexora.sqlite3` under this project. Keep secrets and deployment configuration in backend-only environment variables; never add secrets to `NEXT_PUBLIC_*` variables.

## Run locally

Open two PowerShell terminals in the project root. In each, set the H: temporary/cache paths shown in Install.

Terminal 1, API:

```powershell
backend/.venv/Scripts/uvicorn.exe app.main:app --reload --port 8000 --app-dir backend --env-file backend/.env
```

Terminal 2, web app:

```powershell
npm.cmd run dev
```

Open `http://localhost:3000`, create an account, and upload a CSV. API documentation is at `http://localhost:8000/docs`. Logout and account deletion are available from the profile menu and Settings page.

## Verify

Run sequentially from the root; Next.js writes generated route types during its build.

```powershell
$project = (Resolve-Path .).Path
$env:TEMP = "$project\runtime\tmp"
$env:TMP = $env:TEMP
$env:UV_CACHE_DIR = "$project\runtime\uv-cache"
$env:PIP_CACHE_DIR = "$project\runtime\pip-cache"
$env:npm_config_cache = "$project\runtime\npm-cache"
backend/.venv/Scripts/python.exe -m pytest backend/tests -o "cache_dir=$project\runtime\pytest-cache"
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run build
```

## ML and analysis limits

- Each upload is limited to 25 MB, 500,000 rows, and 200 columns. Each account can store at most 250 MB across 100 datasets, and the local file store at most 2 GB, by default. Training has stricter ML-specific row/column and bounded runtime limits.
- The ML Studio uses leakage-safe scikit-learn preprocessing pipelines, fixed-seed train/test splits, and two lightweight baselines. Metrics describe the held-out split and are not a guarantee of future performance. Small, imbalanced, high-cardinality, or time-dependent datasets can make metrics unstable or unsuitable.
- Training predictions and feature importance are returned for the active browser request; prediction rows are not retained in history. Reopen the stored dataset to train again. The analyst agent can retrieve saved aggregate training results after a process restart.
- Data cleaning returns a derived preview and never overwrites the source upload.
- The AI analyst is unavailable until a local Ollama server and compatible installed model are configured. It uses bounded read-only tools and does not run generated Python, SQL, or shell. Rule-based ML insights are labeled as rule-based, not generated text.

## Local Ollama (optional)

The server must have its models already installed. Before starting Ollama, configure its model storage under H: and disable cloud features if appropriate:

```powershell
$project = (Resolve-Path .).Path
$env:OLLAMA_MODELS = "$project\runtime\ollama-models"
$env:OLLAMA_NO_CLOUD = "1"
ollama serve
```

Set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REQUEST_TIMEOUT_SECONDS` in `backend/.env`. No model was downloaded by this project setup. Review a model name, download size, and available H: storage before choosing to install one. See [Ollama model storage](https://docs.ollama.com/faq#where-are-models-stored).

## PostgreSQL migration for online deployment

The current application database adapter is SQLite; PostgreSQL is **not currently a supported runtime backend**. Do not deploy this SQLite file on an ephemeral server filesystem or claim that local Ollama is available in a cloud deployment. For an online deployment, first implement and test a PostgreSQL adapter for `backend/app/database.py` (parameter binding, transactions, schema creation, timestamps, and foreign-key cascades), add a pinned PostgreSQL driver, and run the complete API test suite against PostgreSQL. Then provision PostgreSQL with TLS, migrate the `users`, `sessions`, `preferences`, `datasets`, and `analysis_history` tables using a reviewed export/import or migration script, verify row counts and foreign-key ownership, set `DATABASE_URL` server-side, and test restart/backup/restore and account deletion. Move the private CSV store to an authenticated, encrypted persistent volume with the same ownership, quota, integrity, and deletion guarantees. This is a documented deployment migration path, not a claim of current PostgreSQL support.

## Production readiness checklist

- [ ] Implement and test the PostgreSQL adapter and schema migration described above.
- [ ] Configure HTTPS, `APP_ENV=production`, `COOKIE_SECURE=true`, and exact CORS origins.
- [ ] Add API rate limits for signup/login and upload/training endpoints; monitor authentication and upload failures.
- [ ] Configure database backups, restore drills, retention, and operational alerting.
- [ ] Review privacy policy, account deletion behavior, security headers, and deployment secrets.
- [ ] If using Ollama, deploy a private reachable Ollama host with an H:-equivalent persistent model volume; verify networking and model availability. Cloud deployments do not automatically have access to a workstation's local Ollama service.
- [ ] Run the production build and a browser smoke test against the final deployment.
