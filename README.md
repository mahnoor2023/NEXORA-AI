# NEXORA AI

NEXORA AI is a full-stack business analytics workspace built with Next.js, TypeScript, FastAPI, Pandas, scikit-learn, and SQLAlchemy. It supports account signup/login, persistent owner-scoped CSV profiling and safe cleaning, supervised baseline ML, and a deterministic Pandas Data Analysis Assistant. SQLite and private H: local file storage remain the development defaults; PostgreSQL and private S3-compatible storage are configurable for hosted environments.

## What persists

User accounts, salted password hashes, session-token hashes, preferences, dataset metadata, and aggregate analysis/training history are stored in `runtime/nexora.sqlite3` by default. Set `DATABASE_URL` to PostgreSQL in production; otherwise the app uses SQLite and `DATABASE_PATH`. Idempotent, additive startup migrations preserve existing user and dataset rows. They do not copy a local SQLite database into a newly provisioned PostgreSQL database. Validated original CSV files use `runtime/datasets/` locally or a private S3-compatible bucket when `STORAGE_BACKEND=s3`; only generated UUID object keys are stored. Storage is limited to 250 MB and 100 datasets per account; the local backend also has a 2 GB total limit. Configure quotas with `MAX_USER_DATASET_STORAGE_MB`, `MAX_DATASETS_PER_USER`, and `MAX_TOTAL_DATASET_STORAGE_MB`. Row previews, chat content, model artifacts, and ML test-set predictions are not persisted. Reopened datasets rebuild their analytics from the validated source file.

Passwords use PBKDF2-HMAC-SHA256 with per-account random salts. Sessions use random opaque tokens in HttpOnly cookies; only token hashes are stored. Browser writes enforce the configured Origin allowlist. Production requires HTTPS, `COOKIE_SECURE=true`, `COOKIE_SAMESITE=none`, an exact `CORS_ORIGINS` allowlist, PostgreSQL, and S3-compatible storage. Use frontend and API custom domains under the same registrable domain (for example, `app.example.com` and `api.example.com`): they remain cross-origin but are same-site, which avoids many browsers' third-party-cookie blocking on unrelated Vercel and Render domains.

## Requirements

- Node.js 20.9+
- `uv` and Python 3.12
- Dataset chat is available without an API key. It uses deterministic Pandas operations for counts, columns, missing values, summaries, category frequencies, numeric aggregations, and top-N groups. Unsupported or ambiguous requests receive a clear clarification response. It is not generative AI.
- GroqCloud and Ollama adapters remain optional integrations, but Dataset Chat does not call either provider. A configured Groq key is reported as configured but unverified; no status request or chat request sends data to Groq. NEXORA does not install Ollama or download models.

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

Copy environment examples once, then set secrets only in the backend file or deployment secret manager:

```powershell
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
```

No API key is needed for dataset chat. Optional hosted LLM credentials remain backend-only and are not used by the deterministic assistant. Do not reuse a key previously rejected by Groq, and do not copy keys into `.env.local`, a `NEXT_PUBLIC_*` variable, source control, browser storage, or frontend code.

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
- The Pandas Data Analysis Assistant uses a bounded set of recognized request types and direct Pandas calculations against the authenticated user's saved CSV. It sends no CSV data or question to an LLM provider. It never executes generated Python, SQL, or shell and reports unsupported requests without inventing results. Chat is limited to 20,000 rows and 50 columns.

## Optional LLM integrations

GroqCloud and Ollama adapter code is retained, but it is separate from and currently not connected to Dataset Chat. The authenticated status endpoint does not probe a provider; it reports only whether a Groq key is configured and labels it unverified. The deterministic assistant stays available if credentials are missing, invalid, rate-limited, or otherwise unavailable. No live Groq inference has been tested as part of this project work. Do not add a paid key or reuse the previously rejected key without separately reviewing provider account access, terms, and costs.

## Local Ollama (optional)

The server must have its models already installed. Before starting Ollama, configure its model storage under H: and disable cloud features if appropriate:

```powershell
$project = (Resolve-Path .).Path
$env:OLLAMA_MODELS = "$project\runtime\ollama-models"
$env:OLLAMA_NO_CLOUD = "1"
ollama serve
```

Ollama configuration is retained for future optional integration work, but Dataset Chat does not currently route questions to it. No model was downloaded by this project setup.

## Online deployment preparation (not deployed)

The repository includes a Vercel project descriptor and a Render Blueprint for a free FastAPI web service. Hosted app data must be external to Render's ephemeral filesystem: use PostgreSQL and private S3-compatible object storage. `APP_ENV=production` deliberately refuses SQLite and local file storage, so a misconfigured deployment fails at startup instead of silently losing accounts or uploads.

### Free-tier reality check

- [Vercel Hobby](https://vercel.com/docs/plans/hobby) is free for personal, non-commercial use and has usage limits; it is not a general free commercial hosting plan.
- [Render free web services](https://render.com/docs/free) sleep after 15 minutes idle, may take about a minute to wake, have an ephemeral filesystem, and use up to 750 free instance hours per workspace monthly. Free Render Postgres expires after 30 days, so it is not a durable database choice. High outbound traffic to external databases or object storage can also trigger suspension.
- [Neon Free](https://neon.com/blog/neon-free-plan-1-gb-per-project) currently includes 1 GB per project and bounded compute; free compute can run out or sleep. Check the current provider dashboard before relying on it.
- [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/) currently includes 10 GB-month standard storage, 1 million Class A requests, and 10 million Class B requests monthly. Usage beyond that is billable, and [creating an R2 subscription requires a checkout flow](https://developers.cloudflare.com/r2/get-started/). Do not enable it unless you can ensure account setup and usage stay within your no-purchase constraint; this repo cannot enforce a provider billing cap.

These tiers are suitable for constrained evaluation, not a reliability or zero-cost guarantee. No service account was created and no deployment was performed. Object storage is only active when the server's authenticated runtime status reports S3-compatible object storage; local development reports local server storage. No deployed cloud storage configuration has been verified in this work.

### Configure and deploy when ready

1. Create a Vercel project from this repository with the Next.js root at the repository root. Set `NEXT_PUBLIC_API_URL` to the HTTPS FastAPI URL ending in `/api/v1`. This is a public API address, never a secret.
2. Create a PostgreSQL database with TLS and backups. Set its SQLAlchemy-compatible `DATABASE_URL` only in the API host's server-side environment. Existing H: SQLite data is not copied automatically: keep it intact and plan an explicit, reviewed export/import with row counts and password/session handling before switching users.
3. Create a private S3-compatible bucket and restricted application credentials. For Cloudflare R2, configure endpoint `https://<account-id>.r2.cloudflarestorage.com`, region `auto`, bucket name, access key, and secret key. Do not enable public bucket access. R2 encrypts objects at rest; credentials stay only in the backend environment.
4. Create a Render Web Service using the included `render.yaml`. Set the prompted secret values in Render: `CORS_ORIGINS` to the exact frontend origin, `DATABASE_URL`, `OBJECT_STORAGE_BUCKET`, `OBJECT_STORAGE_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`; set `GROQ_API_KEY` only if hosted Analyst Chat is wanted. Confirm `STORAGE_BACKEND=s3`, `COOKIE_SECURE=true`, and `COOKIE_SAMESITE=none` remain set. Do not put API secrets in Vercel variables or any `NEXT_PUBLIC_*` variable.
5. For authentication across the separate frontend and API origins, use HTTPS custom hostnames under the same registrable domain (such as `app.example.com` and `api.example.com`). Keep exact origin allowlisting. `SameSite=None; Secure` permits cross-site cookie sending, but browsers can block third-party cookies entirely when the two default hostnames are on different sites.
6. Check `/api/v1/health`, signup/login/logout, upload/reopen/download/delete, ownership isolation, database restart persistence, and storage restart persistence before inviting users. Keep the H: SQLite file and uploads as the untouched local copy until a migration is verified.

Required production variables are listed in [backend/.env.example](backend/.env.example); the frontend public API variable is in [.env.example](.env.example). The API validates the PostgreSQL, S3, CORS, and cookie configuration at startup and returns a generic 503 health response without connection details when PostgreSQL is unavailable. Schema setup is additive and idempotent (`schema_migrations` version 1); it does not remove or alter existing user rows.

Remaining manual setup: choose providers and regions, confirm account eligibility and free quotas, provision database and private bucket, provide credentials securely, configure same-site custom DNS domains, review the existing local data migration plan, and validate actual production behavior. Add rate limiting, backups/restore drills, monitoring, privacy/retention policy, and provider-specific billing alerts before real users. GroqCloud remains optional and may involve separate provider terms or billing; local Ollama is not available to hosted Render by default. No Ollama model downloads or paid AI APIs were added.
