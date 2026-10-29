# Functionality audit

Scope inspected: App Router pages, shared navigation/header, dashboard upload/results controls, ML Studio, analyst chat, account forms, and every route in `backend/app/main.py`.

## UI inventory

| Area / control | Status | Behavior |
| --- | --- | --- |
| Signup and login forms | Working | Validate input, call FastAPI, receive HttpOnly session cookie, show server errors and loading state. |
| Protected route gate | Working | Checks `/auth/me`; unauthenticated users go to `/login`. |
| Sidebar and mobile navigation | Working | Links to Overview, Datasets, Exploration, and Settings. |
| New Analysis | Working | Offers explicit choices to upload a separate CSV or reopen a saved dataset without replacing the active dataset. |
| Profile menu | Working | Shows database profile, opens Settings, and revokes the current session on logout. |
| CSV browse/drop upload | Working | Checks extension/size in browser and backend; displays loading/errors; stores the original immutable CSV under an owner-scoped generated key and clears stale preview before another upload. |
| Summary, charts, columns, preview search/paging | Working | Uses computed backend result; preview search and paging are functional. The no-op Filter and column-options controls were removed. |
| Safe cleaning dialog | Working | Options produce a derived analysis; original upload bytes remain unchanged. |
| Export preview | Working | Downloads only the displayed first 10 rows and escapes spreadsheet formula prefixes. |
| Dataset cards and reopen | Working | Cards open the saved CSV by owner-checked dataset ID; the API verifies file size/hash and rebuilds analytics. Missing legacy sources request reupload. |
| ML Studio | Working | Calls real training endpoint and displays comparison, metrics, predictions, and feature importance. Requires current CSV to be re-uploaded after browser/server restart. |
| AI Analyst chat | Working when configured | Uses real Ollama tool calls and explicit unavailable/error states. Requires a compatible local model and current CSV upload. |
| Settings profile/password/preferences | Working | Persists to SQLite; password update rotates sessions; default page and compact-table preference are stored. |
| Delete account | Working | Requires password and typed `DELETE`; database cascades remove sessions, metadata, preferences, and history. |
| Dataset deletion | Working | Requires confirmation and deletes only the signed-in owner's row, CSV file, and history. |
| Notifications, Help & support, fake identity/count, no-op filter/options | Removed | No deceptive placeholder interactions remain. |

## API inventory

| Endpoint | Status | Access and persistence |
| --- | --- | --- |
| `GET /api/v1/health` | Working | Public liveness only. |
| `POST /api/v1/auth/signup`, `POST /login`, `POST /logout`, `GET /me` | Working | Cookie sessions, password hash, duplicate/credential errors. |
| `PATCH /api/v1/auth/profile`, `POST /password`, `GET/PATCH /preferences`, `DELETE /account` | Working | Authenticated user only; database-backed; account deletion cascades. |
| `POST /api/v1/datasets/analyze` | Working | Authenticated, bounded CSV; computes analytics and saves owner-scoped metadata/aggregate history. |
| `POST /api/v1/datasets/train` | Working | Authenticated; bounded, real ML pipeline; saves metrics summary only. |
| `GET /api/v1/datasets`, `GET /active`, `GET /history`, `GET /{dataset_id}/analysis`, `GET /{dataset_id}/source`, `DELETE /{dataset_id}` | Working | Authenticated and owner-filtered. Reopen verifies bounded file size and SHA-256; delete removes the owner's file and history. |
| `GET /api/v1/agent/status`, `POST /api/v1/agent/chat` | Working when configured | Authenticated; local Ollama only; bounded analysis tools, no arbitrary code execution. |

## Verification boundaries

Backend automated tests cover signup/login/logout, duplicate accounts, password/profile/preferences/account deletion, protected routes, hostile Origin rejection, account isolation, persistence through a new client, CSV analytics/cleaning, ML classification/regression, and agent behavior. ESLint, TypeScript, and production compilation/build are run separately. A real interactive browser session was not automated in this environment; final browser navigation, cookie handling in the user's browser, and responsive presentation should be smoke-tested locally.

## Remaining product limits

- SQLite is the implemented runtime. PostgreSQL is not yet a supported adapter; see the migration checklist in the README.
- CSV content is retained as an immutable private file with 250 MB/100-dataset per-account and 2 GB local total defaults. Metadata-only records from earlier versions remain and explicitly require owner reupload.
- Local Ollama was not available during this audit and no model was downloaded.
- Browser-based end-to-end tests are not configured; automated UI confidence currently comes from lint, typecheck, build, and the API suite.
