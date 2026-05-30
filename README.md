# PsyMAS-Aberrance (v0.6.0)

**PsyMAS-Aberrance** is a Streamlit app for **forensic psychometrics**: it fits an IRT model (to produce ψ item parameters) and runs **aberrance detection agents**, then produces a **Forensic Verdict** (LLM-assisted).

This repository is structured as a two-process system:

- **UI**: Streamlit (`ui.py`) — runs locally (Windows-friendly) or in Docker/Railway.
- **Backend**: FastAPI (`backend_service.py`) — runs on **Linux** (recommended via Docker or Railway). It hosts R/rpy2 + the LangGraph/agent logic.

Why split it?

- Windows+rpy2+R is unstable for long-running, multi-agent workflows.
- Running the backend in Linux (Docker/Railway) makes Detect runs reliable.
- Detect Progress polls the backend and auto-refreshes until the job finishes.

---

## What’s new in 0.6.0

- **Railway & Docker Compose stack**: One-command local deploy (`docker compose up --build`) with backend + UI; `scripts/run_ui_railway.sh` for Streamlit behind proxies (Railway, Compose).
- **Scalable job storage**: Detect and IRT jobs use **Redis** (multi-replica) → **disk** (`PSYMAS_JOB_DIR`, multi-worker Gunicorn) → in-memory fallback; `/health` reports `job_store` and whether jobs are shared safely.
- **Async IRT**: `POST /irt/start` runs IRT in a subprocess; Preparation polls `GET /irt/{job_id}/status` and `GET /irt/{job_id}/result` (sync `POST /irt` remains for Tools/debug).
- **Forensic reporting**: Branded **Test-Taker Report (PDF)** (compact 12pt layout), **person-level CSV/ZIP export** with column legend and agent index guide, and fixed navigation to **Aberrance Summary** after Detect.
- **Pre-estimated ψ**: Upload item-parameter CSV in Preparation to skip IRT when ψ is already available.
- **Railway URL handling**: UI normalizes `PSYMAS_BACKEND_URL` (adds `https://` for public Railway domains to avoid POST→GET redirects on `/irt`).

---

## Features

- **Scenario**: Choose preset A, B, or D (Custom); optional LLM suggestion from a short description.
- **Preparation**: Upload Response + optional RT CSV; generate ψ (async IRT) or upload pre-estimated ψ; set compromised items / tampering when needed; select agents; start Detect.
- **Detect Progress**: Progress bar, flow diagram with per-agent status, auto-refresh until done; **Generate Report** opens Aberrance Summary.
- **Aberrance Summary**: Forensic Verdict, Collusion Graph and Effort Matrix (when data exists), Watchlist, agent-by-agent reports, examinee-level export, Test-Taker PDF.
- **Tools**: Aberrance / IRT / RT utilities + **Tools → Backend test** (sync and async IRT, detect polling).

---

## Quickstart (recommended on Windows)

### Option A — Docker Compose (backend + UI)

From the repo root:

```bash
cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY / GOOGLE_API_KEY as needed

docker compose up --build
```

First build installs R packages and can take **10–20 minutes**.

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:9000 |
| API docs | http://localhost:9000/docs |
| Health | http://localhost:9000/health |

On Windows, host port **9000** is used because TCP **8000** is often in an OS-reserved range (7971–8070).

Stop: `docker compose down`

**Backend only** (run Streamlit on the host with `uv`):

```bash
docker compose up --build backend
```

Then set `PSYMAS_BACKEND_URL=http://localhost:9000` in `.env` and run Streamlit locally (Option B, step 3).

Optional Redis (multi-worker / multi-replica backend):

```bash
# In .env: REDIS_URL=redis://redis:6379/0 and PSYMAS_GUNICORN_WORKERS=4
docker compose --profile redis up --build
```

### Option B — Manual Docker + host Streamlit

**1) Backend in Docker**

```bash
docker build -t psych-mas-backend .
docker run --rm -p 9000:8000 -e PORT=8000 psych-mas-backend
```

Default image CMD runs Gunicorn on port 8000 inside the container (mapped to host **9000** in the example above).

Verify: `http://localhost:9000/health` → `{"status":"ok", ...}`

**2) Configure `.env`**

```env
PSYMAS_BACKEND_URL=http://localhost:9000
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
```

**3) Run Streamlit on the host**

```bash
uv sync
uv run -- python -m streamlit run ui.py --server.port 8501
```

Use a UI port other than the backend API port.

---

## Deploy on Railway

Typical setup: **two services** from the same image (`psych-mas:latest`):

| Service | Command / notes |
|---------|-----------------|
| **Backend** | Default Dockerfile `CMD` (Gunicorn + uvicorn workers). Set `PORT`; optional `REDIS_URL` + `PSYMAS_GUNICORN_WORKERS` for scale. |
| **UI** | Override start command: `sh scripts/run_ui_railway.sh`. Set `PSYMAS_BACKEND_URL` to the backend’s **public HTTPS** URL (e.g. `https://your-api.up.railway.app`). |

Secrets: `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, and optionally `REDIS_URL` (Railway Redis plugin). Without Redis on multiple backend replicas, use a single replica or rely on disk job dir only within one container.

---

## Backend API

The backend service exposes:

- **`GET /health`**: `{status, job_store, shared_detect_jobs, async_irt, irt_endpoints}` — used by the UI for backend readiness.
- **`POST /irt/start`**: queues IRT in a subprocess; returns `{job_id, job_store}`.
- **`GET /irt/{job_id}/status`**: `{status, progress, error}`.
- **`GET /irt/{job_id}/result`**: `{status, result}` with `result.item_params` when done.
- **`POST /irt`**: synchronous IRT (legacy / Tools); returns ψ immediately.
- **`POST /detect`**: starts a detect job and returns `{run_id, job_store}`.
  - Requires `psi_data` (ψ from Preparation; backend does **not** re-fit IRT during Detect).
  - Runs agents **sequentially** to avoid concurrent rpy2/R failures.
- **`GET /detect/{run_id}/status`**: `{status, progress, irt_status, node_states, error}`.
- **`GET /detect/{run_id}/result`**: final result (`flags`, `final_report`, `psi_data`, etc.).

**Job store priority:** `REDIS_URL` (or Railway Redis env vars) → `PSYMAS_JOB_DIR` (default `/tmp/psymas_detect_jobs` in Docker) → in-memory (single worker only).

---

## ψ (IRT parameters) rules

To avoid re-estimating IRT during Detect:

- **Preparation → Generate item parameters** calls `POST /irt/start` (or upload a ψ CSV) and stores `item_params` in Streamlit session.
- **Preparation → Detect** sends those params as `psi_data` in `POST /detect`.
- If `psi_data` is missing, the backend returns an error (“Missing psi_data …”).

---

## LLM configuration (Forensic Verdict)

The Forensic Verdict uses the currently selected provider:

- **OpenRouter**: set `OPENROUTER_API_KEY`
- **Google Gemini**: set `GOOGLE_API_KEY`

Notes:

- `ui.py` loads `.env` at startup using `python-dotenv`.
- OpenRouter model lists live in `mmls.py`.
- If keys are missing, the verdict falls back to a rule-based summary.

---

## Tools → Backend test

The **Tools → Backend test** page helps you debug:

- whether ψ exists in session,
- the exact JSON payload being sent to `/detect`,
- direct calls to `/health`, `/irt/start`, `/irt`, `/detect`, `/status`, and `/result`.

---

## Project layout

| Path | Purpose |
|------|---------|
| `ui.py` | Streamlit UI (Preparation, Detect Progress, dashboards, Tools) |
| `backend_service.py` | FastAPI backend (`/health`, `/irt`, `/detect`, job storage) |
| `graph.py` | IRT + aberrance agent implementations and synthesizer |
| `mmls.py` | OpenRouter model metadata for the UI |
| `Dockerfile` | Linux image (R + R packages + Python deps) |
| `docker-compose.yml` | Local stack: `backend` + `ui` (+ optional `redis` profile) |
| `scripts/run_ui_railway.sh` | Streamlit entrypoint for Railway / Compose UI service |
| `.env.example` | Template for API keys and backend URL |
| `install_r_packages.R`, `r_packages.txt` | R package install (container) |
| `pyproject.toml` | Python dependencies (package version **0.6.0**) |

---

## Security

- Do **not** commit `.env` (API keys). Use `.env` locally or platform secrets in production.
