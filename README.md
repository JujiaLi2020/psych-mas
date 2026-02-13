# PsyMAS-Aberrance (v0.3.x)

**PsyMAS-Aberrance** is a Streamlit app for **forensic psychometrics**: it fits an IRT model (to produce ψ item parameters) and runs **8 aberrance detection agents**, then produces a **Forensic Verdict** (LLM-assisted).

This repository is structured as a two-process system:

- **UI**: Streamlit (`ui.py`) — runs locally (Windows-friendly).
- **Backend**: FastAPI (`backend_service.py`) — runs on **Linux** (recommended via Docker). It hosts R/rpy2 + the LangGraph/agent logic.

Why split it?

- Windows+rpy2+R is unstable for long-running, multi-agent workflows.
- Running the backend in Linux (Docker/Railway) makes Detect runs reliable.
- Detect Progress can poll a backend job even if the user navigates between pages.

---

## Features

- **Preparation**: upload Response + RT CSV, generate ψ, configure LLM, start Detect.
- **Detect Progress**: progress bar + LangGraph-style agent tree with per-agent node statuses.
- **Aberrance Summary**: dashboard of agent flags + narrative forensic report (“Forensic Verdict”).
- **Tools**: Aberrance / IRT / RT utilities + **Tools → Backend test** for troubleshooting.

---

## Quickstart (recommended on Windows)

### 1) Start the backend in Docker (Linux + R)

From the repo root:

```bash
docker build -t psych-mas .
docker run --rm -p 8000:8000 psych-mas uvicorn backend_service:app --host 0.0.0.0 --port 8000
```

Verify:

- `http://localhost:8000/health` → `{"status":"ok"}`
- `http://localhost:8000/docs` → FastAPI docs (should include `/irt` and `/detect`)

### 2) Configure local environment

Create/update `.env` in the repo root (do **not** commit it):

```env
PSYMAS_BACKEND_URL=http://localhost:8000
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
```

After changing `.env`, restart Streamlit.

### 3) Run Streamlit locally

In another terminal:

```bash
uv sync
uv run streamlit run ui.py --server.port 4000
```

Use any free port (don’t use `8000` — that’s the backend).

---

## Backend API

The backend service exposes:

- **`GET /health`**: lightweight health check used by the UI (“LangGraph Agents” status).
- **`POST /irt`**: runs IRT in the backend and returns ψ (`result.item_params`).
- **`POST /detect`**: starts a detect job and returns `{run_id}`.
  - Requires `psi_data` (ψ is generated on Preparation; backend does **not** re-fit IRT during Detect).
  - Runs agents **sequentially** to avoid concurrent rpy2/R failures.
- **`GET /detect/{run_id}/status`**: returns `{status, progress, irt_status, node_states, error}`.
- **`GET /detect/{run_id}/result`**: returns the final result (`flags`, `final_report`, `psi_data`, etc.).

---

## ψ (IRT parameters) rules

To avoid re-estimating IRT during Detect:

- **Preparation → Generate item parameters** calls `POST /irt` and stores `item_params` in Streamlit session.
- **Preparation → Detect** sends those params as `psi_data` in `POST /detect`.
- If `psi_data` is missing, the backend returns an error (“Missing psi_data …”).

---

## LLM configuration (Forensic Verdict)

The Forensic Verdict uses the currently selected provider:

- **OpenRouter**: set `OPENROUTER_API_KEY`
- **Google Gemini**: set `GOOGLE_API_KEY`

Notes:

- `ui.py` loads `.env` at startup using `python-dotenv`.
- If keys are missing, the verdict falls back to a rule-based summary.

---

## Tools → Backend test

The **Tools → Backend test** page helps you debug:

- whether ψ exists in session,
- the exact JSON payload being sent to `/detect`,
- direct calls to `/health`, `/irt`, `/detect`, `/status`, and `/result`.

---

## Project layout

| Path | Purpose |
|------|---------|
| `ui.py` | Streamlit UI (Preparation, Detect Progress, dashboards, Tools) |
| `backend_service.py` | FastAPI backend (`/health`, `/irt`, `/detect`, polling) |
| `graph.py` | IRT + aberrance agent implementations and synthesizer |
| `Dockerfile` | Linux image (R + R packages + Python deps) |
| `install_r_packages.R`, `r_packages.txt` | R package install (container) |
| `pyproject.toml` | Python dependencies |

---

## Security

- Do **not** commit `.env` (API keys). Use `.env` locally or platform secrets in production.
