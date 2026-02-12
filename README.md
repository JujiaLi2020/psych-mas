# Psych-MAS

**Psychometric Modeling Assistant System** — Load response and response-time data, describe your IRT model in natural language, confirm settings, then run the workflow. View item/person parameters, Wright Map, item fit, ICC plots, and RT histograms. Includes LLM-assisted prompt interpretation, Psych-MAS summaries, and APA report generation.

---

## Features

- **Natural-language prompt** — Describe your analysis (e.g. “2PL IRT, 4 items”) in the Psych-MAS Assistant; the app interprets it and suggests model settings.
- **IRT workflow** — Run IRT fitting (via R `mirt`), item fit, model fit (M2), item/person parameters, Wright Map, and ICC plots.
- **Response-time (RT) analysis** — Upload RT CSV; view latency flags and RT histograms.
- **Psych-MAS Summaries** — LLM summaries for each section (descriptive, model fit, item fit, Wright Map, Item Parameters & ICC, person parameters, latency, RT histograms). Choose **Google Gemini** or **OpenRouter** (free models) in the **Model engine** tab; lock a model for all analyses.
- **APA Report (PDF)** — Generate an APA-style report (Method, Results, tables and figures in UI order) with each Psych-MAS summary next to its related table or figure. Download as PDF.
- **Advanced Analysis** — Chat with the assistant about your results (Q&A over the analysis).
- **LangGraph API** — Optional: run the workflow graph via the LangGraph dev server and Studio UI.

---

## How to use the app

### 1. Install and run

```bash
# Install dependencies (with uv)
uv sync

# Run the Streamlit UI
uv run streamlit run ui.py
```

Without uv: `pip install -r requirements.txt` (or `pip install -e .`), then `python -m streamlit run ui.py` using your project venv.

The app opens in your browser (e.g. `http://localhost:8501`).

---

## Running with the LangGraph backend service (Detect Progress)

For long **Detect** runs (IRT + 8 aberrance agents), you can run a separate LangGraph backend so progress continues even if users change pages.

### 1. Start the backend service

From the project root:

```bash
# Install deps if not already done
uv sync

# Run the backend service (FastAPI + LangGraph)
uv run uvicorn backend_service:app --reload --port 8000
```

This service exposes:

- `GET /health` — lightweight health check (used by the UI “LangGraph Agents” status).
- `POST /irt` — run IRT in the backend and return ψ item parameters (`item_params`).
- `POST /detect` — start a new detection job in a worker process.
- `GET /detect/{run_id}/status` — poll job status, progress, and per‑agent node states.
- `GET /detect/{run_id}/result` — fetch the final LangGraph state (`flags`, `final_report`, ψ, etc.).

### 2. Run the Streamlit UI

In a second terminal:

```bash
uv run streamlit run ui.py
```

By default, the UI talks to the backend at:

```text
BACKEND_URL = "http://localhost:8000"
```

You can point it to a remote backend (e.g. in production) by setting:

```bash
export PSYMAS_BACKEND_URL="https://your-backend-host"
```

### 3. How Detect uses the backend

- On the **Preparation** page, clicking **Detect** sends a single `POST /detect` to the backend with:
  - responses, RT data, itemtype, compromised items, and model settings.
  - ψ item parameters (`psi_data`) generated on Preparation (backend does **not** re-estimate ψ).
  - The backend returns a `run_id`, which is stored in `st.session_state`.
- On the **Detect Progress** page:
  - The page polls `GET /detect/{run_id}/status` each rerun and:
    - updates the main progress bar,
    - colors the LangGraph‑style agent tree (router → 8 agents → manager),
    - updates IRT/RT status.
  - When status becomes `"done"`, it calls `GET /detect/{run_id}/result`, stores the result, and enables the **Go to Aberrance Summary** button.

Because the job runs in the backend service, **navigating to other pages no longer stops Detect**; the progress view simply re‑attaches to the existing job when you return.

### 2. Set up API keys (optional, for LLM features)

Copy `.env.example` to `.env` (if present) or create a `.env` file in the project root:

- **Google Gemini:** `GOOGLE_API_KEY` (and `GEMINI_MODEL` if you want a non-default model).
- **OpenRouter:** `OPENROUTER_API_KEY` — get a key at [openrouter.ai](https://openrouter.ai). Some free models work without a key; for higher limits or paid models, set the key.

The app reads these via `python-dotenv`; do **not** commit `.env` to Git.

### LangSmith (optional — for seeing runs when using LangGraph)

If you run `langgraph dev` and want to see traces/runs in **LangSmith** (LangChain’s tracing UI), add your LangSmith API key to `.env`:

```bash
LANGSMITH_API_KEY=your-langsmith-api-key
```

Get a key at [smith.langchain.com](https://smith.langchain.com) (sign up, then API keys in settings). Without `LANGSMITH_API_KEY`, the LangGraph server still runs; you just won’t see runs in LangSmith.

### 3. Use the workflow

1. **Left column — Psych-MAS Assistant**
   - **Prompt tab:** Enter your analysis description (e.g. “2PL model, 4 items”). Click **Analyze prompt** to interpret it and fill model settings.
   - **Model engine tab:** Choose **Google** or **OpenRouter**, pick a model, optionally **Test API key** and **Check model availability**. Use **Lock current model for all analyses** so the same model is used for prompt analysis and all Psych-MAS Summaries.

2. **Upload data**
   - **Response CSV:** Rows = persons, columns = items (0/1 or item names).
   - **Response-time CSV (optional):** Same structure as response; used for RT analysis and latency flags.

3. **Confirm settings**
   - Review model type, item count, and data preview. Click **Run workflow**.

4. **View results**
   - **Response Results tab:**  
     - **1. Descriptive Summary** — Basic info, item accuracy plot, response matrix; **Psych-MAS Summary** button for an LLM summary.  
     - **2. Model fit** — M2 fit table; **Psych-MAS Summary** button.  
     - **3. Item fit** — Item fit table; **Psych-MAS Summary** button.  
     - **4. Wright Map & Parameters** — Wright Map (with optional **Generate Analysis** for an LLM summary), Item Parameters & ICC (figure + table + **Psych-MAS Summary**), Person Parameters (figure + table + **Psych-MAS Summary**).  
     - **Advanced Analysis** — Chat with the assistant about your results.  
     - **APA Report** — Click **Generate APA Report (PDF)**, then **Download APA Report (PDF)**.
   - **RT Analysis tab:** Latency flags and RT histograms, each with a **Psych-MAS Summary** button.

5. **APA report**
   - Content order matches the UI; each LLM summary appears next to its related table or figure. Requires `reportlab` (included in `pyproject.toml`).

---

## LangGraph interface

The workflow is implemented as a LangGraph graph (`graph.py:app`). You can run it locally and use the LangGraph API and Studio UI.

### Start the LangGraph dev server

```bash
# From the project root (where langgraph.json is)
langgraph dev
```

- **API base URL:** `http://127.0.0.1:2024` (or the URL shown in the terminal).
- **LangSmith runs:** To see traces in LangSmith, add `LANGSMITH_API_KEY` to your `.env` (see [LangSmith (optional)](#langsmith-optional--for-seeing-runs-when-using-langgraph) above).
- **LangGraph Studio:** If you use the LangGraph Studio app, connect it to this server to inspect and run the graph visually.

### Config

- `langgraph.json` points to the graph: `./graph.py:app` (graph name `psych_workflow`).
- The Streamlit UI can run **without** the LangGraph server: it invokes the workflow directly via `graph.py`. Use `langgraph dev` when you want to call the graph via the API or Studio.

### Example: call the graph via API

Once `langgraph dev` is running, you can POST to the LangGraph API (e.g. invoke the `psych_workflow` graph) according to the [LangGraph API docs](https://langchain-ai.github.io/langgraph/concepts/langgraph_api/). The Streamlit app does not require the server to be up.

---

## Requirements

- **Python:** 3.13+ (see `pyproject.toml`; adjust if you use an older version).
- **Local (full IRT + Wright Map):** R on `PATH` plus R packages `mirt`, `WrightMap`, `psych`. System deps: `packages.txt` (apt). R packages: run once `Rscript install_r_packages.R` (or see `r_packages.txt`).
- **Streamlit Cloud:** No R; IRT fitting, ICC, and Wright Map are skipped. LLM and RT features work; set secrets (see below).

### Local R setup (for IRT, ICC, Wright Map)

1. **System packages (Debian/Ubuntu):**  
   `sudo apt-get update && sudo apt-get install -y $(cat packages.txt)`

2. **R packages (run once):**  
   `Rscript install_r_packages.R`  
   Or in R: `install.packages(c('mirt','WrightMap','psych'), repos='https://cloud.r-project.org')`

---

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (do **not** push `.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, **New app**.
3. Set **Main file path** to `ui.py`.
4. In the app **Settings** → **Secrets**, add:

```toml
GOOGLE_API_KEY = "your-google-api-key"
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

5. Deploy. Share the app URL; users can upload their own CSVs and run the workflow (without R-dependent features on Cloud).

---

## Deploy on Railway (Docker, full IRT)

Use **Railway** when you want **full IRT** (mirt, WrightMap, psych) in the cloud with GitHub version control and no cold starts. The repo includes a `Dockerfile` that installs R, R packages, and Python/Streamlit.

### 1. Prerequisites

- GitHub repo with this project (including `Dockerfile`, `.dockerignore`, `ui.py`, `graph.py`, `install_r_packages.R`, etc.).
- [Railway](https://railway.app) account (sign in with GitHub).

### 2. Create a new project on Railway

1. Go to [railway.app](https://railway.app) → **Start a New Project**.
2. Choose **Deploy from GitHub repo**.
3. Select your **psych-mas** repository (and branch, e.g. `main`).
4. Railway will detect the **Dockerfile** and build the image (R + R packages + Python; first build can take several minutes).

### 3. Configure the service

1. In the service **Settings**:
   - **Root Directory:** leave default (repo root).
   - **Dockerfile Path:** `Dockerfile` (default).
   - **Watch Paths:** leave default so pushes to the repo trigger redeploys.
2. Under **Variables**, add your secrets (same as Streamlit Cloud):
   - `GOOGLE_API_KEY` = your Google API key  
   - `OPENROUTER_API_KEY` = your OpenRouter API key  

Railway sets `PORT` automatically; the Dockerfile runs Streamlit on that port.

### 4. Deploy and share

1. Click **Deploy** (or push to GitHub to trigger a deploy).
2. Once built, open **Settings** → **Networking** → **Generate Domain** to get a public URL.
3. Share that URL; users get the full app including IRT, ICC, and Wright Map (no R install on their side).

### 5. Local Docker test (optional)

```bash
docker build -t psych-mas .
docker run -p 8501:8501 -e PORT=8501 psych-mas
```

Then open `http://localhost:8501`.

### 6. Expose the LangGraph API on Railway (optional)

You do **not** run this on your local computer. You add a **second service** in the same Railway project so the LangGraph API runs in the cloud next to your Streamlit app.

1. In your Railway project (where the Streamlit app is already deployed), click **+ New** → **Empty Service** (or **Add service**).
2. For the new service, choose **Deploy from GitHub repo** and select the **same** psych-mas repo (and branch).
3. Railway will use the **same Dockerfile** and build the same image.
4. **Override the start command** so this service runs the LangGraph server instead of Streamlit:
   - Open the new service → **Settings** → **Deploy** (or **Build & Deploy**).
   - Find **Custom start command** / **Start Command** (or **Override** for the run command).
   - Set it to (use a shell so `PORT` is expanded):  
     `sh -c 'langgraph dev --port ${PORT:-2024}'`  
     Railway sets `PORT`; the server will listen on it. Locally, port 2024 is used if `PORT` is unset.
5. **Deploy** the service. After the build finishes, go to **Settings** → **Networking** → **Generate Domain**.
6. The new service’s URL (e.g. `https://your-langgraph-api.up.railway.app`) is your **LangGraph API** base URL. Use it to invoke `psych_workflow` (e.g. `POST /runs` or the path shown in [LangGraph API docs](https://langchain-ai.github.io/langgraph/concepts/langgraph_api/)).

Your Streamlit app keeps its own URL; the LangGraph API has this separate public URL. No activation on your local machine—both run on Railway.

---

## Publishing for users without R

If you are **creating and publishing** this app for users who do **not** have R installed:

### Option 1: Streamlit Community Cloud (simplest)

1. **Deploy** the app on [Streamlit Community Cloud](https://share.streamlit.io) (see **Deploy on Streamlit Community Cloud** above).
2. **Share the app URL** with your users. They open it in a browser; no install, no R.
3. **Set secrets** (Settings → Secrets) so LLM features work: `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`.

**What users get without R:**

- Upload response and response-time CSVs  
- Prompt analysis (natural-language model settings)  
- Psych-MAS summaries (descriptive, model fit, item fit, Wright Map, ICC, person parameters, RT) — requires API key in Model engine  
- Response-time analysis (latency flags, RT histograms)  
- APA report (PDF) — without IRT tables/figures when R is not available  
- Advanced Analysis chat  

**What is skipped without R:** IRT fitting, ICC plots, Wright Map, model fit (M2), item/person parameters. The app shows a short notice at the top and skips those steps with a clear message.

### Option 2: Railway + Docker (full IRT, users still don’t install R)

To give users **full IRT** (ICC, Wright Map, mirt) without each user installing R, deploy with **Railway** using the included **Dockerfile** (see **Deploy on Railway** above). The image installs R, mirt/WrightMap/psych, and Streamlit; users only need the app URL. You can also use the same Dockerfile on Render, Fly.io, or any host that runs Docker.

---

## Project layout

| Path              | Purpose |
|-------------------|--------|
| `ui.py`           | Streamlit UI, LLM calls, APA PDF builder |
| `graph.py`        | LangGraph workflow (IRT, RT, prompt analysis) |
| `Dockerfile`      | Docker image for Railway/Render (R + R packages + Streamlit) |
| `.dockerignore`   | Excludes unneeded files from Docker build context |
| `langgraph.json`  | LangGraph CLI config (`psych_workflow` → `graph.py:app`) |
| `openrouter_models.py` | OpenRouter free model list for Model engine |
| `pyproject.toml`  | Dependencies (langgraph, streamlit, rpy2, reportlab, etc.) |
| `requirements.txt` | Pip-installable deps for Cloud / non-uv use |

---

## Function summary

### Backend (`backend_service.py`)

- **`GET /health`**: Simple “is the service up?” check used by the Streamlit UI.
- **`POST /irt`**: Runs `graph.irt_agent(...)` in the backend (Linux/Docker recommended) and returns:
  - `result.item_params` (ψ item parameters)
  - optionally `result.person_params`, plots, and other IRT artifacts.
- **`POST /detect`**: Starts a detect job and returns `run_id`.
  - Requires `psi_data` in the request (ψ is generated on the Preparation page; backend does not re-fit IRT for Detect).
  - Runs aberrance agents **sequentially** (avoids concurrent rpy2/R calls).
- **`GET /detect/{run_id}/status`**: Returns `{status, progress, irt_status, node_states, error}`.
- **`GET /detect/{run_id}/result`**: Returns the final state including `flags` and `final_report`.

### Workflow + agents (`graph.py`)

- **`irt_agent(state)`**: Fits the IRT model (via R `mirt`) and produces ψ item parameters (`item_params`) and (optionally) person parameters.
- **`rt_agent(state)`**: Response-time analysis (RT histograms / latency flags).
- **Aberrance agents**:
  - `pm_agent` (parametric misfit), `nm_agent` (nonparametric/Guttman),
  - `ac_agent` (answer copying), `as_agent` (answer similarity),
  - `rg_agent` (rapid guessing), `cp_agent` (change point),
  - `tt_agent` (tampering stub), `pk_agent` (preknowledge).
- **`manager_synthesizer(state)`**: Produces the “Forensic Verdict” narrative using the selected LLM provider/model.

### Streamlit UI (`ui.py`)

- **Preparation**: uploads data, runs backend IRT (`POST /irt`) to generate ψ, and starts Detect (`POST /detect`).
- **Detect Progress**: polls backend `/status` and `/result`, and renders the agent tree + progress UI.
- **Tools → Backend test**: debug utility to inspect session ψ/payload and call backend endpoints directly.

## License

Use and adapt as needed for your project.
