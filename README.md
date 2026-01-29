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

### 2. Set up API keys (optional, for LLM features)

Copy `.env.example` to `.env` (if present) or create a `.env` file in the project root:

- **Google Gemini:** `GOOGLE_API_KEY` (and `GEMINI_MODEL` if you want a non-default model).
- **OpenRouter:** `OPENROUTER_API_KEY` — get a key at [openrouter.ai](https://openrouter.ai). Some free models work without a key; for higher limits or paid models, set the key.

The app reads these via `python-dotenv`; do **not** commit `.env` to Git.

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
- **LangGraph Studio:** If you use the LangGraph Studio app, connect it to this server to inspect and run the graph visually.

### Config

- `langgraph.json` points to the graph: `./graph.py:app` (graph name `psych_workflow`).
- The Streamlit UI can run **without** the LangGraph server: it invokes the workflow directly via `graph.py`. Use `langgraph dev` when you want to call the graph via the API or Studio.

### Example: call the graph via API

Once `langgraph dev` is running, you can POST to the LangGraph API (e.g. invoke the `psych_workflow` graph) according to the [LangGraph API docs](https://langchain-ai.github.io/langgraph/concepts/langgraph_api/). The Streamlit app does not require the server to be up.

---

## Requirements

- **Python:** 3.13+ (see `pyproject.toml`; adjust if you use an older version).
- **Local (full IRT + Wright Map):** R on `PATH` with packages `mirt` and `WrightMap`; Python deps from `pyproject.toml` or `requirements.txt`.
- **Streamlit Cloud:** No R; IRT fitting, ICC, and Wright Map are skipped with an explanatory message. LLM and RT features work; set secrets (see below).

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

## Project layout

| Path              | Purpose |
|-------------------|--------|
| `ui.py`           | Streamlit UI, LLM calls, APA PDF builder |
| `graph.py`        | LangGraph workflow (IRT, RT, prompt analysis) |
| `langgraph.json`  | LangGraph CLI config (`psych_workflow` → `graph.py:app`) |
| `openrouter_models.py` | OpenRouter free model list for Model engine |
| `pyproject.toml`  | Dependencies (langgraph, streamlit, rpy2, reportlab, etc.) |
| `requirements.txt` | Pip-installable deps for Cloud / non-uv use |

---

## License

Use and adapt as needed for your project.
