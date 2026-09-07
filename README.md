# PsyMAS Workbench (v0.7.3)

PsyMAS is a human-in-the-loop psychometric forensics workbench. It combines deterministic aberrance-detection routines, rulebook-based evidence governance, AI-assisted case explanation, and human review records.

The system treats statistical flags as review triggers. It does not determine misconduct.

## What Is Included

This is a clean runnable distribution. It keeps only the files needed to run the app and the bundled demo:

- Streamlit entry point and page composition: `ui.py`
- FastAPI backend: `backend_service.py`
- deterministic node implementations: `graph.py`
- shared graph contracts and topology: `psymas_graph/`
- UI support modules: `psymas_ui/`
- rulebook and thresholds: `config/`
- sample/demo data: `data/`
- Docker and dependency files

Manuscripts, screenshots, literature PDFs, old backups, tests, and temporary research exports have been removed from this clean copy.

## Workflow

The interface is organized as:

1. Scenario
2. Review Workspace
   - 01 Data
   - 02 Evidence
   - 03 AI Review
   - 04 Human Review
   - 05 Record
3. Research Tools
4. Configuration

The Demo scenario loads bundled simulated data and the evaluated run snapshot so users can inspect results without recomputing all indices.

## Data Kept In This Copy

| Path | Purpose |
| --- | --- |
| `data/sample/` | Minimal CSV examples for user input templates |
| `data/upload/` | Bundled demo input files |
| `data/psymas_demo_evaluated_snapshot.zip` | Demo snapshot loaded by the Demo scenario |

The application creates `data/output/psymas_run.sqlite` and export files locally at runtime. These generated records may contain review decisions and are excluded from version control. The bundled evaluated snapshot contains the reproducible demonstration state used by the Demo scenario.

## Configuration Files

| Path | Purpose |
| --- | --- |
| `config/rulebook_index.csv` | Software-readable index registry and evidence-use mapping |
| `config/b3_index_mapping.yaml` | Domain-evidence mapping support |
| `config/default_thresholds.yaml` | Default domain and threshold settings |
| `config/index_thresholds.yaml` | Index-level threshold configuration |
| `.env.example` | Local environment template |

## Quickstart With Docker

From the project root:

```bash
copy .env.example .env
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The backend is available at:

```text
http://localhost:9000
```

The first Docker build installs R and R packages, so it can take several minutes.

## Quickstart Without Docker UI

Run the backend in Docker:

```bash
docker compose up --build backend
```

Create `.env` from the template and set:

```env
PSYMAS_BACKEND_URL=http://localhost:9000
```

Install Python dependencies and start Streamlit:

```bash
uv sync
uv run -- python -m streamlit run ui.py --server.port 8501
```

## LLM Configuration

PsyMAS supports:

- OpenRouter hosted models
- Local Ollama models

For OpenRouter, add the key to `.env`:

```env
OPENROUTER_API_KEY=your_key_here
```

For local Ollama, start Ollama separately and make sure the selected local model is available, for example:

```bash
ollama pull llama3.1:8b
```

The LLM is used only to summarize governed evidence and draft cautious reviewer-facing language. It cannot compute indices, change thresholds, create flags, infer intent, determine misconduct, or recommend sanctions.

## Backend API

The backend provides:

- `GET /health`
- `POST /irt/start`
- `GET /irt/{job_id}/status`
- `GET /irt/{job_id}/result`
- `POST /irt`
- `POST /detect`
- `GET /detect/{run_id}/status`
- `GET /detect/{run_id}/result`

The backend runs R-based psychometric routines through `rpy2`; Docker is recommended on Windows.

## Code Architecture

The runnable entry points remain stable while implementation details are split into focused packages:

```text
PsyMAS
├── ui.py                    Streamlit entry point and page composition
├── backend_service.py       FastAPI jobs and R-backed analysis endpoints
├── graph.py                 Deterministic and IRT node implementations
├── psymas_graph/            Shared graph state and workflow topology
├── psymas_ui/               UI services, evidence logic, exports, and storage
├── mmls.py                  Curated hosted/local LLM model metadata
├── config/                  Rulebook, domain rules, and thresholds
├── data/                    Samples, demo inputs, snapshots, and run database
└── scripts/                 Deployment entry points
```

`ui.py` and `graph.py` are compatibility entry points. Streamlit still starts with `ui.py`, and `langgraph.json` still imports graph objects from `graph.py`. New reusable logic should be placed in `psymas_ui/` or `psymas_graph/` instead of expanding either entry point.

### Root Python Files

| Path | Responsibility |
| --- | --- |
| `ui.py` | Initializes Streamlit, manages session-level page routing, and composes the Data, Evidence, AI Review, Human Review, Record, Research Tools, and Configuration views. It imports reusable behavior from `psymas_ui/`. |
| `backend_service.py` | FastAPI service for health checks, asynchronous IRT jobs, and deterministic detection jobs. It coordinates Python/R execution for the UI. |
| `graph.py` | Implements IRT, detector, manager, synthesizer, and reporter nodes. It exports `psych_workflow`, `forensic_workflow`, and `app` for existing callers. Shared state, topology, thresholds, serialization, and RT plotting are imported from `psymas_graph/`. |
| `mmls.py` | Defines the curated OpenRouter and local Ollama model catalog, display labels, pricing notes, and recommended uses. |
| `main.py` | Minimal command-line placeholder retained for package/tool compatibility; it is not the Streamlit entry point. |

### `psymas_graph/`

| Path | Responsibility |
| --- | --- |
| `psymas_graph/state.py` | Defines the typed shared LangGraph state and the reducer used to merge parallel specialist outputs. |
| `psymas_graph/workflows.py` | Owns graph topology. It builds the parallel forensic workflow and the IRT/response-time workflow from injected node functions. |
| `psymas_graph/thresholds.py` | Reads YAML-backed detector settings, normalizes alpha/threshold values, controls enabled indices, and orients pairwise flag matrices. |
| `psymas_graph/serialization.py` | Converts pandas, NumPy, mapping, and sequence results into JSON-ready record lists. |
| `psymas_graph/rt_visuals.py` | Generates the item-level response-time histogram artifact used by the psychometric workflow. |
| `psymas_graph/llm_client.py` | Provides the small hosted-model HTTP clients used by graph prompt analysis and report-synthesis nodes. |
| `psymas_graph/__init__.py` | Exposes the shared graph state as the package-level contract. |

### `psymas_ui/`

| Path | Responsibility |
| --- | --- |
| `psymas_ui/app_config.py` | Central application version, domain order/labels, detector-to-agent mapping, visualization palette, and demo paths. |
| `psymas_ui/backend_client.py` | Normalizes backend URLs, performs backend HTTP requests, and summarizes backend/Detect status for the UI. |
| `psymas_ui/input_data.py` | Cleans uploaded CSV tables, validates binary responses and answer-change records, aligns response-time columns, and parses compromised-item files. |
| `psymas_ui/llm.py` | OpenRouter and Ollama clients, provider/model selection, model discovery, connection tests, and selected-model dispatch. |
| `psymas_ui/components.py` | Shared Streamlit styling and reusable workspace/KPI components. |
| `psymas_ui/evidence_governance.py` | Family-level evidence aggregation helpers, including correction-variant handling and B3 eligibility. |
| `psymas_ui/evidence_tree.py` | Builds individual/cohort evidence-lineage visualizations and shared strength/priority styling. |
| `psymas_ui/review.py` | Builds the final human-review queue from governed evidence and detector flags. |
| `psymas_ui/run_store.py` | SQLite schema and persistence API for run inputs, indices, governed evidence, LLM outputs, review decisions, and audit records. |
| `psymas_ui/run_snapshot.py` | Packs and restores portable evaluated-run snapshots used by the Demo workflow. |
| `psymas_ui/exports.py` | Builds the consolidated master-results table. |
| `psymas_ui/research_export.py` | Packages research-facing datasets, audit outputs, validation material, and expert-review templates. |
| `psymas_ui/worked_example.py` | Selects worked-example cases and generates paper/tutorial tables and figures. |
| `psymas_ui/__init__.py` | Marks the UI support package. |

### Runtime and Configuration Files

| Path | Responsibility |
| --- | --- |
| `config/rulebook_index.csv` | Index registry: domain, family, role, evidence use, and provenance rules. |
| `config/b3_index_mapping.yaml` | Domain-evidence mapping and aggregation support. |
| `config/default_thresholds.yaml` | Default domain-strength and review-priority rules. |
| `config/index_thresholds.yaml` | Index-level activation and threshold settings. |
| `langgraph.json` | Registers `psych_workflow` and `forensic_workflow` for LangGraph tooling. |
| `install_r_packages.R` | Installs the R packages required by the backend. |
| `Dockerfile` | Builds the Python/R runtime image. |
| `docker-compose.yml` | Starts the Streamlit UI and FastAPI backend locally. |
| `scripts/run_ui_railway.sh` | Streamlit container entry point for Railway-style deployment. |
| `pyproject.toml`, `uv.lock` | Python project metadata and reproducible dependency lock. |
| `requirements.txt`, `packages.txt` | Compatibility dependency lists for deployment platforms. |
| `r_packages.txt`, `r_packages.lock` | R package requirements and recorded versions. |

## Dependency Direction

To keep the codebase maintainable, dependencies should flow in one direction:

```text
ui.py → psymas_ui/* → config and data
backend_service.py → graph.py → psymas_graph/*
graph.py → R/Python statistical libraries
```

Modules in `psymas_graph/` must not import `ui.py`. Modules in `psymas_ui/` should not import the Streamlit entry point. `graph.py` must remain usable without starting Streamlit.

## Security

Do not share `.env` if it contains API keys. Use `.env.example` as the public template.
