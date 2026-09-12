# PsyMAS Workbench (v0.7.6)

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

## Input Files

Only `responses.csv` is required to start. Optional files enable additional evidence:

| File | Requirement |
| --- | --- |
| `responses.csv` | Required response matrix |
| `response_times.csv` | Optional response-time matrix |
| `item_params.csv` | Optional; otherwise estimated from responses |
| `compromised_items.csv` | Optional exposure labels |
| `answer_changes.csv` | Optional initial/final response records |

Example files are available in `data/sample/`.

## Configuration Files

| Path | Purpose |
| --- | --- |
| `config/rulebook_index.csv` | Software-readable index registry and evidence-use mapping |
| `config/b3_index_mapping.yaml` | Domain-evidence mapping support |
| `config/default_thresholds.yaml` | Default domain and threshold settings |
| `config/index_thresholds.yaml` | Index-level threshold configuration |
| `.env.example` | Local environment template |

## Install the `psymas` CLI

The CLI requires Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). It installs
PsyMAS in an isolated environment and adds the `psymas` command to your PATH.

### Install directly from GitHub

Install the current version from the `main` branch without cloning the
repository:

```bash
uv tool install "psych-mas @ git+https://github.com/JujiaLi2020/psych-mas.git@main"
```

For a reproducible installation, install the CLI-enabled v0.7.6 release:

```bash
uv tool install "psych-mas @ git+https://github.com/JujiaLi2020/psych-mas.git@v0.7.6"
```

Verify the installation and launch the complete UI + analysis backend stack:

```bash
psymas --version
psymas doctor
psymas start
```

`psymas start` checks Docker, downloads the versioned PsyMAS image, starts the
UI and R-backed analysis backend, waits for the application to become healthy,
and opens it at `http://localhost:8501`. No repository clone or separate R
installation is required. The lifecycle commands below use the same persistent
data directory:

```bash
psymas status
psymas logs --follow
psymas restart
psymas stop
```

If your shell cannot find `psymas`, run `uv tool update-shell`, restart the
terminal, and try again.

### Install from a local checkout

Developers and users who have cloned or downloaded the repository can run this
from the repository root (the directory containing `pyproject.toml`):

```bash
uv tool install .
```

`pipx install .` is equivalent. During development, keep the installed command
linked to source edits:

```bash
uv tool install --editable .
```

### Upgrade or remove the CLI

Upgrade an installation that tracks `main`:

```bash
uv tool upgrade psych-mas
```

To reinstall from GitHub when replacing an existing installation:

```bash
uv tool install --force "psych-mas @ git+https://github.com/JujiaLi2020/psych-mas.git@main"
```

Remove the CLI with:

```bash
uv tool uninstall psych-mas
```

### Advanced: run UI and backend separately

Most users should use `psymas start`. For development, the UI can instead
connect to an independently running backend:

```bash
psymas ui --backend-url http://localhost:9000
```

On Linux or macOS with R and the required R packages installed, the backend can
also be launched from the CLI:

```bash
psymas backend --port 9000
```

The native R-backed backend is not installed on Windows; use the unified
Docker-based `psymas start` command or the Windows Installer there.

Use `psymas --help` for all available commands and options.

## Windows Installation (Recommended)

### 1. Check the requirements

- Windows 10 or 11, 8 GB RAM, and 5-10 GB free disk space recommended
- Internet access for the installer and container image
- Available host ports `8501` and `9000`

The installer uses Docker Desktop to provide the Python and R environment. Docker may require WSL 2, hardware virtualization, acceptance of Docker's terms, or a restart on first installation.

### 2. Download the installer

Download `PsyMAS-Setup-Windows-v0.7.6.exe` from [GitHub Releases](https://github.com/JujiaLi2020/psych-mas/releases/tag/v0.7.6) and run it. Windows may ask you to confirm software downloaded from the internet.

Optional integrity check in PowerShell:

```powershell
Get-FileHash .\PsyMAS-Setup-Windows-v0.7.6.exe -Algorithm SHA256
```

Compare the result with the `.sha256` file attached to the same GitHub Release.

### 3. Complete the guided setup

1. Allow the installer to install or start Docker Desktop when needed.
2. Choose OpenRouter (recommended), No AI, or Local Ollama (advanced).
3. Wait while the versioned PsyMAS image is downloaded and started.
4. PsyMAS opens automatically at `http://localhost:8501`.

The installer adds Start, Stop, and Configure PsyMAS AI shortcuts. Assessment and review records are stored under `%LOCALAPPDATA%\PsyMAS` and remain available across container updates.

## Manual Docker Installation

### 1. Install Docker Desktop

Install Docker Desktop with Docker Compose v2, start it, and wait until the Docker engine is running.

### 2. Download PsyMAS v0.7.6

```bash
git clone https://github.com/JujiaLi2020/psych-mas.git
cd psych-mas
git switch --detach v0.7.6
```

### 3. Create the environment file

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

API keys are optional unless AI-assisted reporting is used.

### 4. Pull and start the prebuilt image

```bash
docker compose -f docker-compose.release.yml pull
docker compose -f docker-compose.release.yml up -d
```

This downloads the published Python/R environment instead of compiling R packages locally.

### 5. Open and verify PsyMAS

1. Open the interface at `http://localhost:8501`.
2. Confirm backend health at `http://localhost:9000/health`.
3. Select **Demo** to load the bundled data and evaluated snapshot without recomputing all indices.

### 6. Stop or restart PsyMAS

Stop the background services with:

```bash
docker compose -f docker-compose.release.yml down
```

Restart without rebuilding:

```bash
docker compose -f docker-compose.release.yml up -d
```

## Advanced: Run the UI Outside Docker

### 1. Start only the backend

```bash
docker compose up --build backend
```

### 2. Configure the host UI

Create `.env` and set:

```env
PSYMAS_BACKEND_URL=http://localhost:9000
```

### 3. Install and start the UI

Python 3.11 and `uv` are required:

```bash
uv sync
uv run -- python -m streamlit run ui.py --server.port 8501
```

## LLM Configuration

PsyMAS supports OpenRouter hosted models and local Ollama models. Choose one option:

### 1. OpenRouter

Open **Configuration**, choose **OpenRouter**, enter the API key, and select **Save key**. Choose a curated model or enter any valid OpenRouter model ID, then use **Test selected model**. The key is stored locally in the persistent PsyMAS data directory and is not included in exports.

For unattended deployments, the key can instead be added to `.env`:

```env
OPENROUTER_API_KEY=your_key_here
```

### 2. Local Ollama

Install Ollama, download a supported model, and start the service:

```bash
ollama pull llama3.1:8b
ollama serve
```

In **Configuration**, set the Ollama chat endpoint and select **Discover installed
models**, or enter an exact model name manually. PsyMAS stores separate model
choices for OpenRouter and Ollama, so switching providers does not discard the
other provider's selection. Select **Save provider & model settings** to retain
the choices across restarts.

When both PsyMAS services run in Docker on Windows or macOS, add this to `.env`:

```env
OLLAMA_CHAT_URL=http://host.docker.internal:11434/api/chat
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

## Persistent Run Records

Docker Compose mounts `data/output/` into the UI container. The SQLite run database, human-review decisions, and generated exports therefore remain available after containers are recreated. Do not commit this directory because it may contain assessment and review data.

## Troubleshooting

```bash
docker compose ps
docker compose logs backend
docker compose logs ui
```

Confirm backend health at `http://localhost:9000/health`. If a port is already occupied, change the host-side port in `docker-compose.yml`. To rebuild after dependency changes, run `docker compose up --build`.

## Code Architecture

The runnable entry points remain stable while implementation details are split into focused packages:

```text
PsyMAS
├── ui.py                    Streamlit entry point and page composition
├── backend_service.py       FastAPI jobs and R-backed analysis endpoints
├── psymas_cli.py            Installed `psymas` command
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
| `psymas_cli.py` | Implements the installed `psymas ui`, `psymas backend`, and `psymas doctor` commands. |
| `graph.py` | Implements IRT, detector, manager, synthesizer, and reporter nodes. It exports `psych_workflow`, `forensic_workflow`, and `app` for existing callers. Shared state, topology, thresholds, serialization, and RT plotting are imported from `psymas_graph/`. |
| `mmls.py` | Defines the curated OpenRouter and local Ollama model catalog, display labels, pricing notes, and recommended uses. |
| `main.py` | Legacy placeholder retained for compatibility; the installed CLI entry point is `psymas_cli.py`. |

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

## License

PsyMAS is released under the MIT License. See `LICENSE`.
