# PsyMAS Workbench (v0.7.7)

PsyMAS is a human-in-the-loop psychometric forensics workbench. It combines deterministic forensic indices, rule-based evidence governance, AI-assisted explanation, and human review. Statistical flags are review triggers; PsyMAS does not determine misconduct.

PsyMAS helps assessment teams organize and review unusual response behavior in a traceable workflow. It accepts response, response-time, item-parameter, exposure, and answer-change data; runs reproducible psychometric and forensic checks; maps eligible index flags into evidence domains; and prepares cautious case summaries for human review. The system keeps index values, rules, thresholds, source columns, domain profiles, reviewer decisions, and audit records connected so that a reviewer can see what was observed, why it was included, and what remains uncertain. It is intended for research, demonstrations, and review support rather than automatic misconduct classification.

## Install and Start

Choose one method.

### 1. Windows installer (recommended)

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Download `PsyMAS-Setup-Windows-v0.7.7.exe` from the [v0.7.7 release](https://github.com/JujiaLi2020/psych-mas/releases/tag/v0.7.7).
3. Run the installer. When prompted, choose OpenRouter, No AI, or Local Ollama.
4. Open `http://localhost:8501`.

Windows 10/11, Docker Desktop, 8 GB RAM, and 5-10 GB free disk space are recommended.

### 2. CLI installation

Requirements: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv tool install "psych-mas @ git+https://github.com/JujiaLi2020/psych-mas.git@v0.7.7"
psymas doctor
psymas start
```

Open `http://localhost:8501`. Useful commands:

```bash
psymas status
psymas logs --follow
psymas restart
psymas stop
```

### 3. Docker Compose

```bash
git clone https://github.com/JujiaLi2020/psych-mas.git
cd psych-mas
git switch --detach v0.7.7
```

Create `.env` from `.env.example`, then start the published image:

```bash
docker compose -f docker-compose.release.yml pull
docker compose -f docker-compose.release.yml up -d
```

Open `http://localhost:8501`. Stop with `docker compose -f docker-compose.release.yml down`.

## First Use

1. Open **Scenario** and select **Demo**, or choose an input scenario.
2. For a new run, open **Review Workspace > 01 Data**.
3. Upload `responses.csv`; it is the only required file.
4. Add optional files when available: `response_times.csv`, `item_params.csv`, `compromised_items.csv`, and `answer_changes.csv`.
5. Select the IRT model when parameters must be estimated, then click **Run Full Forensic Review**.
6. Review **02 Evidence**, **03 AI Review**, **04 Human Review**, and **05 Record**.

The Demo loads an evaluated snapshot without recomputing all indices. The snapshot was generated with PsyMAS `v0.7.6`, packaged unchanged in the `v0.7.7` distribution, and preserves its original run ID.

## Input Files

| File | Use |
| --- | --- |
| `responses.csv` | Wide binary response matrix; required |
| `response_times.csv` | Response-time matrix; enables timing evidence |
| `item_params.csv` | Item parameters; otherwise estimated with `mirt` |
| `compromised_items.csv` | Exposed or compromised item labels |
| `answer_changes.csv` | Initial/final responses for answer-change review |

Sample files and downloadable templates are in `data/sample/` and the Data page.

## LLM Configuration

LLM support is optional. The installer can install Ollama and download the default `llama3.1:8b` model when Local Ollama is selected. In **Configuration**, users can later change the provider, endpoint, and model. Choose OpenRouter and enter an API key, choose Local Ollama, or choose No LLM. For manual Ollama setup:

```bash
ollama pull llama3.1:8b
```

The LLM summarizes governed indices and selected raw-data summaries. It cannot compute indices, create or change flags, change thresholds, infer intent, determine misconduct, or recommend sanctions. On repeat installer runs, an already-installed Ollama model is detected and is not downloaded again.

## Outputs and Storage

Run data are stored locally in `data/output/psymas_run.sqlite`, including detector outputs, evidence records, domain profiles, review priorities, LLM explanations, human decisions, and audit traces. These files may contain sensitive assessment data and should not be committed or shared without authorization.

```text
Data -> Deterministic Evidence -> Evidence Governance -> AI Review -> Human Review -> Record
```

## Reproducibility

The capsule in `reproducibility/` contains the generation script, manifest, and snapshot checksum. The saved Demo run used supplied simulation item parameters and did not re-estimate them. A new run without `item_params.csv` can estimate parameters with `mirt`.

Docker image:

```text
ghcr.io/jujiali2020/psych-mas:0.7.7
sha256:321f10aaa409557b877b5677b5d396a37602afb7508daa59c4c47c6204b59d58
```

Zenodo archive: [10.5281/zenodo.22761090](https://doi.org/10.5281/zenodo.22761090).

## Project Structure

```text
ui.py                 Streamlit entry point
backend_service.py    FastAPI and R-backed analysis service
graph.py              Detector and IRT workflow entry point
psymas_graph/         Shared workflow state and graph services
psymas_ui/            UI, governance, storage, review, and export services
config/               Rulebook and threshold configuration
data/                 Samples, Demo snapshot, and runtime storage
reproducibility/      Generation script, manifest, and checksums
```

## Troubleshooting

```bash
docker compose ps
docker compose logs backend
docker compose logs ui
```

Check `http://localhost:9000/health`. If `psymas` is not found after CLI installation, run `uv tool update-shell` and restart the terminal. Do not share `.env` when it contains an API key.

## License

PsyMAS is released under the [MIT License](LICENSE).

## Concepts

**Psychometrics** uses statistical models to describe how examinees respond to test items. In PsyMAS, `mirt` can estimate item and person parameters when supplied item parameters are unavailable. These estimates support the analysis; they are not misconduct findings.

**Forensic indices** are statistical indicators of unusual response, timing, similarity, preknowledge, answer-change, or change-pattern behavior. A flag means that an output meets its package rule or approved threshold. It is a reason to review a case, not proof of intent or misconduct.

**Agents** are separate deterministic workers, such as rapid-guessing, person-fit, preknowledge, similarity, copying, tampering, and change-point workers. A workflow router runs only the workers supported by the available inputs, then writes their outputs to a shared evidence record. Agents compute or organize evidence; they do not independently decide the final case outcome.

**Evidence governance** determines which outputs may count toward a domain profile. Package-returned or approved-threshold flags may be eligible; uncalibrated or descriptive outputs remain visible for context and audit. The LLM can explain governed evidence and selected raw-data summaries, but cannot create flags, alter rules, or replace human adjudication.
