# Development log

## Week of 2026-04-28 — 2026-05-30 (v0.6.0)

### Function

- **Railway deployment** — Docker Compose stack (backend + UI), `run_ui_railway.sh`, HTTPS-aware `PSYMAS_BACKEND_URL` for public Railway domains.
- **Async IRT** — `POST /irt/start` with status/result polling; Preparation uses async path by default.
- **Test-Taker Report PDF** — Compact branded PDF per examinee from Aberrance Summary.
- **Person-level export** — CSV/ZIP with column legend and per-agent index guide.
- **Pre-estimated ψ upload** — Skip IRT when item parameters are supplied via CSV.
- **Forensic navigation** — Reliable jump to Aberrance Summary after Detect completes.

### Technology

- **Job storage tiers** — Redis → `PSYMAS_JOB_DIR` disk → in-memory; subprocess workers for detect and IRT.
- **Gunicorn + optional Redis** — Multi-worker backend documented in Dockerfile and Compose redis profile.
- **mmls.py** — OpenRouter model list moved from `openrouter_models.py`.
- **Version:** README, `pyproject.toml`, and UI set to **v0.6.0**.

---

## Week of 2026-03-03 — 2026-03-06

### Function

- **Delete online scenario** — Removed Scenario C (Online/Unproctored); presets are A (Low-Stakes), B (High-Stakes), D (Custom).
- **Detailed downloadable person-level report** — Designed and implemented a person-level report (e.g. flags, indices) available for download from the UI.
- **Pre-estimated parameter data input** — Added support for pre-estimated (IRT) item/person parameters as input so users can skip IRT estimation and supply ψ directly.
- **Detailed agents report with LLM** — Agent-by-agent reports plus LLM-generated Forensic Verdict summarizing specialist outputs and high-risk examinees.
- **Compromised items default** — When no preknowledge items are entered, default to items 1..n-1 as compromised (R requires ≥1 secure item). UI and backend aligned.

---

### Technology

- **Backend IRT:** Fixed rpy2 conversion in `/irt` (activate numpy2ri/pandas2ri in request thread; run IRT R code inside `_converter.context()`).
- **Docker:** Documented correct run — override CMD with `uvicorn backend_service:app --host 0.0.0.0 --port 8000` so LangGraph backend is reachable.
- **Streamlit plots:** Set `matplotlib.use("Agg")` and reinstalled matplotlib/networkx so collusion graph and other figures render.
- **Preknowledge default:** Backend defaults to 1..n-1 when compromised list is empty; UI text updated.
- **Version:** README and release set to **v0.5.0**; changes pushed to GitHub.
