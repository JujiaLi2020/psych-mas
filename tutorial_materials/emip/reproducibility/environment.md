# Environment record

## Current source distribution

- PsyMAS: `0.7.7`
- Python requirement: `>=3.11`
- Python lockfile: `uv.lock`
- R package lock: `r_packages.lock`
- Streamlit: `1.58.0`
- FastAPI: `0.137.0`
- LangGraph: `1.2.5`
- LangGraph CLI: `0.4.29`
- `mirt`: `1.46.1`
- `aberrance`: `0.3.0`
- `rpy2`: `3.5.17` on non-Windows hosts
- Docker Compose release image tag: `0.7.7`

These are current distribution metadata, not proof that the v0.7.6 snapshot
was generated with exactly the same dependency versions. The snapshot manifest
is authoritative for the saved-run version boundary.

## LLM boundary

The saved Demo does not contain a persisted LLM explanation for Examinee 332.
No API key, model response, temperature, or external LLM call was used during
this collection. The application supports OpenRouter and local Ollama, but the
tutorial packet treats AI output as a live optional reporting layer rather than
as part of the archived detector computation.
