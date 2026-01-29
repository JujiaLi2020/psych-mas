# Psych MAS

Psychometric analysis app: load response and response-time data, describe your IRT model in natural language, confirm settings, then run the workflow. View item/person parameters, Wright Map, item fit, ICC plots, and RT histograms. Includes LLM-assisted prompt interpretation and analysis chat.

## Run locally

```bash
# Install dependencies (e.g. with uv or pip)
uv sync
# or: pip install -r requirements.txt

# Run the Streamlit UI
streamlit run ui.py
```

Optional: copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (and `GEMINI_MODEL` if needed) for LLM features.

## Publish online with Streamlit + GitHub (no Docker)

You can run the app on **Streamlit Community Cloud** and share the link with your team.

### 1. Push the project to GitHub

- Create a repository on GitHub and push this project (or use an existing repo).
- Ensure these files are in the repo root: `ui.py`, `graph.py`, `main.py`, `langgraph.json`, `requirements.txt`, and the `.streamlit` folder.

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your **repository**, **branch**, and set **Main file path** to `ui.py`.
4. (Optional) Under **Advanced settings**, set **Python version** (e.g. 3.11 or 3.12 if available).
5. Click **Deploy**. The app will build and get a URL like `https://<your-app>.streamlit.app`.

### 3. Add secrets (for LLM features)

For prompt analysis and analysis chat to work in the cloud:

1. In the Streamlit Cloud app page, open **Settings** (⚙️) → **Secrets**.
2. Add your secrets, for example:

```toml
GOOGLE_API_KEY = "your-google-api-key"
GEMINI_MODEL = "models/gemini-1.5-flash-latest"
```

Save; the app will restart and use these when calling the Gemini API.

### 4. Share the link

Send your team the app URL (e.g. `https://<your-app>.streamlit.app`). They can open it in a browser, upload their own response and response-time CSVs, and use the workflow. No need to install anything.

### IRT and Wright Map on the cloud

- **Streamlit Community Cloud does not include R.** The app is built to run without R: if R is missing, it shows a clear message and skips IRT fitting, ICC plots, and Wright Map (R is used for `mirt` and `WrightMap`).
- **For full IRT + Wright Map**, team members can run the app **locally** (with R and the required R packages installed). The same GitHub repo runs both locally (full features) and on Streamlit Cloud (LLM, file upload, and RT analysis; IRT/Wright Map disabled with an explanation).

## Requirements

- **Local (full features):** Python 3.10+, R with packages `mirt` and `WrightMap`, and the Python dependencies in `requirements.txt`.
- **Streamlit Cloud:** Only the contents of `requirements.txt`; R is not available.

## LangGraph API (optional)

For local development with the LangGraph API:

```bash
langgraph dev
# API: http://127.0.0.1:2024
```

The Streamlit UI can run without the LangGraph server; the workflow is invoked directly from `graph.py`.