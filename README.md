# Psychometric Multi-Agent System (Psych MAS)

**Psych MAS** is a specialized psychometric analysis platform designed to bridge the gap between complex Item Response Theory (IRT) modeling and actionable research insights. By utilizing a multi-agent graph architecture, the system coordinates specialized nodes for IRT parameter estimation, response-time (RT) latency analysis, and LLM-driven interpretation.

## 🏮 Key Features

* **Intelligent Model Selection:** Uses an LLM-based "Interpreter" to map natural language descriptions (e.g., *"I suspect guessing behavior"*) to specific IRT models like **3PL** or **4PL**.
* **Dual-Data Stream Analysis:** Processes both **Response Data** (binary 0/1) and **Response-Time (RT) Data** simultaneously to identify patterns like rapid guessing.
* **R-Python Hybrid Backend:** Leverages the `mirt` and `WrightMap` R packages via `rpy2` for gold-standard psychometric accuracy.
* **AI Interpretation:** Features an "Analysis Chat" and a "Paper-Ready Summary" generator that uses Gemini models to describe Wright Maps and item parameters in formal academic language.

---

## 🏗️ Architecture

The project uses a **StateGraph** (via `langgraph`) to orchestrate the workflow:

1.  **Orchestrator Node:** Dispatches the data to specialized agents.
2.  **IRT Agent:** Interfaces with R's `mirt` package to calculate $\theta$ (person ability), item difficulty ($b$), and discrimination ($a$).
3.  **RT Agent:** Analyzes latency data to flag anomalous response patterns like rapid guessing.
4.  **Analyze Agent:** Synthesizes results from both streams into a final reporting state.

---

## 📊 Visualizations

Psych MAS prioritizes "single-page efficiency" for large datasets:
* **Wright Maps:** Align person ability distributions with item difficulty thresholds on the same latent scale.
* **Item Characteristic Curves (ICC):** Visualize the probability of a correct response across the $\theta$ spectrum.
* **RT Histograms:** Distributed plots for items showing correct/incorrect proportions relative to time taken.

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **R Environment:** Must be installed and available on your `PATH`.
* **R Packages:** `mirt`, `WrightMap`, `lattice`, `grDevices`.
* **API Key:** A `GOOGLE_API_KEY` in a `.env` file for LLM features.

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install streamlit pandas matplotlib rpy2 langgraph python-dotenv requests


### Run the application:

   ```Bash
   uv sync
   pip install streamlit
   streamlit run ui.py
   ```

## Data Requirements
The system expects two CSV files with matching dimensions:

Responses: Rows as persons, columns as items (Values: binary 0 or 1).

Response-Times: Rows as persons, columns as items (Values: numeric latency data).

## 🛠️ Tech Stack
Orchestration: LangGraph

Frontend: Streamlit

Psychometric Engine: R (mirt, WrightMap)

LLM Integration: Google Gemini (Generative Language API)

Data Processing: Pandas / NumPy


## INFO:langgraph_api.cli:

        Welcome to

╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

This in-memory server is designed for development and testing.
For production use, please use LangSmith Deployment.








