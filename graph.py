from pathlib import Path
from typing import NotRequired, TypedDict
import json
import os
import re
import tempfile
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "matplotlib is required for RT histograms. Install with: pip install matplotlib"
    ) from e
from langgraph.graph import StateGraph, END

# To run an R package from a node (rt_agent, analyze_agent, etc.):
#   import rpy2.robjects as ro
#   from rpy2.robjects import pandas2ri
#   with (ro.default_converter + pandas2ri.converter).context():
#       ro.r("library(mirt)")   # or your package
#       ro.globalenv["df"] = my_pandas_df
#       ro.r("result <- some_r_func(df)")
# rpy2 is in pyproject.toml; ensure R is on PATH.

_GRAPH_DIR = Path(__file__).resolve().parent


def analyze_prompt(prompt: str) -> dict:
    normalized = (prompt or "").strip()
    if not normalized:
        return {
            "itemtype": "2PL",
            "r_code": "model <- mirt(df, 1, itemtype='2PL')",
            "suggestion": "Defaulting to 2PL.",
            "reason": "No model preference provided.",
            "feedback": "Please describe your goal (e.g., guessing behavior, Rasch/1PL, or asymmetric response patterns).",
            "source": "heuristic",
        }

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    if not api_key:
        return _heuristic_prompt_mapping(prompt)

    system = (
        "You are a psychometrics assistant for an IRT analysis app. If the prompt "
        "is about IRT models (1PL, 2PL, 3PL, 4PL), map it to an itemtype and provide "
        "structured feedback with suggestion and reason. If the prompt is NOT related "
        "to psychometrics/IRT, respond in TWO PARTS: (1) FIRST, answer their question "
        "naturally and conversationally like a normal helpful assistant, (2) THEN add a "
        "brief note that this app is designed for psychometric analysis and guide them back. "
        "Always return JSON."
    )
    user = (
        f"Prompt: {normalized}\n"
        "Return JSON with keys: itemtype, suggestion, reason, feedback, note, r_code.\n"
        "If prompt is off-topic: set itemtype='2PL', set suggestion='' and reason=''. "
        "In 'feedback', write a TWO-PART response: (1) First, answer their question naturally "
        "like a normal LLM would, (2) Then add 'However, this app is designed for psychometric "
        "analysis using Item Response Theory (IRT). I can help you analyze test data with models "
        "like 1PL, 2PL, 3PL, or 4PL. Please describe your psychometric analysis needs.' "
        "Use r_code format: model <- mirt(df, 1, itemtype='2PL')."
    )
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}
        ]
    }

    # Try different model name variations if the first one fails
    model_variants = [
        model,  # Try the configured/default model first
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash",
        "models/gemini-pro",
    ]
    
    resp = None
    last_error = None
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            resp.raise_for_status()
            # If successful, update model for this call and break
            model = model_variant
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                last_error = e
                resp = None
                continue  # Try next variant
            else:
                raise  # Re-raise non-404 errors
        except Exception as e:
            last_error = e
            resp = None
            continue
    
    # If all variants failed, fall back to heuristic
    if resp is None or resp.status_code != 200:
        print(f"All model variants failed. Last error: {last_error}")
        return _heuristic_prompt_mapping(prompt)
    
    try:
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            return _heuristic_prompt_mapping(prompt)
        
        # Try to extract JSON from text (might be wrapped in markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find JSON object in text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
        
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, check if text looks like conversational feedback
            # If it mentions psychometrics, use it as feedback
            if "psychometric" in text.lower() or "irt" in text.lower() or "app" in text.lower():
                return {
                    "itemtype": "2PL",
                    "r_code": "model <- mirt(df, 1, itemtype='2PL')",
                    "suggestion": "",
                    "reason": "",
                    "feedback": text.strip(),
                    "note": "",
                    "source": "llm",
                }
            return _heuristic_prompt_mapping(prompt)
        
        itemtype = str(parsed.get("itemtype", "2PL")).upper()
        if itemtype not in {"1PL", "2PL", "3PL", "4PL"}:
            itemtype = "2PL"
        suggestion = parsed.get("suggestion", "")
        reason = parsed.get("reason", "")
        feedback = parsed.get("feedback", "Here is a suggested model based on your prompt.")
        # If feedback mentions psychometrics app but suggestion/reason are empty, prioritize feedback
        if not suggestion and not reason and feedback:
            suggestion = ""
            reason = ""
        return {
            "itemtype": itemtype,
            "r_code": f"model <- mirt(df, 1, itemtype='{itemtype}')",
            "suggestion": suggestion,
            "reason": reason,
            "feedback": feedback,
            "note": parsed.get("note", ""),
            "source": "llm",
        }
    except Exception as e:
        # Log the error for debugging but still fall back to heuristic
        print(f"LLM call failed: {type(e).__name__}: {e}")
        return _heuristic_prompt_mapping(prompt)


def _heuristic_prompt_mapping(prompt: str) -> dict:
    normalized = (prompt or "").lower()
    itemtype = "2PL"
    suggestion = "Defaulting to 2PL."
    reason = "No model preference detected."
    note = ""
    feedback = "I will pick a baseline model unless you specify constraints or behaviors."
    is_greeting = any(
        token in normalized
        for token in ["hello", "hi", "hey", "what can you do", "help", "start", "how to start"]
    )
    if is_greeting:
        feedback = (
            "Hi! Tell me what you want to analyze. For example: "
            "'I suspect guessing, use 3PL', or 'Use Rasch/1PL'."
        )
        note = "You can mention 1PL/2PL/3PL/4PL, guessing, or asymmetry."
    if "3pl" in normalized or "guess" in normalized:
        itemtype = "3PL"
        suggestion = "Guessing indicated; recommend 3PL."
        reason = "Prompt mentions guessing; 3PL includes a guessing parameter."
        feedback = "Noted guessing behavior; a guessing parameter can capture that."
    elif "4pl" in normalized:
        itemtype = "4PL"
        suggestion = "Asymmetry indicated; recommend 4PL."
        reason = "Prompt implies asymmetric lower/upper bounds."
        feedback = "You mentioned asymmetry; a 4PL can model lower/upper asymptotes."
    elif "1pl" in normalized or "rasch" in normalized:
        itemtype = "1PL"
        suggestion = "Rasch/1PL requested; recommend 1PL."
        reason = "Prompt requests Rasch/1PL."
        feedback = "Rasch/1PL noted; using a single discrimination parameter."
    elif normalized and not is_greeting and not any(
        token in normalized for token in ["1pl", "2pl", "3pl", "4pl", "rasch", "guess"]
    ):
        # Check for common off-topic patterns
        off_topic_keywords = ["weather", "time", "date", "news", "sport", "movie", "music", "food", "recipe", "translate", "calculate", "math", "python", "code", "programming"]
        is_off_topic = any(keyword in normalized for keyword in off_topic_keywords)
        
        if is_off_topic:
            suggestion = ""
            reason = ""
            feedback = (
                "I understand your question, but this app is specifically designed for psychometric analysis "
                "using Item Response Theory (IRT). I can help you analyze test data with models like 1PL, 2PL, 3PL, or 4PL. "
                "Please describe your psychometric analysis needs, such as: 'I suspect guessing behavior, use 3PL' "
                "or 'Use Rasch/1PL model'."
            )
            note = ""
        else:
            # For vague prompts that might be off-topic or just unclear
            suggestion = ""
            reason = ""
            feedback = (
                "I'm not sure how to help with that. This app is designed for psychometric analysis using "
                "Item Response Theory (IRT). I can help you analyze test data with models like 1PL, 2PL, 3PL, or 4PL. "
                "Please describe your psychometric analysis needs, such as: 'I suspect guessing behavior, use 3PL' "
                "or 'Use Rasch/1PL model'."
            )
            note = ""
    return {
        "itemtype": itemtype,
        "r_code": f"model <- mirt(df, 1, itemtype='{itemtype}')",
        "suggestion": suggestion,
        "reason": reason,
        "feedback": feedback,
        "note": note,
        "source": "heuristic",
    }


# 1. Define the Shared State (The Clipboard)
class State(TypedDict):
    responses: list          # item responses (rows × items), as list of dicts for DataFrame
    rt_data: list            # response times (rows × items), as list of dicts for DataFrame
    theta: float
    latency_flags: list[str]
    next_step: str
    model_settings: NotRequired[dict]
    is_verified: NotRequired[bool]
    rt_plot_path: NotRequired[str]  # path to RT histogram figure (set by irt_agent)
    icc_plot_path: NotRequired[str]
    icc_error: NotRequired[str]
    item_params: NotRequired[list[dict]]
    person_params: NotRequired[list[dict]]
    item_fit: NotRequired[list[dict]]


def _dv_python(rt_df: pd.DataFrame, resp_df: pd.DataFrame, color: str = "lightgray") -> str:
    """Python port of R's dv(): RT histograms with correct proportions. Returns path to saved figure."""
    nresp = len(resp_df)
    nrt = rt_df.shape[1]
    nresp_cols = resp_df.shape[1]
    if nrt != nresp_cols:
        raise ValueError(f"RT and Resp must have the same number of columns. RT: {nrt}, Resp: {nresp_cols}")
    if nresp == 0:
        raise ValueError("Resp has no rows")
    if nrt == 0:
        raise ValueError("RT has no columns")
    p = resp_df.sum(axis=0)
    p1 = (p / nresp).round(2)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    n_show = min(nrt, 12)  # 3x4 = 12 panels; avoid IndexError if nrt > 12
    for i in range(n_show):
        ax = axes[i]
        x = rt_df.iloc[:, i].dropna()
        ax.hist(x, bins=15, color=color, edgecolor="white")
        prop = p1.iloc[i]
        prop_str = f"{prop:.2f}" if pd.notna(prop) else "—"
        ax.set_title(f"RT Distr. for Item {i + 1}\nCorrect Proportion: {prop_str}", fontsize=9)
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    # Prefer data/rt_hist.png; fall back to temp if project dir has permission issues (e.g. Box sync)
    candidates = [_GRAPH_DIR / "data" / "rt_hist.png", Path(tempfile.gettempdir()) / "psych_mas_rt_hist.png"]
    out_path = None
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=100, bbox_inches="tight")
            out_path = p
            break
        except OSError:
            continue
    plt.close(fig)
    if out_path is None:
        raise RuntimeError("Could not save rt_hist.png to project data/ or temp dir")
    return str(out_path)


# def _plot_icc(resp_df: pd.DataFrame) -> str:
#     # Placeholder for IRT logic (e.g., using a library like girth)
#     # 1. Fit your IRT model
#     # 2. Generate the S-shaped ICC curves
#     fig, ax = plt.subplots()
#     # ... plotting logic ...
#     path = _GRAPH_DIR / "data" / "icc_plot.png"
#     fig.savefig(path)
#     plt.close(fig)
#     return str(path)

# 2. Define the IRT Agent Node (uses Python/matplotlib; use rpy2 when you need an R package)
# def irt_agent(state):
#     print("--- IRT AGENT: Running RT visualizations (Python) ---")
#     resp_df = pd.DataFrame(state["responses"])
#     rt_df = pd.DataFrame(state["rt_data"])
#     out = {"theta": 0.85, "next_step": "analyze_timing"}

#     try:
#         path = _dv_python(rt_df, resp_df, color="skyblue")
#         out["rt_plot_path"] = path
#         print("RT histogram saved successfully.")
#     except Exception as e:
#         print(f"Error generating RT plot: {type(e).__name__}: {e}")

#     return out
def irt_agent(state):
    if not state.get("is_verified"):
        return {
            "next_step": "analyze_timing",
            "icc_error": "Model settings not verified. Confirm settings to run IRT.",
        }

    resp_df = pd.DataFrame(state["responses"])

    # Guard against non-binary or constant response columns (mirt 2PL requires 0/1 with 2 categories).
    invalid_cols = []
    constant_cols = []
    keep_cols = []
    for col in resp_df.columns:
        values = pd.to_numeric(resp_df[col], errors="coerce").dropna().unique().tolist()
        if not values:
            constant_cols.append(str(col))
            continue
        if not set(values).issubset({0, 1}):
            invalid_cols.append(str(col))
            continue
        if len(set(values)) < 2:
            constant_cols.append(str(col))
            continue
        keep_cols.append(col)
    if invalid_cols or constant_cols:
        details = []
        if invalid_cols:
            details.append(f"non-binary columns: {', '.join(invalid_cols)}")
        if constant_cols:
            details.append(f"single-category columns: {', '.join(constant_cols)}")
        message = "ICC skipped; " + "; ".join(details)
        print(message)
        if not keep_cols:
            return {"next_step": "analyze_timing", "icc_error": message}

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except Exception as exc:
        message = f"ICC skipped; rpy2/R not available ({type(exc).__name__}: {exc})"
        print(message)
        return {"next_step": "analyze_timing", "icc_error": message}

    settings = state.get("model_settings") or {}
    itemtype = settings.get("itemtype", "2PL")

    icc_path = _GRAPH_DIR / "data" / "icc_plot.png"
    icc_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (ro.default_converter + pandas2ri.converter).context():
            ro.globalenv["df"] = resp_df[keep_cols] if keep_cols else resp_df
            ro.globalenv["icc_path"] = str(icc_path)
            ro.globalenv["itemtype"] = itemtype
            ro.r(
                """
                library(mirt)
                library(grDevices)
                library(lattice)
                n_items <- ncol(df)
                model <- mirt(
                    df,
                    1,
                    itemtype=itemtype,
                    verbose=FALSE,
                    technical=list(NCYCLES=50)
                )
                png(icc_path, width=1400, height=900, res=150, type="cairo")
                p <- plot(model, type="trace", which.items=1:n_items)
                print(p)
                dev.off()

                item_pars <- as.data.frame(coef(model, IRTpars=TRUE, simplify=TRUE)$items)
                item_pars$item <- rownames(item_pars)
                rownames(item_pars) <- NULL

                person_pars <- as.data.frame(fscores(model))
                item_fit <- as.data.frame(itemfit(model))
                item_fit$item <- rownames(item_fit)
                rownames(item_fit) <- NULL
                """
            )
            item_pars = ro.conversion.rpy2py(ro.r("item_pars"))
            person_pars = ro.conversion.rpy2py(ro.r("person_pars"))
            item_fit = ro.conversion.rpy2py(ro.r("item_fit"))
    except Exception as exc:
        message = f"ICC skipped; R error ({type(exc).__name__}: {exc})"
        print(message)
        return {"next_step": "analyze_timing", "icc_error": message}

    if not icc_path.exists():
        message = "ICC skipped; plot file was not created."
        print(message)
        return {"next_step": "analyze_timing", "icc_error": message}

    return {
        "icc_plot_path": str(icc_path),
        "item_params": item_pars.to_dict(orient="records"),
        "person_params": person_pars.to_dict(orient="records"),
        "item_fit": item_fit.to_dict(orient="records"),
        "next_step": "analyze_timing",
    }
        

# 3. Define the RT Agent Node
def rt_agent(state: State):
    print("--- RT AGENT ANALYZING LATENCY ---")
    # Simulating a check for rapid guessing. Do not return next_step:
    # irt and rt run in parallel; only one value per key per step is allowed.
    return {"latency_flags": ["item_3_rapid_guess"]}


# 4. Define the Orchestrator Agent Node
def orchestrator_agent(state: State):
    print("--- ORCHESTRATOR DISPATCHING ---")
    # Routes to irt and rt; pass through state
    return {"next_step": "dispatch"}


# 5. Define the Analyze Agent Node
def analyze_agent(state: State):
    print("--- ANALYZE COMBINING THETA + LATENCY ---")
    # Combine theta and latency_flags; placeholder for real logic
    return {"next_step": "end"}


# 6. Connect the Agents (The Graph)
workflow = StateGraph(State)

workflow.add_node("Orchestrator_node", orchestrator_agent)
workflow.add_node("irt_node", irt_agent)
workflow.add_node("rt_node", rt_agent)
workflow.add_node("Analyze_node", analyze_agent)

workflow.set_entry_point("Orchestrator_node")
workflow.add_edge("Orchestrator_node","irt_node")
workflow.add_edge("Orchestrator_node","rt_node")
workflow.add_edge("irt_node","Analyze_node")
workflow.add_edge("rt_node","Analyze_node")

workflow.add_edge("Analyze_node", END)



# 7. Compile the Graph
psych_workflow = app = workflow.compile()