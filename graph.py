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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call_openrouter(api_key: str, model_id: str, messages: list[dict], timeout: int = 30) -> tuple[str | None, str | None]:
    """Call OpenRouter chat completions. Return (text, None) on success or (None, error_message)."""
    if not model_id or not messages:
        return None, "No model or messages."
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        body = {"model": model_id, "messages": messages}
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        return (text.strip() or "No response from model.", None)
    except requests.exceptions.HTTPError as e:
        try:
            err_body = e.response.json() if e.response is not None else {}
            msg = err_body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        code = e.response.status_code if e.response is not None else 0
        if code == 402:
            msg += " Add credits at openrouter.ai/credits (free models require a non-negative balance)."
        return None, f"{code}: {msg}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _get_llm_model_variants(api_key: str) -> list[str]:
    """Return list of model names to try: discovered from API first, then hardcoded fallback."""
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        resp = requests.get(list_url, timeout=15)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
        candidates = []
        for m in models_data:
            methods = m.get("supportedGenerationMethods", [])
            name = m.get("name", "")
            if "generateContent" in methods and "gemini" in name.lower():
                candidates.append(name)
        # Prefer flash (faster), then pro
        flash = [n for n in candidates if "flash" in n.lower()]
        pro = [n for n in candidates if "pro" in n.lower() and n not in flash]
        other = [n for n in candidates if n not in flash and n not in pro]
        if flash or pro or other:
            return list(dict.fromkeys(flash + pro + other))
    except Exception:
        pass
    # Fallback: names that are commonly available (no gemini-pro or bare gemini-1.5-flash)
    return [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
    ]


def _llm_generate_text(api_key: str, prompt_text: str, model_variants: list) -> str | None:
    """Call LLM once; return raw text (1–2 sentence response) or None on failure."""
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": (
                        "The user said: \"{}\"\n\n"
                        "Reply in one or two short sentences only. "
                        "If they ask about IRT models (1PL, 2PL, 3PL, 4PL), say what you recommend; "
                        "if they ask something else (e.g. who are you, what is today), answer directly "
                        "then add: 'This app is for psychometric analysis (IRT); I can help with models like 1PL, 2PL, 3PL, or 4PL.'"
                    ).format(prompt_text.replace('"', '\\"'))
                }]
            }
        ]
    }
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text.strip() if text else None
        except Exception:
            continue
    return None


def analyze_prompt(
    prompt: str,
    *,
    llm_provider: str = "google",
    llm_model_id: str | None = None,
    openrouter_api_key: str | None = None,
    google_api_key: str | None = None,
) -> dict:
    normalized = (prompt or "").strip()
    if not normalized:
        return {
            "itemtype": "2PL",
            "r_code": "model <- mirt(df, 1, itemtype='2PL')",
            "suggestion": "Defaulting to 2PL.",
            "feedback": "Please describe your goal (e.g., guessing behavior, Rasch/1PL, or asymmetric response patterns).",
            "reason": "No model preference provided.",
            "source": "heuristic",
        }

    load_dotenv()
    use_openrouter = (llm_provider or "").strip().lower() == "openrouter" and openrouter_api_key and llm_model_id
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

    if use_openrouter:
        # Step 1: OpenRouter one- or two-sentence response
        feedback_prompt = (
            "The user said: \"{}\"\n\n"
            "Reply in one or two short sentences only. "
            "If they ask about IRT models (1PL, 2PL, 3PL, 4PL), say what you recommend; "
            "if they ask something else (e.g. who are you, what is today), answer directly "
            "then add: 'This app is for psychometric analysis (IRT); I can help with models like 1PL, 2PL, 3PL, or 4PL.'"
        ).format(normalized.replace('"', '\\"'))
        text, err = _call_openrouter(openrouter_api_key.strip(), llm_model_id, [{"role": "user", "content": feedback_prompt}], timeout=30)
        feedback_text = text if text and not err else None
        # Step 2: OpenRouter structured JSON
        system = (
            "You are a psychometrics assistant for an IRT app. Return ONLY valid JSON with keys: "
            "itemtype (1PL/2PL/3PL/4PL), suggestion, reason, r_code. "
            "Map 'basic'/'simple' model -> 1PL; guessing -> 3PL; asymmetry -> 4PL; Rasch/1PL -> 1PL. "
            "If the prompt is not about IRT models, use itemtype='2PL' and empty suggestion and reason. "
            "r_code must be: model <- mirt(df, 1, itemtype='XPL') with X = 1, 2, 3, or 4."
        )
        user = f"Prompt: {normalized}\n\nReturn JSON only (no other text): itemtype, suggestion, reason, r_code."
        text2, err2 = _call_openrouter(openrouter_api_key.strip(), llm_model_id, [{"role": "user", "content": f"{system}\n\n{user}"}], timeout=30)
        if text2 and not err2:
            try:
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text2, re.DOTALL)
                if json_match:
                    text2 = json_match.group(1)
                else:
                    m = re.search(r'\{.*\}', text2, re.DOTALL)
                    if m:
                        text2 = m.group(0)
                parsed = json.loads(text2)
                itemtype = str(parsed.get("itemtype", "2PL")).upper()
                if itemtype not in {"1PL", "2PL", "3PL", "4PL"}:
                    itemtype = "2PL"
                suggestion = parsed.get("suggestion", "")
                reason = parsed.get("reason", "")
                r_code = parsed.get("r_code", f"model <- mirt(df, 1, itemtype='{itemtype}')")
                if "itemtype=" not in r_code and "itemtype'" not in r_code:
                    r_code = f"model <- mirt(df, 1, itemtype='{itemtype}')"
                feedback = feedback_text or parsed.get("feedback", "Here is a suggested model based on your prompt.")
                return {
                    "itemtype": itemtype,
                    "r_code": r_code,
                    "suggestion": suggestion,
                    "reason": reason,
                    "feedback": feedback,
                    "note": parsed.get("note", ""),
                    "source": "llm",
                }
            except (json.JSONDecodeError, Exception):
                pass
        if feedback_text:
            out = _heuristic_prompt_mapping(prompt)
            out["feedback"] = feedback_text
            out["source"] = "llm"
            return out
        return _heuristic_prompt_mapping(prompt)

    if not api_key:
        return _heuristic_prompt_mapping(prompt)

    # Prefer configured model, then discovered models, then fallback list
    configured = os.getenv("GEMINI_MODEL", "").strip()
    model_variants = _get_llm_model_variants(api_key)
    if configured and configured not in model_variants:
        model_variants = [configured] + model_variants

    # Step 1: Use LLM to generate a one- or two-sentence response first (no JSON)
    feedback_text = _llm_generate_text(api_key, normalized, model_variants)

    # Step 2: Get structured response (itemtype, suggestion, reason, r_code) from LLM
    system = (
        "You are a psychometrics assistant for an IRT app. Return ONLY valid JSON with keys: "
        "itemtype (1PL/2PL/3PL/4PL), suggestion, reason, r_code. "
        "Map 'basic'/'simple' model -> 1PL; guessing -> 3PL; asymmetry -> 4PL; Rasch/1PL -> 1PL. "
        "If the prompt is not about IRT models, use itemtype='2PL' and empty suggestion and reason. "
        "r_code must be: model <- mirt(df, 1, itemtype='XPL') with X = 1, 2, 3, or 4."
    )
    user = f"Prompt: {normalized}\n\nReturn JSON only (no other text): itemtype, suggestion, reason, r_code."
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}
        ]
    }

    resp = None
    last_error = None
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                last_error = e
                resp = None
                continue
            raise
        except Exception as e:
            last_error = e
            resp = None
            continue

    if resp is None or resp.status_code != 200:
        # Structured call failed; if we have feedback from step 1, use it with heuristic for structure
        if feedback_text:
            out = _heuristic_prompt_mapping(prompt)
            out["feedback"] = feedback_text
            out["source"] = "llm"
            return out
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
            if feedback_text:
                out = _heuristic_prompt_mapping(prompt)
                out["feedback"] = feedback_text
                out["source"] = "llm"
                return out
            return _heuristic_prompt_mapping(prompt)

        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if feedback_text:
                out = _heuristic_prompt_mapping(prompt)
                out["feedback"] = feedback_text
                out["source"] = "llm"
                return out
            return _heuristic_prompt_mapping(prompt)

        itemtype = str(parsed.get("itemtype", "2PL")).upper()
        if itemtype not in {"1PL", "2PL", "3PL", "4PL"}:
            itemtype = "2PL"
        suggestion = parsed.get("suggestion", "")
        reason = parsed.get("reason", "")
        r_code = parsed.get("r_code", f"model <- mirt(df, 1, itemtype='{itemtype}')")
        if "itemtype=" not in r_code and "itemtype'" not in r_code:
            r_code = f"model <- mirt(df, 1, itemtype='{itemtype}')"
        # Use the 1–2 sentence response from step 1 as feedback when we have it
        feedback = feedback_text if feedback_text else parsed.get("feedback", "Here is a suggested model based on your prompt.")
        return {
            "itemtype": itemtype,
            "r_code": r_code,
            "suggestion": suggestion,
            "reason": reason,
            "feedback": feedback,
            "note": parsed.get("note", ""),
            "source": "llm",
        }
    except Exception as e:
        print(f"LLM call failed: {type(e).__name__}: {e}")
        if feedback_text:
            out = _heuristic_prompt_mapping(prompt)
            out["feedback"] = feedback_text
            out["source"] = "llm"
            return out
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
    elif "basic" in normalized or ("simple" in normalized and "model" in normalized):
        itemtype = "1PL"
        suggestion = "Basic/simple model requested; recommend 1PL (Rasch)."
        reason = "In IRT, the basic or simplest model is 1PL (one parameter: difficulty)."
        feedback = "Using 1PL (Rasch): the basic model with a single difficulty parameter per item."
    elif normalized and not is_greeting and not any(
        token in normalized for token in ["1pl", "2pl", "3pl", "4pl", "rasch", "guess", "basic", "simple"]
    ):
        psychometrics_reminder = (
            "This app is designed for psychometric analysis using Item Response Theory (IRT). "
            "I can help you analyze test data with models like 1PL, 2PL, 3PL, or 4PL—for example: "
            "'I suspect guessing, use 3PL' or 'Use Rasch/1PL model'."
        )
        suggestion = ""
        reason = ""
        note = ""
        # Answer identity questions directly, then remind
        is_identity_question = any(
            phrase in normalized
            for phrase in ["who are you", "what are you", "who is this", "what is this", "who're you", "what're you"]
        )
        if is_identity_question:
            feedback = f"I'm a psychometrics assistant for this app. {psychometrics_reminder}"
        else:
            off_topic_keywords = ["weather", "time", "date", "today", "news", "sport", "movie", "music", "food", "recipe", "translate", "calculate", "math", "python", "code", "programming"]
            is_off_topic = any(keyword in normalized for keyword in off_topic_keywords)
            if is_off_topic:
                feedback = f"That's outside what I can help with here. {psychometrics_reminder}"
            else:
                feedback = f"I'm not sure how to help with that. {psychometrics_reminder}"
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
    model_fit: NotRequired[dict]  # M2 and related overall fit stats from mirt::M2


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
        message = (
            f"ICC skipped; rpy2/R not available ({type(exc).__name__}: {exc}). "
            "IRT, ICC, and Wright Map require R (mirt, WrightMap). Run the app locally with R installed for full analysis."
        )
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
                model_fit_df <- tryCatch(M2(model), error = function(e) data.frame(M2 = NA_real_, df = NA_real_, p = NA_real_, RMSEA = NA_real_, SRMSR = NA_real_, TLI = NA_real_, CFI = NA_real_, stringsAsFactors = FALSE))
                model_fit_list <- as.list(model_fit_df[1, ])
                names(model_fit_list) <- colnames(model_fit_df)
                """
            )
            item_pars = ro.conversion.rpy2py(ro.r("item_pars"))
            person_pars = ro.conversion.rpy2py(ro.r("person_pars"))
            item_fit = ro.conversion.rpy2py(ro.r("item_fit"))
            model_fit = {}
            try:
                model_fit_df = ro.conversion.rpy2py(ro.r("model_fit_df"))
                if hasattr(model_fit_df, "iloc") and len(model_fit_df) > 0:
                    model_fit = model_fit_df.iloc[0].to_dict()
            except Exception:
                pass
            if not model_fit:
                try:
                    model_fit_r = ro.r("model_fit_list")
                    rnames = getattr(model_fit_r, "names", None)
                    if rnames is not None and len(rnames) > 0:
                        model_fit = {}
                        for i, name in enumerate(rnames):
                            try:
                                val = model_fit_r[i]
                                model_fit[str(name)] = ro.conversion.rpy2py(val)
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception as exc:
        message = (
            f"ICC skipped; R error ({type(exc).__name__}: {exc}). "
            "Run the app locally with R and packages mirt, WrightMap for IRT analysis."
        )
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
        "model_fit": model_fit,
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