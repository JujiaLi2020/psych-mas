from pathlib import Path
import json
import os
import re
import tempfile
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

from psymas_graph.state import State
from psymas_graph.rt_visuals import response_time_histograms as _dv_python
from psymas_graph.serialization import records as _records
from psymas_graph.llm_client import (
    call_openrouter as _call_openrouter,
    generate_google_text as _llm_generate_text,
    google_model_variants as _get_llm_model_variants,
)
from psymas_graph.thresholds import (
    orient_pair_flag_matrix as _orient_pair_flag_matrix,
    pairwise_alpha as _pairwise_alpha,
    r_number as _r_num,
    threshold_alpha as _threshold_alpha,
    threshold_block as _threshold_block,
    threshold_enabled as _threshold_enabled,
    threshold_float as _threshold_float,
    threshold_rules as _threshold_rules,
)
from psymas_graph.workflows import (
    FORENSIC_SPECIALIST_AGENT_ORDER,
    build_forensic_workflow,
    build_psychometric_workflow,
)


_GRAPH_DIR = Path(__file__).resolve().parent

# To run an R package from a node (rt_agent, analyze_agent, etc.):
#   import rpy2.robjects as ro
#   from rpy2.robjects import pandas2ri
#   with (ro.default_converter + pandas2ri.converter).context():
#       ro.r("library(mirt)")   # or your package
#       ro.globalenv["df"] = my_pandas_df
#       ro.r("result <- some_r_func(df)")
# rpy2 is in pyproject.toml; ensure R is on PATH.

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
        try:
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
            _converter = ro.default_converter + pandas2ri.converter + numpy2ri.converter
        except Exception:
            _converter = ro.default_converter + pandas2ri.converter
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
        # Run all R and rpy2py inside a conversion context so it works in uvicorn worker threads.
        with _converter.context():
            # Build df in R from pure Python list (no numpy) to avoid py2rpy numpy.ndarray
            block = resp_df[keep_cols] if keep_cols else resp_df
            nrow, ncol = int(block.shape[0]), int(block.shape[1])
            x_flat = []
            for i in range(nrow):
                for j in range(ncol):
                    v = block.iloc[i, j]
                    x_flat.append(int(v) if v is not None and str(v) not in ("nan", "NaN", "") else 0)
            ro.globalenv["x_vec"] = ro.IntVector(x_flat)
            ro.globalenv["icc_path"] = str(icc_path)
            ro.globalenv["itemtype"] = itemtype
            ro.r(f"df <- as.data.frame(matrix(x_vec, nrow={nrow}, ncol={ncol}, byrow=TRUE))")
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
            with _converter.context():
                model_fit_df = ro.conversion.rpy2py(ro.r("model_fit_df"))
                if hasattr(model_fit_df, "iloc") and len(model_fit_df) > 0:
                    model_fit = model_fit_df.iloc[0].to_dict()
        except Exception:
            pass
        if not model_fit:
            try:
                with _converter.context():
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
        "item_params": _records(item_pars),
        "person_params": _records(person_pars),
        "item_fit": _records(item_fit),
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


# 5b. Define the Aberrance Agent Node (R package aberrance: nonparametric misfit, etc.)
def aberrance_agent(state: State):
    """Run R package aberrance (e.g. detect_nm for nonparametric person misfit). Uses state['aberrance_functions'] when provided."""
    print("--- ABERRANCE AGENT: Running R package aberrance ---")
    selected = state.get("aberrance_functions") or []
    run_nm = "detect_nm" in selected if selected else True
    run_pm = "detect_pm" in selected if selected else True
    run_ac = "detect_ac" in selected if selected else False
    run_pk = "detect_pk" in selected if selected else False
    run_rg = "detect_rg" in selected if selected else False
    run_tt = "detect_tt" in selected if selected else False
    resp_df = pd.DataFrame(state["responses"])
    # Use same binary-column logic as irt_agent for dichotomous scores
    keep_cols = []
    for col in resp_df.columns:
        values = pd.to_numeric(resp_df[col], errors="coerce").dropna().unique().tolist()
        if not values:
            continue
        if not set(values).issubset({0, 1}):
            continue
        if len(set(values)) < 2:
            continue
        keep_cols.append(col)
    if not keep_cols:
        return {
            "aberrance_results": {"error": "No valid dichotomous item columns for aberrance (need 0/1 with variation)."},
            "next_step": "end",
        }
    if selected and not (run_nm or run_pm or run_ac or run_pk or run_rg or run_tt):
        return {
            "aberrance_results": {"info": "No detection functions selected. Select detect_nm, detect_pm, detect_ac, detect_pk, detect_rg, and/or detect_tt."},
            "next_step": "end",
        }
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        try:
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
        except Exception:
            pass
    except Exception as exc:
        return {
            "aberrance_results": {"error": f"rpy2/R not available: {exc}. Install R and rpy2 for aberrance analysis."},
            "next_step": "end",
        }
    import numpy as np
    records = None
    methods = ["ZU3_S", "HT_S"]  # default so we never return methods=[]
    flagged_persons = []
    # When pm/ac/pk are selected, always run detect_nm too so we have a baseline table and methods (ZU3_S, HT_S)
    # even if parametric/preknowledge fail or need extra inputs (e.g. compromised_items for detect_pk)
    item_params = state.get("item_params")
    if not run_nm and (run_pm or run_ac or run_pk):
        run_nm = True
    if not run_nm and (run_rg or run_tt):
        run_nm = True  # run_rg: need baseline table; run_tt: no erasure data so run nonparametric only
    if not run_nm and (not run_pm or not item_params or len(item_params) == 0) and not (run_ac and item_params) and not (run_pk and item_params):
        run_nm = True
    try:
        ro.r("library(aberrance)")
        # Build x in R from pure Python list (no numpy) to avoid py2rpy numpy.ndarray
        block = resp_df[keep_cols]
        nrow, ncol = int(block.shape[0]), int(block.shape[1])
        x_flat = []
        for i in range(nrow):
            for j in range(ncol):
                v = block.iloc[i, j]
                x_flat.append(int(v) if v is not None and str(v) not in ("nan", "NaN", "") else 0)
        ro.globalenv["x_vec"] = ro.IntVector(x_flat)
        ro.r(f"x <- matrix(x_vec, nrow={nrow}, ncol={ncol}, byrow=TRUE)")
        if run_nm:
            # Nonparametric person-fit: run the full set of score-based indices
            # (G_S, NC_S, U1_S, U3_S, ZU3_S, A_S, D_S, E_S, C_S, MC_S, PC_S, HT_S).
            ro.r("out <- detect_nm(method = c('G_S','NC_S','U1_S','U3_S','ZU3_S','A_S','D_S','E_S','C_S','MC_S','PC_S','HT_S'), x = x)")
            ro.r("stat_r <- as.data.frame(out$stat)")
            stat_r = ro.r("stat_r")
            stat_df = ro.conversion.rpy2py(stat_r)
            methods = ["ZU3_S", "HT_S"]
            if stat_df is not None and hasattr(stat_df, "to_dict"):
                records = stat_df.to_dict(orient="records")
                if hasattr(stat_df, "columns"):
                    methods = list(stat_df.columns)
            elif stat_df is not None and hasattr(stat_df, "__array__"):
                arr = np.asarray(stat_df)
                if arr.size > 0 and arr.ndim == 2:
                    try:
                        col_names = ro.conversion.rpy2py(ro.r("colnames(out$stat)"))
                        if col_names is not None and len(col_names) == arr.shape[1]:
                            methods = list(col_names)
                    except Exception:
                        pass
                    records = [dict(zip(methods, row)) for row in arr.tolist()]
            if records:
                df_nm = pd.DataFrame(records)
                if "ZU3_S" in df_nm.columns:
                    zu3 = pd.to_numeric(df_nm["ZU3_S"], errors="coerce")
                    if zu3.notna().any():
                        flagged_persons.extend(np.where(zu3 < -2)[0].tolist())
                if "HT_S" in df_nm.columns:
                    ht = pd.to_numeric(df_nm["HT_S"], errors="coerce")
                    if ht.notna().any():
                        q05 = ht.quantile(0.05)
                        if pd.notna(q05):
                            flagged_persons.extend(np.where(ht <= q05)[0].tolist())
                flagged_persons = sorted(set(flagged_persons))
        n_persons = len(records) if records else resp_df[keep_cols].shape[0]
        # methods: only include nonparametric method names when we actually ran detect_nm
        result = {
            "nonparametric_misfit": records or [],
            "methods": list(methods) if (records or []) else [],
            "n_persons": n_persons,
            "flagged_persons": flagged_persons,
            "n_flagged": len(flagged_persons),
        }
        if run_nm and not records:
            return {
                "aberrance_results": {"error": "aberrance detect_nm returned no stat matrix."},
                "next_step": "end",
            }
        # When run_nm was False (only pm/ac/pk selected), initialize result so run_pm/run_pk can run
        if not run_nm:
            n_persons = resp_df[keep_cols].shape[0]
            result = {
                "nonparametric_misfit": [],
                "methods": [],
                "n_persons": n_persons,
                "flagged_persons": [],
                "n_flagged": 0,
            }
        # Parametric person-fit (detect_pm) when selected and IRT item params available
        item_params = state.get("item_params")
            # Create psi in R when any of pm/ac/pk need it (so detect_ac and detect_pk can use psi even if detect_pm is not selected)
        psi_ready = False
        if (run_pm or run_ac or run_pk) and item_params and len(item_params) == len(keep_cols):
            try:
                ip_df = pd.DataFrame(item_params)
                a_col = "a1" if "a1" in ip_df.columns else "a"
                b_col = "b"
                # detect_pm requires psi with columns a, b, c (guessing)
                # IRT (mirt IRTpars=TRUE) returns: a, b, g, u — 'g' is the guessing param → map to 'c'
                c_col = "g" if "g" in ip_df.columns else ("c" if "c" in ip_df.columns else None)
                if a_col in ip_df.columns and b_col in ip_df.columns:
                    a_vals = [float(v) for v in ip_df[a_col].tolist()]
                    b_vals = [float(v) for v in ip_df[b_col].tolist()]
                    ro.globalenv["psi_a"] = ro.FloatVector(a_vals)
                    ro.globalenv["psi_b"] = ro.FloatVector(b_vals)
                    if c_col and c_col in ip_df.columns:
                        c_vals = [float(v) for v in ip_df[c_col].tolist()]
                    else:
                        # 2PL: guessing = 0 for all items
                        c_vals = [0.0] * len(a_vals)
                    ro.globalenv["psi_c"] = ro.FloatVector(c_vals)
                    ro.r("psi <- as.matrix(cbind(a = psi_a, b = psi_b, c = psi_c))")
                    # Verify psi was created correctly
                    psi_ncol = int(ro.r("ncol(psi)")[0])
                    psi_nrow = int(ro.r("nrow(psi)")[0])
                    if psi_nrow == len(keep_cols) and psi_ncol == 3:
                        psi_ready = True
                    else:
                        result["info"] = (result.get("info") or "") + f" ψ matrix has unexpected dimensions [{psi_nrow}, {psi_ncol}]; expected [{len(keep_cols)}, 3]."
                else:
                    avail_cols = list(ip_df.columns)
                    result["info"] = (result.get("info") or "") + f" Item params missing '{a_col}' or '{b_col}'. Available columns: {avail_cols}."
            except Exception as e:
                result["info"] = (result.get("info") or "") + f" Could not build ψ: {e}"
        elif (run_pm or run_ac or run_pk) and not item_params:
            result["info"] = (result.get("info") or "") + " No item parameters available; IRT may not have run. Model misfit / answer copying / preknowledge skipped."
        elif (run_pm or run_ac or run_pk) and item_params and len(item_params) != len(keep_cols):
            result["info"] = (result.get("info") or "") + f" Item params ({len(item_params)}) vs response columns ({len(keep_cols)}) mismatch; cannot build ψ."
        if run_pm and psi_ready:
            try:
                # detect_pm requires psi with columns a, b, c, alpha, beta AND y (log RT)
                # Estimate alpha/beta from RT data if available; otherwise score-only fallback
                pm_rt_data = state.get("rt_data") or []
                pm_n_persons = resp_df[keep_cols].shape[0]
                pm_has_rt = bool(pm_rt_data) and len(pm_rt_data) == pm_n_persons
                if pm_has_rt:
                    # Build log-RT matrix y and estimate alpha/beta per item
                    pm_rt_df = pd.DataFrame(pm_rt_data)
                    pm_rt_block = pm_rt_df.iloc[:, :len(keep_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.01)
                    # y = log response time matrix
                    y_flat = np.log(pm_rt_block.values.clip(min=0.001)).flatten().tolist()
                    ro.globalenv["y_vec"] = ro.FloatVector(y_flat)
                    ro.r(f"y <- matrix(y_vec, nrow={pm_n_persons}, ncol={len(keep_cols)}, byrow=FALSE)")
                    # Estimate alpha (time discrimination) and beta (time intensity) per item
                    ro.r("""
                        pm_beta_est  <- apply(y, 2, mean)
                        pm_alpha_est <- 1 / apply(y, 2, sd)
                        pm_alpha_est[!is.finite(pm_alpha_est)] <- 1.0
                        psi <- cbind(psi, alpha = pm_alpha_est, beta = pm_beta_est)
                    """)
                    # Call detect_pm with x AND y
                    ro.r("""
                        assign('pm_err', NULL, envir = .GlobalEnv)
                        pm_out <- tryCatch(
                            detect_pm(method = c('L_S_TS', 'L_T', 'Q_ST_TS', 'L_ST_TS'), psi = psi, x = x, y = y, alpha = %s),
                            error = function(e) { assign('pm_err', conditionMessage(e), envir = .GlobalEnv); NULL }
                        )
                    """ % _r_num(_threshold_alpha(state, "detect_pm")))
                else:
                    # No RT data: try score-only call (psi = a, b, c only)
                    ro.r("""
                        assign('pm_err', NULL, envir = .GlobalEnv)
                        pm_out <- tryCatch(
                            detect_pm(method = c('L_S_TS', 'L_T', 'Q_ST_TS', 'L_ST_TS'), psi = psi, x = x, alpha = %s),
                            error = function(e) { assign('pm_err', conditionMessage(e), envir = .GlobalEnv); NULL }
                        )
                    """ % _r_num(_threshold_alpha(state, "detect_pm")))
                has_pm_r = ro.r("!is.null(pm_out)")
                # Convert has_pm to Python bool safely
                try:
                    has_pm_py = ro.conversion.rpy2py(has_pm_r)
                    has_pm = bool(list(has_pm_py)[0]) if hasattr(has_pm_py, '__iter__') else bool(has_pm_py)
                except Exception:
                    has_pm = False
                if has_pm:
                    # $stat is matrix [n_persons, n_methods]; $pval same; $flag is [n_persons, n_methods, 1]
                    # Extract dimensions and method names entirely in R, return as single strings/ints
                    ro.r("""
                        pm_stat_mat  <- pm_out$stat
                        pm_pval_mat  <- pm_out$pval
                        pm_flag_arr  <- pm_out$flag
                        pm_n         <- as.integer(nrow(pm_stat_mat))
                        pm_methods   <- colnames(pm_stat_mat)
                        pm_n_methods <- as.integer(length(pm_methods))
                        pm_methods_str <- paste(pm_methods, collapse = '|')
                    """)
                    pm_n = int(ro.r("pm_n")[0])
                    pm_n_methods = int(ro.r("pm_n_methods")[0])
                    pm_methods_str = str(ro.r("pm_methods_str")[0])
                    pm_methods_r = pm_methods_str.split("|") if pm_methods_str else []
                    if pm_n > 0 and pm_n_methods > 0 and len(pm_methods_r) == pm_n_methods:
                        # Read each stat and pval column as a plain list via R indexing
                        pm_data = {}
                        for mi in range(pm_n_methods):
                            mname = pm_methods_r[mi]
                            stat_r = ro.r(f"as.numeric(pm_stat_mat[, {mi + 1}])")
                            pm_data[mname] = [float(v) for v in stat_r]
                            pval_r = ro.r(f"as.numeric(pm_pval_mat[, {mi + 1}])")
                            pm_data[f"{mname}_pval"] = [float(v) for v in pval_r]
                        pm_df = pd.DataFrame(pm_data)
                        result["parametric_misfit"] = pm_df.to_dict(orient="records")
                        # Flag: 3D array [n_persons, n_methods, 1]; person flagged if ANY method flags
                        try:
                            flag_any_r = ro.r("as.logical(apply(pm_flag_arr, 1, any))")
                            pm_flagged = [i for i, v in enumerate(flag_any_r) if v]
                            result["flagged_persons_pm"] = pm_flagged
                            result["flagged_persons"] = sorted(set(flagged_persons + pm_flagged))
                            result["n_flagged"] = len(result["flagged_persons"])
                        except Exception:
                            pass
                    else:
                        result["info"] = (result.get("info") or "") + f" detect_pm stat unexpected ({pm_n}×{pm_n_methods}, methods={pm_methods_r})."
                else:
                    pm_err_r = ro.r("get0('pm_err', envir = .GlobalEnv, ifnotfound = 'unknown error')")
                    pm_err_msg = str(ro.conversion.rpy2py(pm_err_r)) if pm_err_r is not None else "unknown"
                    result["info"] = (result.get("info") or "") + f" detect_pm failed. R: {pm_err_msg}"
            except Exception as e:
                result["info"] = (result.get("info") or "") + f" detect_pm error: {e}"
        # Answer copying (detect_ac): needs psi, x; returns stat/pval/flag per (source, copier) pair
        if run_ac and psi_ready:
            try:
                ro.r(f"ac_out <- tryCatch(detect_ac(method = c('OMG_S', 'GBT_S'), psi = psi, x = x, alpha = {_r_num(_threshold_alpha(state, 'detect_ac'))}), error = function(e) NULL)")
                has_ac = ro.r("!is.null(ac_out)")
                if has_ac and ro.conversion.rpy2py(has_ac):
                    ac_stat = ro.r("as.data.frame(ac_out$stat)")
                    ac_stat_py = ro.conversion.rpy2py(ac_stat)
                    ac_flag = ro.r("ac_out$flag")
                    ac_flag_py = ro.conversion.rpy2py(ac_flag) if ac_flag is not None else None
                    N = resp_df[keep_cols].shape[0]
                    # R uses combn(N, 2) order: (1,2), (1,3), ..., (N-1,N) -> n_pairs = N*(N-1)/2
                    pairs_0based = [(i, j) for i in range(N) for j in range(i + 1, N)]
                    if ac_stat_py is not None and len(pairs_0based) > 0:
                        if hasattr(ac_stat_py, "to_dict"):
                            ac_records = ac_stat_py.to_dict(orient="records")
                        else:
                            arr = np.asarray(ac_stat_py)
                            if arr.ndim == 2 and arr.shape[0] >= len(pairs_0based):
                                cols = getattr(ac_stat_py, "columns", None) or [f"AC_{k}" for k in range(arr.shape[1])]
                                ac_records = [dict(zip(cols, row)) for row in arr[: len(pairs_0based)].tolist()]
                            else:
                                ac_records = []
                        pair_rows = []
                        flagged_copiers = set()
                        for idx, (i, j) in enumerate(pairs_0based):
                            if idx >= len(ac_records):
                                break
                            row = {"Source": i + 1, "Copier": j + 1, **ac_records[idx]}
                            pair_rows.append(row)
                            if ac_flag_py is not None and hasattr(ac_flag_py, "__array__"):
                                fl = np.asarray(ac_flag_py)
                                if fl.ndim >= 2 and idx < fl.shape[0] and np.any(fl[idx] if fl.ndim == 2 else fl[idx]):
                                    flagged_copiers.add(j)
                        result["answer_copying_pairs"] = pair_rows
                        result["flagged_copiers"] = sorted(flagged_copiers)
                        result["flagged_persons"] = sorted(set(result.get("flagged_persons", flagged_persons) + flagged_copiers))
                        result["n_flagged"] = len(result["flagged_persons"])
            except Exception:
                pass
        # Preknowledge (detect_pk): needs ci (compromised item indices, 1-based), psi, x; returns stat per person
        # R requires at least one secure item (not in ci), so we cannot default to "all items"
        compromised_items = state.get("compromised_items") or []
        if not compromised_items and run_pk and psi_ready and len(keep_cols) > 1:
            compromised_items = list(range(1, len(keep_cols)))  # default: items 1..n-1, leave last as secure
        if run_pk and psi_ready and len(compromised_items) > 0:
            try:
                ci_1based = [int(i) for i in compromised_items if isinstance(i, (int, float))]
                if not ci_1based:
                    ci_1based = [int(x) for x in compromised_items]
                ro.globalenv["ci"] = ro.r("c(" + ",".join(map(str, ci_1based)) + ")")
                ro.r(f"pk_out <- tryCatch(detect_pk(method = c('L_S', 'S_S', 'W_S'), ci = ci, psi = psi, x = x, alpha = {_r_num(_threshold_alpha(state, 'detect_pk'))}), error = function(e) NULL)")
                has_pk = ro.r("!is.null(pk_out)")
                if has_pk and ro.conversion.rpy2py(has_pk):
                    pk_stat = ro.r("as.data.frame(pk_out$stat)")
                    pk_stat_py = ro.conversion.rpy2py(pk_stat)
                    pk_flag = ro.r("pk_out$flag")
                    if pk_stat_py is not None and hasattr(pk_stat_py, "to_dict"):
                        pk_records = pk_stat_py.to_dict(orient="records")
                        result["preknowledge"] = pk_records
                        if pk_flag is not None:
                            flag_py = ro.conversion.rpy2py(pk_flag)
                            if hasattr(flag_py, "__array__"):
                                arr = np.asarray(flag_py)
                                if arr.ndim >= 2:
                                    pk_flagged = np.where(np.any(arr, axis=tuple(range(1, arr.ndim))))[0].tolist()
                                    result["flagged_persons_pk"] = pk_flagged
                                    result["flagged_persons"] = sorted(set(result.get("flagged_persons", flagged_persons) + pk_flagged))
                                    result["n_flagged"] = len(result["flagged_persons"])
            except Exception:
                pass
        # Rapid guessing (detect_rg): needs response time matrix t and optional scores x; method NT = normative threshold
        rt_data = state.get("rt_data") or []
        n_persons_resp = resp_df[keep_cols].shape[0]
        n_items = len(keep_cols)
        rg_error = None  # capture R or conversion error for user message
        if run_rg and rt_data and len(rt_data) == n_persons_resp:
            rt_df = pd.DataFrame(rt_data)
            # Use first n_items columns by position; coerce to numeric, fill NA so R doesn't choke
            try:
                if rt_df.shape[1] >= n_items:
                    block = rt_df.iloc[:, :n_items].apply(
                        lambda s: pd.to_numeric(s, errors="coerce")
                    )
                    block = block.fillna(0.01)  # R often requires no NA; use small positive
                    t_block = block.astype(np.float64)
                else:
                    t_block = None
            except (ValueError, TypeError) as e:
                rg_error = str(e)
                t_block = None
            if t_block is not None and t_block.shape[0] == n_persons_resp and t_block.shape[1] == n_items:
                try:
                    # Build RT matrix entirely in R to avoid all py2rpy numpy issues
                    flat = t_block.values.flatten().tolist()  # plain Python list of floats
                    ro.globalenv["t_vec"] = ro.FloatVector(flat)
                    ro.r(f"t <- matrix(t_vec, nrow={int(t_block.shape[0])}, ncol={int(t_block.shape[1])}, byrow=TRUE)")
                    # Capture R error message if detect_rg fails
                    ro.r("assign('rg_err', NULL, envir = .GlobalEnv); rg_out <- tryCatch(detect_rg(method = 'NT', t = t, x = x, nt = 10), error = function(e) { assign('rg_err', conditionMessage(e), envir = .GlobalEnv); NULL })")
                    has_rg = ro.r("!is.null(rg_out)")
                    if not (has_rg and ro.conversion.rpy2py(has_rg)):
                        re = ro.r("get0('rg_err', envir = .GlobalEnv, ifnotfound = NULL)")
                        if re is not None:
                            rg_error = ro.conversion.rpy2py(re)
                    if has_rg and ro.conversion.rpy2py(has_rg):
                        rg_flag = ro.r("rg_out$flag")
                        if rg_flag is not None:
                            flag_py = ro.conversion.rpy2py(rg_flag)
                            if hasattr(flag_py, "__array__"):
                                arr = np.asarray(flag_py)
                                if arr.size > 0:
                                    if arr.ndim == 1:
                                        rg_flagged = np.where(arr)[0].tolist()
                                    else:
                                        rg_flagged = np.where(np.any(arr, axis=tuple(range(1, arr.ndim))))[0].tolist()
                                    result["flagged_persons_rg"] = rg_flagged
                                    result["flagged_persons"] = sorted(set(result.get("flagged_persons", flagged_persons) + rg_flagged))
                                    result["n_flagged"] = len(result["flagged_persons"])
                        rte = ro.r("rg_out$rte")
                        if rte is not None:
                            rte_py = ro.conversion.rpy2py(rte)
                            if hasattr(rte_py, "__array__"):
                                result["rapid_guessing_rte"] = np.asarray(rte_py).flatten().tolist()
                        result["rapid_guessing"] = True
                    else:
                        r_err = ro.r("rg_err")
                        if r_err is not None and str(r_err) != "NULL":
                            rg_error = ro.conversion.rpy2py(r_err)
                except Exception as e:
                    rg_error = str(e)
        if rg_error:
            result["rapid_guessing_error"] = rg_error
        # methods: list all method names from whatever functions actually ran (not just detect_nm)
        all_methods = list(result.get("methods", []))
        if result.get("parametric_misfit"):
            # Get actual method names from the parametric_misfit records (e.g. L_S_TS, L_T, Q_ST_TS, L_ST_TS)
            pm_cols = [k for k in result["parametric_misfit"][0].keys() if not k.endswith("_pval")]
            all_methods.extend(pm_cols)
        if result.get("answer_copying_pairs"):
            all_methods.extend(["OMG_S", "GBT_S"])
        if result.get("preknowledge"):
            all_methods.extend(["L_S", "S_S", "W_S"])
        if result.get("rapid_guessing"):
            all_methods.extend(["RG_NT"])
        result["methods"] = list(dict.fromkeys(all_methods))  # preserve order, no duplicates
        if run_tt:
            if state.get("answer_changes"):
                result["info"] = (result.get("info") or "") + (
                    " " if result.get("info") else ""
                ) + "Test Tampering is computed by the dedicated tt_agent from the loaded initial/final answer-change data."
            else:
                result["info"] = (result.get("info") or "") + (
                    " " if result.get("info") else ""
                ) + "Test Tampering requires long initial/final answer-change data."
        elif run_rg and not result.get("rapid_guessing"):
            has_rt = (state.get("rt_data") or []) and len(state.get("rt_data") or []) == (resp_df[keep_cols].shape[0] if keep_cols else 0)
            if not has_rt:
                result["info"] = (result.get("info") or "") + (" " if result.get("info") else "") + "Rapid Guessing (detect_rg) requires response-time data; upload RT data and run again."
            else:
                err_detail = result.get("rapid_guessing_error") or ""
                msg = "Rapid Guessing (detect_rg) could not be completed."
                if err_detail:
                    msg += f" R reported: {err_detail}"
                else:
                    msg += " Check that RT values are numeric and column count/layout matches responses."
                result["info"] = (result.get("info") or "") + (" " if result.get("info") else "") + msg
        return {
            "aberrance_results": result,
            "next_step": "end",
        }
    except Exception as exc:
        return {
            "aberrance_results": {"error": f"aberrance failed: {type(exc).__name__}: {exc}"},
            "next_step": "end",
        }


# ──────────────────────────────────────────────────────────────────────────────
# FORENSIC SPECIALIST NODES  (each reads State, calls one R function, writes flags)
# ──────────────────────────────────────────────────────────────────────────────

def _forensic_keep_cols(resp_df: pd.DataFrame) -> list[str]:
    """Return list of valid dichotomous (0/1) columns from a response DataFrame."""
    keep = []
    for col in resp_df.columns:
        vals = pd.to_numeric(resp_df[col], errors="coerce").dropna().unique().tolist()
        if not vals or not set(vals).issubset({0, 1}) or len(set(vals)) < 2:
            continue
        keep.append(col)
    return keep


def _forensic_init_r(resp_df: pd.DataFrame, keep_cols: list[str]):
    """Import rpy2, push x matrix into R (pure Python list only — no numpy). return (ro, pandas2ri, np)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    import numpy as np
    try:
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except Exception:
        pass
    ro.r("library(aberrance)")
    # Build x in R from pure Python list only (no numpy types) to avoid py2rpy numpy.ndarray
    block = resp_df[keep_cols]
    nrow, ncol = int(block.shape[0]), int(block.shape[1])
    x_flat = []
    for i in range(nrow):
        for j in range(ncol):
            v = block.iloc[i, j]
            x_flat.append(int(v) if v is not None and str(v) not in ("nan", "NaN", "") else 0)
    ro.globalenv["x_vec"] = ro.IntVector(x_flat)
    ro.r(f"x <- matrix(x_vec, nrow={nrow}, ncol={ncol}, byrow=TRUE)")
    return ro, pandas2ri, np


def _forensic_build_psi(ro, ip_df: pd.DataFrame, n_items: int):
    """Push psi (a, b, c) matrix into R. Returns True on success."""
    a_col = "a1" if "a1" in ip_df.columns else "a"
    b_col = "b"
    c_col = "g" if "g" in ip_df.columns else ("c" if "c" in ip_df.columns else None)
    if a_col not in ip_df.columns or b_col not in ip_df.columns:
        return False
    a_vals = [float(v) for v in ip_df[a_col].tolist()]
    b_vals = [float(v) for v in ip_df[b_col].tolist()]
    c_vals = [float(v) for v in ip_df[c_col].tolist()] if c_col and c_col in ip_df.columns else [0.0] * len(a_vals)
    ro.globalenv["psi_a"] = ro.FloatVector(a_vals)
    ro.globalenv["psi_b"] = ro.FloatVector(b_vals)
    ro.globalenv["psi_c"] = ro.FloatVector(c_vals)
    ro.r("psi <- as.matrix(cbind(a = psi_a, b = psi_b, c = psi_c))")
    nrow = int(ro.r("nrow(psi)")[0])
    return nrow == n_items


# ---------- 1. nm_agent (Nonparametric Fit) -----------------------------------
def nm_agent(state: State) -> dict:
    """detect_nm: full set of nonparametric person-fit indices (G_S, NC_S, U1_S, U3_S, ZU3_S, A_S, D_S, E_S, C_S, MC_S, PC_S, HT_S)."""
    print("--- FORENSIC nm_agent: detect_nm ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_nm" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    if not keep_cols:
        return {"flags": {"nm_agent": {"error": "No valid dichotomous columns."}}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        with (ro.default_converter + pandas2ri.converter).context():
            ro.r("""
                assign('nm_err', NULL, envir = .GlobalEnv)
                nm_out <- tryCatch(
                    detect_nm(
                        method = c('G_S','NC_S','U1_S','U3_S','ZU3_S','A_S','D_S','E_S','C_S','MC_S','PC_S','HT_S'),
                        x = x
                    ),
                    error = function(e) { assign('nm_err', conditionMessage(e), envir = .GlobalEnv); NULL })
            """)
            has_nm = ro.r("!is.null(nm_out)")
            if not (has_nm and bool(list(ro.conversion.rpy2py(has_nm))[0])):
                err = str(ro.r("get0('nm_err', envir = .GlobalEnv, ifnotfound = 'unknown')"))
                return {"flags": {"nm_agent": {"error": f"detect_nm failed: {err}"}}}
            # Extract columns safely via R to avoid rpy2py conversion issues
            ro.r("""
                nm_stat_mat <- nm_out$stat
                nm_n <- as.integer(nrow(nm_stat_mat))
                nm_methods <- colnames(nm_stat_mat)
                nm_n_methods <- as.integer(length(nm_methods))
                nm_methods_str <- paste(nm_methods, collapse = '|')
            """)
            nm_n = int(ro.r("nm_n")[0])
            nm_methods_str = str(ro.r("nm_methods_str")[0])
            nm_methods_list = nm_methods_str.split("|") if nm_methods_str else ["ZU3_S", "HT_S"]
            nm_n_methods = int(ro.r("nm_n_methods")[0])
            # Read each column as a plain numeric vector
            nm_data = {}
            for mi in range(nm_n_methods):
                mname = nm_methods_list[mi]
                col_r = ro.r(f"as.numeric(nm_stat_mat[, {mi + 1}])")
                nm_data[mname] = [float(v) for v in col_r]
            nm_df = pd.DataFrame(nm_data)
            records = nm_df.to_dict(orient="records")
            methods = nm_methods_list
            # Flag extraction
            flagged = []
            if "ZU3_S" in nm_df.columns:
                zu3 = pd.to_numeric(nm_df["ZU3_S"], errors="coerce")
                if _threshold_enabled(state, "detect_nm", "ZU3_S", True):
                    zu3_cut = _threshold_float(state, "detect_nm", "ZU3_S", "value", -2.0)
                    flagged.extend(np.where(zu3 < zu3_cut)[0].tolist())
            if "HT_S" in nm_df.columns:
                ht = pd.to_numeric(nm_df["HT_S"], errors="coerce")
                if ht.notna().any() and _threshold_enabled(state, "detect_nm", "HT_S", True):
                    ht_q = _threshold_float(state, "detect_nm", "HT_S", "quantile", 0.05)
                    q05 = ht.quantile(ht_q)
                    if pd.notna(q05):
                        flagged.extend(np.where(ht <= q05)[0].tolist())
            return {"flags": {"nm_agent": {
                "stat": records, "methods": methods,
                "flagged": sorted(set(flagged)), "n_persons": nm_n,
            }}}
    except Exception as e:
        return {"flags": {"nm_agent": {"error": str(e)}}}


# ---------- 2. pm_agent (Parametric Fit) --------------------------------------
def pm_agent(state: State) -> dict:
    """detect_pm: run all supported score/response/time person-fit method families."""
    print("--- FORENSIC pm_agent: detect_pm ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_pm" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    if not keep_cols or not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"pm_agent": {"error": "Missing valid response columns or item parameters."}}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        ip_df = pd.DataFrame(psi_src)
        with (ro.default_converter + pandas2ri.converter).context():
            if not _forensic_build_psi(ro, ip_df, len(keep_cols)):
                return {"flags": {"pm_agent": {"error": "Could not build psi matrix (a/b columns missing or dimension mismatch)."}}}
            n_persons = resp_df[keep_cols].shape[0]
            rt_data = state.get("rt_data") or []
            has_rt = bool(rt_data) and len(rt_data) == n_persons
            if has_rt:
                pm_rt_df = pd.DataFrame(rt_data)
                pm_rt_block = pm_rt_df.iloc[:, :len(keep_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.01)
                y_flat = np.log(pm_rt_block.astype(float).clip(lower=0.001).values.flatten()).tolist()
                ro.globalenv["y_vec"] = ro.FloatVector(y_flat)
                ro.r(f"y <- matrix(y_vec, nrow={n_persons}, ncol={len(keep_cols)}, byrow=TRUE)")
                ro.r("""
                    pm_beta_est  <- apply(y, 2, mean)
                    pm_alpha_est <- 1 / apply(y, 2, sd)
                    pm_alpha_est[!is.finite(pm_alpha_est)] <- 1.0
                    psi <- cbind(psi, alpha = pm_alpha_est, beta = pm_beta_est)
                """)

            ro.r("r <- x")
            pm_data: dict[str, list] = {}
            flagged_pm: set[int] = set()
            methods_run: list[str] = []
            method_errors: list[str] = []

            def _run_pm_group(methods: list[str], call_args: str) -> None:
                nonlocal pm_data, flagged_pm, methods_run, method_errors
                ro.globalenv["pm_methods_req"] = ro.StrVector(methods)
                alpha_pm = _threshold_alpha(state, "detect_pm")
                ro.r(f"""
                    assign('pm_err', NULL, envir = .GlobalEnv)
                    pm_out <- tryCatch(
                        detect_pm(method = pm_methods_req, psi = psi, {call_args}, alpha = {_r_num(alpha_pm)}),
                        error = function(e) {{ assign('pm_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
                """)
                if not bool(ro.r("!is.null(pm_out)")[0]):
                    err = "no output"
                    try:
                        if not bool(ro.r("is.null(get0('pm_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                            err = str(ro.r("as.character(pm_err)")[0])
                    except Exception:
                        pass
                    method_errors.append(f"{', '.join(methods)}: {err}")
                    return
                ro.r("""
                    pm_stat_mat <- pm_out$stat
                    pm_pval_mat <- pm_out$pval
                    pm_flag_arr <- pm_out$flag
                    pm_n <- as.integer(nrow(pm_stat_mat))
                    pm_methods <- colnames(pm_stat_mat)
                    pm_n_methods <- as.integer(length(pm_methods))
                    pm_methods_str <- paste(pm_methods, collapse='|')
                """)
                pm_n_methods = int(ro.r("pm_n_methods")[0])
                group_methods = str(ro.r("pm_methods_str")[0]).split("|") if pm_n_methods else []
                pm_flag_np = None
                try:
                    pm_flag_np = np.asarray(ro.conversion.rpy2py(ro.r("pm_flag_arr")))
                except Exception:
                    pm_flag_np = None
                for mi, mname in enumerate(group_methods):
                    stat_r = ro.r(f"as.numeric(pm_stat_mat[, {mi + 1}])")
                    pm_data[mname] = [float(v) for v in stat_r]
                    pval_r = ro.r(f"as.numeric(pm_pval_mat[, {mi + 1}])")
                    pm_data[f"{mname}_pval"] = [float(v) for v in pval_r]
                    if pm_flag_np is not None and pm_flag_np.ndim >= 2 and mi < pm_flag_np.shape[1]:
                        method_flags = pm_flag_np[:, mi]
                        if method_flags.ndim >= 2:
                            method_flags = np.any(method_flags, axis=tuple(range(1, method_flags.ndim)))
                        pm_data[f"{mname}_flag"] = [1 if bool(v) else 0 for v in method_flags.tolist()]
                try:
                    flag_any_r = ro.r("as.logical(apply(pm_flag_arr, 1, any))")
                    flagged_pm.update(i for i, v in enumerate(flag_any_r) if v)
                except Exception:
                    pass
                methods_run.extend(group_methods)

            _run_pm_group(["ECI2_S_*", "ECI4_S_*", "L_S_*"], "x = x")
            _run_pm_group(["L_R_*"], "r = r")
            if has_rt:
                _run_pm_group(["L_T"], "y = y")
                _run_pm_group(["Q_ST_*", "L_ST_*"], "x = x, y = y")
                _run_pm_group(["Q_RT_*", "L_RT_*"], "r = r, y = y")
            if not methods_run:
                return {"flags": {"pm_agent": {"error": "detect_pm failed for all method groups: " + "; ".join(method_errors)}}}
            pm_df_out = pd.DataFrame(pm_data)
            pm_n = len(pm_df_out)
            return {"flags": {"pm_agent": {
                "stat": pm_df_out.to_dict(orient="records"), "methods": list(dict.fromkeys(methods_run)),
                "flagged": sorted(flagged_pm), "n_persons": pm_n, "method_errors": method_errors,
            }}}
    except Exception as e:
        return {"flags": {"pm_agent": {"error": str(e)}}}


# ---------- 3. ac_agent (Answer Copying) --------------------------------------
def ac_agent(state: State) -> dict:
    """detect_ac: all supported score- and response-based answer-copying methods."""
    print("--- FORENSIC ac_agent: detect_ac ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_ac" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    if not keep_cols or not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"ac_agent": {"error": "Missing response columns or item parameters."}}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        ip_df = pd.DataFrame(psi_src)
        with (ro.default_converter + pandas2ri.converter).context():
            if not _forensic_build_psi(ro, ip_df, len(keep_cols)):
                return {"flags": {"ac_agent": {"error": "Could not build psi matrix."}}}
            ro.r("r <- x")
            N = len(resp_df)
            pairs_0based = [(i, j) for i in range(N) for j in range(i + 1, N)]
            pair_records: dict[int, dict] = {}
            flagged_copiers = set()
            methods_run: list[str] = []
            method_errors: list[str] = []
            alpha_ac = _pairwise_alpha(state, "detect_ac", len(pairs_0based))

            def _run_ac_group(methods: list[str], call_args: str) -> None:
                nonlocal pair_records, flagged_copiers, methods_run, method_errors
                ro.globalenv["ac_methods_req"] = ro.StrVector(methods)
                ro.r(f"""
                    assign('ac_err', NULL, envir = .GlobalEnv)
                    ac_out <- tryCatch(
                        detect_ac(method = ac_methods_req, psi = psi, {call_args}, alpha = {_r_num(alpha_ac)}),
                        error = function(e) {{ assign('ac_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
                """)
                if not bool(ro.r("!is.null(ac_out)")[0]):
                    err = "no output"
                    try:
                        if not bool(ro.r("is.null(get0('ac_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                            err = str(ro.r("as.character(ac_err)")[0])
                    except Exception:
                        pass
                    method_errors.append(f"{', '.join(methods)}: {err}")
                    return
                ac_stat_py = ro.conversion.rpy2py(ro.r("as.data.frame(ac_out$stat)"))
                ac_records = ac_stat_py.to_dict(orient="records") if hasattr(ac_stat_py, "to_dict") else []
                ro.r("ac_pval_df <- tryCatch(as.data.frame(ac_out$pval), error = function(e) NULL)")
                pval_records = []
                try:
                    if bool(ro.r("!is.null(ac_pval_df)")[0]):
                        ac_pval_py = ro.conversion.rpy2py(ro.r("ac_pval_df"))
                        if hasattr(ac_pval_py, "to_dict"):
                            pval_records = ac_pval_py.to_dict(orient="records")
                except Exception:
                    pass
                ac_flag_py = None
                try:
                    ac_flag_py = ro.conversion.rpy2py(ro.r("ac_out$flag"))
                except Exception:
                    pass
                ac_method_names = list(ac_records[0].keys()) if ac_records else []
                ac_flag_arr = None
                if ac_flag_py is not None:
                    try:
                        ac_flag_arr = _orient_pair_flag_matrix(
                            np.asarray(ac_flag_py, dtype=bool),
                            len(ac_records),
                            len(ac_method_names),
                        )
                    except Exception:
                        ac_flag_arr = None
                for idx, (i, j) in enumerate(pairs_0based[:len(ac_records)]):
                    row = pair_records.setdefault(idx, {"Source": i + 1, "Copier": j + 1})
                    row.update(ac_records[idx])
                    if idx < len(pval_records):
                        for pk, pv in pval_records[idx].items():
                            pkey = pk if pk.endswith("_pval") else f"{pk}_pval"
                            row[pkey] = pv
                    if ac_flag_arr is not None:
                        fl = ac_flag_arr
                        if fl.ndim >= 2 and idx < fl.shape[0]:
                            for mi, mname in enumerate(ac_method_names):
                                if mi >= fl.shape[1]:
                                    continue
                                method_flag = fl[idx, mi]
                                if np.ndim(method_flag) >= 1:
                                    method_flag = np.any(method_flag)
                                row[f"{mname}_flag"] = 1 if bool(method_flag) else 0
                    min_p = 1.0
                    for pk, pv in row.items():
                        if pk.endswith("_pval") and isinstance(pv, (int, float)):
                            min_p = min(min_p, pv)
                    if min_p < alpha_ac:
                        row["flagged"] = True
                        flagged_copiers.add(j)
                methods_run.extend(methods)

            _run_ac_group(["OMG_S", "GBT_S"], "x = x")
            _run_ac_group(["OMG_R", "GBT_R"], "r = r")
            if not methods_run:
                return {"flags": {"ac_agent": {"error": "detect_ac failed for all method groups: " + "; ".join(method_errors)}}}

            pair_rows = []
            flagged_pair_set: set[tuple[int, int]] = set()
            # Keep pairs with p < 0.10 (generous storage cutoff; UI slider does final filtering)
            _STORAGE_P_CUTOFF = 0.10
            for row in pair_records.values():
                row["flagged"] = bool(row.get("flagged", False))
                if row["flagged"]:
                    try:
                        flagged_pair_set.add((int(row["Source"]) - 1, int(row["Copier"]) - 1))
                    except (KeyError, TypeError, ValueError):
                        pass
                # Keep pair if flagged OR if any p-value is below storage cutoff
                min_p = 1.0
                for pk, pv in row.items():
                    if pk.endswith("_pval") and isinstance(pv, (int, float)):
                        min_p = min(min_p, pv)
                if row["flagged"] or min_p < _STORAGE_P_CUTOFF:
                    pair_rows.append(row)
            return {"flags": {"ac_agent": {
                "pairs": pair_rows, "flagged_copiers": sorted(flagged_copiers),
                "flagged_pairs": [list(pair) for pair in sorted(flagged_pair_set)],
                "methods": list(dict.fromkeys(methods_run)), "method_errors": method_errors,
            }}}
    except Exception as e:
        return {"flags": {"ac_agent": {"error": str(e)}}}


# ---------- 4. as_agent (Answer Similarity / Clusters) ------------------------
def as_agent(state: State) -> dict:
    """detect_as: answer-similarity detection across available score/raw/time methods."""
    print("--- FORENSIC as_agent: detect_as ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_as" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    if not keep_cols:
        return {"flags": {"as_agent": {"error": "No valid dichotomous columns."}}}
    if not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"as_agent": {"error": "Missing item parameters for detect_as."}}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        ip_df = pd.DataFrame(psi_src)
        with (ro.default_converter + pandas2ri.converter).context():
            if not _forensic_build_psi(ro, ip_df, len(keep_cols)):
                return {"flags": {"as_agent": {"error": "Could not build psi matrix."}}}
            ro.r("r <- x")

            rt_data = state.get("rt_data") or []
            has_rt = bool(rt_data) and len(rt_data) == len(resp_df)
            if has_rt:
                rt_df = pd.DataFrame(rt_data)
                if rt_df.shape[1] >= len(keep_cols):
                    t_block = rt_df.iloc[:, :len(keep_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.01)
                    y_flat = np.log(t_block.astype(float).clip(lower=0.001).values.flatten()).tolist()
                    ro.globalenv["as_y_vec"] = ro.FloatVector(y_flat)
                    ro.r(f"y <- matrix(as_y_vec, nrow={int(t_block.shape[0])}, ncol={int(t_block.shape[1])}, byrow=TRUE)")
                    ro.r("""
                        as_beta_est  <- apply(y, 2, mean)
                        as_alpha_est <- 1 / apply(y, 2, sd)
                        as_alpha_est[!is.finite(as_alpha_est)] <- 1.0
                        psi <- cbind(psi, alpha = as_alpha_est, beta = as_beta_est)
                    """)
                else:
                    has_rt = False

            N = len(resp_df)
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            records_by_idx: dict[int, dict] = {}
            flagged = []
            flagged_pair_set: set[tuple[int, int]] = set()
            methods_run: list[str] = []
            method_errors: list[str] = []
            _STORAGE_P_CUTOFF = 0.10
            alpha_as = _pairwise_alpha(state, "detect_as", len(pairs))

            def _run_as_group(methods: list[str], call_args: str) -> None:
                nonlocal flagged, methods_run, method_errors
                ro.globalenv["as_methods"] = ro.StrVector(methods)
                ro.r(f"""
                    assign('as_err', NULL, envir = .GlobalEnv)
                    as_out <- tryCatch(
                        detect_as(method = as_methods, psi = psi, {call_args}, alpha = {_r_num(alpha_as)}),
                        error = function(e) {{ assign('as_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
                """)
                has_as = ro.r("!is.null(as_out)")
                if not (has_as and ro.conversion.rpy2py(has_as)):
                    err = ""
                    try:
                        if not bool(ro.r("is.null(get0('as_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                            err = str(ro.r("as.character(as_err)")[0])
                    except Exception:
                        err = "unknown"
                    method_errors.append(f"{', '.join(methods)}: {err or 'no output'}")
                    return

                as_stat = ro.r("as.data.frame(as_out$stat)")
                as_stat_py = ro.conversion.rpy2py(as_stat)
                stat_records = as_stat_py.to_dict(orient="records") if hasattr(as_stat_py, "to_dict") else []

                ro.r("as_pval_df <- tryCatch(as.data.frame(as_out$pval), error = function(e) NULL)")
                pval_records = []
                try:
                    has_pval = bool(ro.r("!is.null(as_pval_df)")[0])
                    if has_pval:
                        as_pval_py = ro.conversion.rpy2py(ro.r("as_pval_df"))
                        if hasattr(as_pval_py, "to_dict"):
                            pval_records = as_pval_py.to_dict(orient="records")
                except Exception:
                    pass

                flag_arr = None
                as_method_names = list(stat_records[0].keys()) if stat_records else []
                try:
                    fl_py = ro.conversion.rpy2py(ro.r("as_out$flag"))
                    flag_arr = _orient_pair_flag_matrix(
                        np.asarray(fl_py, dtype=bool),
                        min(len(stat_records), len(pairs)),
                        len(as_method_names),
                    )
                except Exception:
                    pass

                for idx in range(min(len(stat_records), len(pairs))):
                    rec = records_by_idx.setdefault(idx, {"_pair": pairs[idx]})
                    for k, v in stat_records[idx].items():
                        rec[k] = v
                    if idx < len(pval_records):
                        for pk, pv in pval_records[idx].items():
                            pkey = pk if pk.endswith("_pval") else f"{pk}_pval"
                            rec[pkey] = pv
                    is_flagged = False
                    if flag_arr is not None:
                        if flag_arr.ndim >= 2 and idx < flag_arr.shape[0]:
                            for mi, mname in enumerate(as_method_names):
                                if mi >= flag_arr.shape[1]:
                                    continue
                                method_flag = flag_arr[idx, mi]
                                if np.ndim(method_flag) >= 1:
                                    method_flag = np.any(method_flag)
                                rec[f"{mname}_flag"] = 1 if bool(method_flag) else 0
                        if flag_arr.ndim == 1 and idx < len(flag_arr) and flag_arr[idx]:
                            is_flagged = True
                        elif flag_arr.ndim >= 2 and idx < flag_arr.shape[0] and np.any(flag_arr[idx]):
                            is_flagged = True
                    min_p = 1.0
                    for pk, pv in rec.items():
                        if pk.endswith("_pval") and isinstance(pv, (int, float)):
                            min_p = min(min_p, pv)
                    if is_flagged and min_p < alpha_as:
                        flagged.extend(pairs[idx])
                        flagged_pair_set.add(pairs[idx])
                methods_run.extend(methods)

            _run_as_group(["OMG_S", "WOMG_S", "GBT_S", "M4_S"], "x = x")
            _run_as_group(["OMG_R", "WOMG_R", "GBT_R", "M4_R"], "r = r")
            if has_rt:
                _run_as_group(["OMG_ST", "GBT_ST"], "x = x, y = y")
                _run_as_group(["OMG_RT", "GBT_RT"], "r = r, y = y")

            flagged_set = set(flagged)
            records = []
            for idx, rec in records_by_idx.items():
                min_p = 1.0
                for pk, pv in rec.items():
                    if pk.endswith("_pval") and isinstance(pv, (int, float)):
                        min_p = min(min_p, pv)
                pair = pairs[idx]
                if pair[0] in flagged_set or pair[1] in flagged_set or min_p < _STORAGE_P_CUTOFF:
                    records.append(rec)
            if not methods_run:
                return {"flags": {"as_agent": {"error": "detect_as failed for all method groups: " + "; ".join(method_errors)}}}
            return {"flags": {"as_agent": {
                "stat": records,
                "methods": list(dict.fromkeys(methods_run)),
                "flagged_pairs": [list(pair) for pair in sorted(flagged_pair_set)],
                "flagged_participants": sorted(set(flagged)),
                "method_errors": method_errors,
            }}}
    except Exception as e:
        return {"flags": {"as_agent": {"error": str(e)}}}


# ---------- 5. rg_agent (Rapid Guessing) --------------------------------------
def rg_agent(state: State) -> dict:
    """detect_rg: all supported threshold/inspection rapid-guessing methods."""
    print("--- FORENSIC rg_agent: detect_rg ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_rg" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    rt_data = state.get("rt_data") or []
    n_persons = resp_df[keep_cols].shape[0] if keep_cols else 0
    n_items = len(keep_cols)
    if not keep_cols or not rt_data or len(rt_data) != n_persons:
        return {"flags": {"rg_agent": {"error": "Requires response-time data matching response rows."}}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        rt_df = pd.DataFrame(rt_data)
        if rt_df.shape[1] < n_items:
            return {"flags": {"rg_agent": {"error": f"RT has {rt_df.shape[1]} cols, need {n_items}."}}}
        t_block = rt_df.iloc[:, :n_items].apply(pd.to_numeric, errors="coerce").fillna(0.01).astype(np.float64)
        with (ro.default_converter + pandas2ri.converter).context():
            flat = t_block.values.flatten().tolist()
            ro.globalenv["t_vec"] = ro.FloatVector(flat)
            ro.r(f"t <- matrix(t_vec, nrow={n_persons}, ncol={n_items}, byrow=TRUE)")
            methods_run: list[str] = []
            method_errors: list[str] = []
            rte_cols: dict[str, list] = {}
            flagged_set: set[int] = set()
            flagged_by_method: dict[str, list[int]] = {}

            def _run_rg(method: str, args: str) -> None:
                nonlocal methods_run, method_errors, rte_cols, flagged_set, flagged_by_method
                ro.globalenv["rg_method_req"] = ro.StrVector([method])
                ro.r(f"""
                    assign('rg_err', NULL, envir = .GlobalEnv)
                    rg_out <- tryCatch(
                        detect_rg(method = rg_method_req, t = t, {args}),
                        error = function(e) {{ assign('rg_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
                """)
                if not bool(ro.r("!is.null(rg_out)")[0]):
                    err = "no output"
                    try:
                        if not bool(ro.r("is.null(get0('rg_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                            err = str(ro.r("as.character(rg_err)")[0])
                    except Exception:
                        pass
                    method_errors.append(f"{method}: {err}")
                    return
                methods_run.append(method)
                try:
                    if bool(ro.r("'rte' %in% names(rg_out)")[0]):
                        rte_py = ro.conversion.rpy2py(ro.r("rg_out$rte"))
                        arr = np.asarray(rte_py)
                        if arr.ndim == 1:
                            rte_cols[method] = arr.astype(float).tolist()
                        elif arr.ndim >= 2:
                            for j in range(arr.shape[1]):
                                rte_cols[f"{method}_{j+1}"] = arr[:, j].astype(float).tolist()
                except Exception:
                    pass
                try:
                    if bool(ro.r("'flag' %in% names(rg_out)")[0]):
                        fl_py = ro.conversion.rpy2py(ro.r("rg_out$flag"))
                        arr = np.asarray(fl_py)
                        if arr.ndim == 1 and len(arr) == n_persons:
                            method_flagged = set(np.where(arr)[0].tolist())
                        elif arr.ndim >= 2 and arr.shape[0] == n_persons:
                            method_flagged = set(np.where(np.any(arr, axis=1))[0].tolist())
                        else:
                            method_flagged = set()
                        flagged_by_method[method] = sorted(method_flagged)
                        flagged_set.update(method_flagged)
                except Exception:
                    pass

            if _threshold_enabled(state, "detect_rg", "CT", True):
                ct_thr = _threshold_float(state, "detect_rg", "CT", "thr", 3.0)
                _run_rg("CT", f"thr = {_r_num(ct_thr)}")
            if _threshold_enabled(state, "detect_rg", "NT", True):
                nt_values = _threshold_block(state, "detect_rg", "NT").get("nt", [5, 10, 15, 20, 25, 30, 35])
                if not isinstance(nt_values, list) or not nt_values:
                    nt_values = [5, 10, 15, 20, 25, 30, 35]
                nt_expr = ",".join(_r_num(float(v)) for v in nt_values)
                _run_rg("NT", f"nt = c({nt_expr})")
            if _threshold_enabled(state, "detect_rg", "CUMP", True):
                cump_outlier = _threshold_float(state, "detect_rg", "CUMP", "outlier", 90.0)
                _run_rg("CUMP", f"x = x, outlier = {_r_num(cump_outlier)}")
            if _threshold_enabled(state, "detect_rg", "VI", True):
                vi_outlier = _threshold_float(state, "detect_rg", "VI", "outlier", 90.0)
                _run_rg("VI", f"outlier = {_r_num(vi_outlier)}")
            if _threshold_enabled(state, "detect_rg", "VITP", True):
                vitp_outlier = _threshold_float(state, "detect_rg", "VITP", "outlier", 90.0)
                _run_rg("VITP", f"x = x, outlier = {_r_num(vitp_outlier)}")
            if not methods_run:
                return {"flags": {"rg_agent": {"error": "detect_rg failed for all methods: " + "; ".join(method_errors)}}}
            rte_vals = rte_cols.get("NT_2") or rte_cols.get("NT") or (next(iter(rte_cols.values())) if rte_cols else [])
            rg_cfg = _threshold_rules(state).get("detect_rg") or {}
            flag_methods = rg_cfg.get("flag_methods", ["NT"]) if isinstance(rg_cfg, dict) else ["NT"]
            if isinstance(flag_methods, str):
                flag_methods = [flag_methods]
            if not isinstance(flag_methods, list) or not flag_methods:
                flag_methods = ["NT"]
            final_flagged_set: set[int] = set()
            for method in flag_methods:
                final_flagged_set.update(flagged_by_method.get(str(method), []))
            flagged = sorted(final_flagged_set)
            return {"flags": {"rg_agent": {
                "rte": rte_vals, "rte_by_method": rte_cols, "flagged": flagged,
                "flag_methods": [str(method) for method in flag_methods],
                "flagged_by_method": flagged_by_method,
                "flagged_any_method": sorted(flagged_set),
                "methods": methods_run, "method_errors": method_errors,
            }}}
    except Exception as e:
        return {"flags": {"rg_agent": {"error": str(e)}}}


# ---------- 6. cp_agent (Change Point) ----------------------------------------
def cp_agent(state: State) -> dict:
    """detect_cp: all supported score/time change-point method families."""
    print("--- FORENSIC cp_agent: detect_cp ---")
    selected = state.get("aberrance_functions") or []
    # Only run CP when rapid-guessing / low-effort analysis is selected.
    if selected and "detect_rg" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    if not keep_cols:
        return {"flags": {"cp_agent": {"error": "No valid dichotomous columns."}}}
    if not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"cp_agent": {
            "info": "Change-point method families in aberrance require item parameters (psi); cp_agent skipped.",
            "flagged": [],
            "methods": [],
        }}}
    try:
        import numpy as np
        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        ip_df = pd.DataFrame(psi_src)
        with (ro.default_converter + pandas2ri.converter).context():
            if not _forensic_build_psi(ro, ip_df, len(keep_cols)):
                return {"flags": {"cp_agent": {"error": "Could not build psi matrix."}}}
            n_persons, n_items = resp_df[keep_cols].shape
            rt_data = state.get("rt_data") or []
            has_rt = bool(rt_data) and len(rt_data) == n_persons
            if has_rt:
                rt_df = pd.DataFrame(rt_data)
                if rt_df.shape[1] >= n_items:
                    y_block = rt_df.iloc[:, :n_items].apply(pd.to_numeric, errors="coerce").fillna(0.01)
                    y_flat = np.log(y_block.astype(float).clip(lower=0.001).values.flatten()).tolist()
                    ro.globalenv["cp_y_vec"] = ro.FloatVector(y_flat)
                    ro.r(f"y <- matrix(cp_y_vec, nrow={n_persons}, ncol={n_items}, byrow=TRUE)")
                    ro.r("""
                        cp_beta_est  <- apply(y, 2, mean)
                        cp_alpha_est <- 1 / apply(y, 2, sd)
                        cp_alpha_est[!is.finite(cp_alpha_est)] <- 1.0
                        psi <- cbind(psi, alpha = cp_alpha_est, beta = cp_beta_est)
                    """)
                else:
                    has_rt = False
            ro.globalenv["cp_methods_req"] = ro.StrVector(
                ["L_S_*", "S_S_*", "W_S_*"] + (["L_T_*", "W_T_*"] if has_rt else [])
            )
            call_args = "x = x, y = y" if has_rt else "x = x"
            ro.r(f"""
                assign('cp_err', NULL, envir = .GlobalEnv)
                cp_out <- tryCatch(
                    detect_cp(method = cp_methods_req, cpi = c(1, {max(1, n_items - 1)}), psi = psi, {call_args}),
                    error = function(e) {{ assign('cp_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
            """)
            if not bool(ro.r("!is.null(cp_out)")[0]):
                err = "no output"
                try:
                    if not bool(ro.r("is.null(get0('cp_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                        err = str(ro.r("as.character(cp_err)")[0])
                except Exception:
                    pass
                return {"flags": {"cp_agent": {"error": f"detect_cp failed: {err}"}}}
            cp_py = ro.conversion.rpy2py(ro.r("as.data.frame(cp_out$stat)"))
            cp_stat_records = cp_py.to_dict(orient="records") if hasattr(cp_py, "to_dict") else []
            try:
                cp_est_py = ro.conversion.rpy2py(ro.r("as.data.frame(cp_out$cp)"))
                cp_est_records = cp_est_py.to_dict(orient="records") if hasattr(cp_est_py, "to_dict") else []
                for i, rec in enumerate(cp_stat_records):
                    if i < len(cp_est_records):
                        for k, v in cp_est_records[i].items():
                            rec[f"{k}_cp"] = v
            except Exception:
                pass
            methods = list(cp_stat_records[0].keys()) if cp_stat_records else list(ro.conversion.rpy2py(ro.r("cp_methods_req")))
            return {"flags": {"cp_agent": {
                "stat": cp_stat_records, "flagged": [], "methods": methods,
            }}}
    except Exception as e:
        return {"flags": {"cp_agent": {"error": str(e)}}}


# ---------- 7. tt_agent (Test Tampering) ---------------------------------------
def tt_agent(state: State) -> dict:
    """Run score/distractor-based aberrance::detect_tt from long answer changes."""
    print("--- FORENSIC tt_agent: detect_tt (score-change mode) ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_tt" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state.get("responses") or [])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    changes = pd.DataFrame(state.get("answer_changes") or [])
    if not keep_cols or not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"tt_agent": {"error": "Missing valid response columns or item parameters."}}}
    if changes.empty:
        return {"flags": {"tt_agent": {
            "error": "detect_tt requires long answer-change data.",
            "flagged": [],
            "methods": [],
        }}}

    def _find_col(candidates: list[str]) -> str | None:
        by_lower = {str(col).strip().lower(): col for col in changes.columns}
        for candidate in candidates:
            if candidate in by_lower:
                return by_lower[candidate]
        return None

    person_col = _find_col(["examinee_id", "person_id", "id"])
    item_col = _find_col(["item_id", "item", "item_position"])
    initial_col = _find_col(["initial_response", "initial_score", "x_0"])
    final_col = _find_col(["final_response", "final_score", "x"])
    if not person_col or not item_col or not initial_col or not final_col:
        return {"flags": {"tt_agent": {
            "error": (
                "Answer-change data must include examinee_id, item_id, and initial/final "
                "response columns (initial_response/final_response or initial_score/final_score)."
            ),
            "flagged": [],
            "methods": [],
        }}}

    try:
        import numpy as np

        work = changes[[person_col, item_col, initial_col, final_col]].copy()
        work[person_col] = work[person_col].astype(str)
        work[item_col] = work[item_col].astype(str)
        work[initial_col] = pd.to_numeric(work[initial_col], errors="coerce")
        work[final_col] = pd.to_numeric(work[final_col], errors="coerce")
        if work[[initial_col, final_col]].isna().any().any():
            raise ValueError("Initial/final answer-change values contain missing or nonnumeric data.")
        if not set(work[initial_col].astype(int).unique()).issubset({0, 1}):
            raise ValueError("Initial answer-change values must be dichotomous 0/1 scores.")
        if not set(work[final_col].astype(int).unique()).issubset({0, 1}):
            raise ValueError("Final answer-change values must be dichotomous 0/1 scores.")

        person_order = sorted(work[person_col].unique().tolist())
        item_values = work[item_col].unique().tolist()
        item_by_text = {str(item): str(item) for item in item_values}
        if set(map(str, keep_cols)).issubset(item_by_text):
            item_order = [str(col) for col in keep_cols]
        else:
            def _item_sort_key(value: str):
                digits = "".join(ch for ch in str(value) if ch.isdigit())
                return (int(digits) if digits else 10**9, str(value))
            item_order = sorted(map(str, item_values), key=_item_sort_key)

        initial_matrix = (
            work.pivot(index=person_col, columns=item_col, values=initial_col)
            .reindex(index=person_order, columns=item_order)
        )
        final_matrix = (
            work.pivot(index=person_col, columns=item_col, values=final_col)
            .reindex(index=person_order, columns=item_order)
        )
        n_persons = int(resp_df.shape[0])
        n_items = len(keep_cols)
        if initial_matrix.shape != (n_persons, n_items):
            raise ValueError(
                f"Answer-change data resolve to {initial_matrix.shape[0]} examinees x "
                f"{initial_matrix.shape[1]} items; expected {n_persons} x {n_items}."
            )
        if initial_matrix.isna().any().any() or final_matrix.isna().any().any():
            raise ValueError("Answer-change data do not contain one complete row per examinee and item.")

        operational_final = resp_df[keep_cols].apply(pd.to_numeric, errors="coerce").astype(int).to_numpy()
        change_final = final_matrix.astype(int).to_numpy()
        if not np.array_equal(operational_final, change_final):
            raise ValueError("Final answer-change scores do not align with the operational response matrix.")
        initial_np = initial_matrix.astype(int).to_numpy()

        ro, pandas2ri, _ = _forensic_init_r(resp_df, keep_cols)
        ip_df = pd.DataFrame(psi_src)
        with (ro.default_converter + pandas2ri.converter).context():
            if not _forensic_build_psi(ro, ip_df, n_items):
                return {"flags": {"tt_agent": {"error": "Could not build psi matrix."}}}

            # Binary scores do not contain actual distractor identifiers. A stable
            # placeholder marks unchanged incorrect responses so detect_tt can use
            # its score-change path without inventing extra response categories.
            d_np = np.where(operational_final == 0, 1, 0).astype(int)
            d0_np = np.where(initial_np == 0, 1, 0).astype(int)

            def _push_int_matrix(name: str, values) -> None:
                flat = [int(v) for v in values.reshape(-1).tolist()]
                ro.globalenv[f"{name}_vec"] = ro.IntVector(flat)
                ro.r(f"{name} <- matrix({name}_vec, nrow={n_persons}, ncol={n_items}, byrow=TRUE)")

            _push_int_matrix("x", operational_final)
            _push_int_matrix("x_0", initial_np)
            _push_int_matrix("d", d_np)
            _push_int_matrix("d_0", d0_np)

            methods_requested = ["EDI_SD_NO", "EDI_SD_CO", "GBT_SD", "L_SD"]
            ro.globalenv["tt_methods_req"] = ro.StrVector(methods_requested)
            alpha_tt = _threshold_alpha(state, "detect_tt")
            ro.r(f"""
                assign('tt_err', NULL, envir = .GlobalEnv)
                tt_out <- tryCatch(
                    detect_tt(
                        method = tt_methods_req, psi = psi,
                        x = x, x_0 = x_0, d = d, d_0 = d_0,
                        alpha = {_r_num(alpha_tt)}
                    ),
                    error = function(e) {{
                        assign('tt_err', conditionMessage(e), envir = .GlobalEnv)
                        NULL
                    }}
                )
            """)
            if not bool(ro.r("!is.null(tt_out)")[0]):
                err = str(ro.r("as.character(tt_err)")[0])
                return {"flags": {"tt_agent": {"error": f"detect_tt failed: {err}"}}}

            stat_py = ro.conversion.rpy2py(ro.r("as.data.frame(tt_out$stat)"))
            pval_py = ro.conversion.rpy2py(ro.r("as.data.frame(tt_out$pval)"))
            stat_records = stat_py.to_dict(orient="records") if hasattr(stat_py, "to_dict") else []
            pval_records = pval_py.to_dict(orient="records") if hasattr(pval_py, "to_dict") else []
            methods = list(stat_records[0].keys()) if stat_records else methods_requested
            raw_flag_arr = np.asarray(ro.conversion.rpy2py(ro.r("tt_out$flag")))
            # rpy2 represents R logical NA as a missing numeric/object value.
            # Converting that array directly to bool would incorrectly turn NA
            # into True, so only explicit package TRUE values count as flags.
            flag_arr = np.where(pd.isna(raw_flag_arr), False, raw_flag_arr == True).astype(bool)
            if flag_arr.ndim == 3:
                flag_arr = np.any(flag_arr, axis=2)
            elif flag_arr.ndim == 1:
                flag_arr = flag_arr.reshape(-1, 1)

            flagged_by_method: dict[str, list[int]] = {}
            flagged_all: set[int] = set()
            for row_index, record in enumerate(stat_records):
                if row_index < len(pval_records):
                    for method, value in pval_records[row_index].items():
                        record[f"{method}_pval"] = value
                for method_index, method in enumerate(methods):
                    is_flagged = (
                        row_index < flag_arr.shape[0]
                        and method_index < flag_arr.shape[1]
                        and bool(flag_arr[row_index, method_index])
                    )
                    record[f"{method}_flag"] = int(is_flagged)
                    if is_flagged:
                        flagged_by_method.setdefault(method, []).append(row_index)
                        flagged_all.add(row_index)
            for method in methods:
                flagged_by_method.setdefault(method, [])

            return {"flags": {"tt_agent": {
                "stat": stat_records,
                "flagged": sorted(flagged_all),
                "flagged_by_method": flagged_by_method,
                "methods": methods,
                "n_persons": n_persons,
                "alpha": alpha_tt,
                "data_mode": "binary_score_change",
                "info": (
                    "Package-derived detect_tt score-change statistics. The input contains "
                    "initial/final dichotomous scores but no observed distractor identifiers; "
                    "unchanged incorrect responses use a stable placeholder solely for the "
                    "package score-change calculation."
                ),
            }}}
    except Exception as e:
        return {"flags": {"tt_agent": {"error": str(e), "flagged": [], "methods": []}}}


# ---------- 8. pk_agent (Preknowledge) ----------------------------------------
def pk_agent(state: State) -> dict:
    """detect_pk: L_S, S_S, W_S preknowledge. Uses file I/O only: no Python arrays passed to R (avoids py2rpy)."""
    print("--- FORENSIC pk_agent: detect_pk (file-based R) ---")
    selected = state.get("aberrance_functions") or []
    if selected and "detect_pk" not in selected:
        return {"flags": {}}
    resp_df = pd.DataFrame(state["responses"])
    keep_cols = _forensic_keep_cols(resp_df)
    psi_src = state.get("psi_data") or state.get("item_params") or []
    ci = state.get("compromised_items") or []
    if not keep_cols or not psi_src or len(psi_src) != len(keep_cols):
        return {"flags": {"pk_agent": {"error": "Missing response columns or item parameters."}}}
    # R requires at least one secure item (not in ci); cannot use "all items" as compromised
    if not ci:
        if len(keep_cols) < 2:
            return {"flags": {"pk_agent": {"error": "Preknowledge requires at least 2 items (one compromised, one secure)."}}}
        ci = list(range(1, len(keep_cols)))  # default: items 1..n-1, leave last as secure
    ci_clean = sorted({int(i) for i in ci if 1 <= int(i) <= len(keep_cols)})
    if not ci_clean:
        return {"flags": {"pk_agent": {"error": f"No valid compromised item IDs within 1..{len(keep_cols)}."}}}
    if len(ci_clean) >= len(keep_cols):
        return {
            "flags": {
                "pk_agent": {
                    "error": "Preknowledge requires at least one secure item; compromised CSV marks all response items as compromised.",
                    "flagged": [],
                    "methods": ["L_S", "S_S", "W_S"],
                }
            }
        }
    try:
        import rpy2.robjects as ro
        ro.r("library(aberrance)")
        n_items = len(keep_cols)
        ci_1based = ci_clean
        ci_r_str = "c(" + ",".join(map(str, ci_1based)) + ")"
        tmpdir = tempfile.mkdtemp(prefix="pk_agent_")
        path_x = str(Path(tmpdir) / "x.csv")
        path_psi = str(Path(tmpdir) / "psi.csv")
        path_y = str(Path(tmpdir) / "y.csv")
        path_stat = str(Path(tmpdir) / "pk_stat.csv")
        path_flag = str(Path(tmpdir) / "pk_flag.csv")
        try:
            block = resp_df[keep_cols].fillna(0).astype(int)
            block.to_csv(path_x, index=False, header=False)
            psi_rows = []
            for d in psi_src:
                a = float(d.get("a1", d.get("a", 0)))
                b = float(d.get("b", 0))
                c = float(d.get("g", d.get("c", 0))) if ("g" in d or "c" in d) else 0.0
                psi_rows.append({"a": a, "b": b, "c": c})
            pd.DataFrame(psi_rows).to_csv(path_psi, index=False)
            rt_data = state.get("rt_data") or []
            has_rt = bool(rt_data) and len(rt_data) == len(resp_df)
            if has_rt:
                rt_df = pd.DataFrame(rt_data)
                if rt_df.shape[1] >= len(keep_cols):
                    y_block = rt_df.iloc[:, :len(keep_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.01)
                    y_block = np.log(y_block.astype(float).clip(lower=0.001))
                    y_block.to_csv(path_y, index=False, header=False)
                else:
                    has_rt = False
            path_x_r = path_x.replace("\\", "/")
            path_psi_r = path_psi.replace("\\", "/")
            path_y_r = path_y.replace("\\", "/")
            path_stat_r = path_stat.replace("\\", "/")
            path_flag_r = path_flag.replace("\\", "/")
            ro.globalenv["path_x"] = path_x_r
            ro.globalenv["path_psi"] = path_psi_r
            ro.globalenv["path_y"] = path_y_r
            ro.globalenv["path_stat"] = path_stat_r
            ro.globalenv["path_flag"] = path_flag_r
            ro.r("ci_r <- " + ci_r_str)
            ro.r("""
                x <- as.matrix(read.csv(path_x, header = FALSE))
                psi_df <- read.csv(path_psi, header = TRUE)
                psi <- as.matrix(psi_df[, c('a','b','c')])
                ci <- ci_r
            """)
            if has_rt:
                ro.r("""
                    y <- as.matrix(read.csv(path_y, header = FALSE))
                    pk_beta_est <- apply(y, 2, mean)
                    pk_alpha_est <- 1 / apply(y, 2, sd)
                    pk_alpha_est[!is.finite(pk_alpha_est)] <- 1.0
                    psi <- cbind(psi, alpha = pk_alpha_est, beta = pk_beta_est)
                """)

            stat_cols: dict[str, list] = {}
            flagged_set: set[int] = set()
            methods: list[str] = []
            method_errors: list[str] = []

            for method, args in [
                *[(m, "x = x") for m in ["L_S", "ML_S", "LR_S", "S_S", "W_S"]],
                *([(m, "y = y") for m in ["L_T", "W_T"]] if has_rt else []),
                *([("L_ST", "x = x, y = y")] if has_rt else []),
            ]:
                ro.globalenv["pk_method_req"] = ro.StrVector([method])
                ro.r(f"""
                    assign('pk_err', NULL, envir = .GlobalEnv)
                    pk_out <- tryCatch(
                        detect_pk(method = pk_method_req, ci = ci, psi = psi, {args}, alpha = {_r_num(_threshold_alpha(state, "detect_pk"))}),
                        error = function(e) {{ assign('pk_err', conditionMessage(e), envir = .GlobalEnv); NULL }})
                """)
                if not bool(ro.r("!is.null(pk_out)")[0]):
                    err = "no output"
                    try:
                        if not bool(ro.r("is.null(get0('pk_err', envir = .GlobalEnv, ifnotfound = NULL))")[0]):
                            err = str(ro.r("as.character(pk_err)")[0])
                    except Exception:
                        pass
                    method_errors.append(f"{method}: {err}")
                    continue
                vals = [float(v) for v in ro.r("as.numeric(pk_out$stat[, 1])")]
                stat_cols[method] = vals
                try:
                    pvals = [float(v) for v in ro.r("as.numeric(pk_out$pval[, 1])")]
                    stat_cols[f"{method}_pval"] = pvals
                except Exception:
                    pass
                try:
                    flag_any = ro.r("as.logical(apply(pk_out$flag, 1, any))")
                    method_flags = [1 if bool(v) else 0 for v in flag_any]
                    stat_cols[f"{method}_flag"] = method_flags
                    flagged_set.update(i for i, v in enumerate(method_flags) if v)
                except Exception:
                    pass
                methods.append(method)

            if not methods:
                return {"flags": {"pk_agent": {"error": "detect_pk failed for all methods: " + "; ".join(method_errors)}}}
            stat_df = pd.DataFrame(stat_cols)
            records = stat_df.to_dict(orient="records")
            flagged = sorted(flagged_set)
            return {"flags": {"pk_agent": {
                "stat": records,
                "flagged": flagged,
                "methods": methods or ["L_S", "S_S", "W_S"],
                "method_errors": method_errors,
            }}}
        finally:
            for p in (path_x, path_psi, path_y, path_stat, path_flag):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                Path(tmpdir).rmdir()
            except Exception:
                pass
    except Exception as e:
        return {"flags": {"pk_agent": {"error": str(e)}}}


# ──────────────────────────────────────────────────────────────────────────────
# MANAGER NODE  (LLM Synthesizer)
# ──────────────────────────────────────────────────────────────────────────────

def manager_router(state: State) -> dict:
    """Initialize flags dict and broadcast to all specialists."""
    print("--- FORENSIC Manager Router: dispatching to 8 specialists ---")
    return {"flags": state.get("flags") or {}, "next_step": "broadcast"}


def manager_synthesizer(state: State) -> dict:
    """Synthesize forensic verdict from all specialist flags via LLM."""
    print("--- FORENSIC Manager Synthesizer: generating final report ---")
    flags = state.get("flags", {})
    model_settings = state.get("model_settings") or {}
    load_dotenv()
    # Resolve LLM provider and API key — prefer OpenRouter, fall back to Google
    llm_provider = model_settings.get("llm_provider", "")
    or_key = model_settings.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")
    google_key = model_settings.get("google_api_key") or os.getenv("GOOGLE_API_KEY", "")
    model_id = model_settings.get("llm_model_id") or ""
    if not llm_provider:
        # Auto-detect: OpenRouter first if key is available
        llm_provider = "openrouter" if or_key else ("google" if google_key else "")
    # If provider is set but its key is missing, fall back to the other provider if possible
    if llm_provider == "openrouter" and not or_key and google_key:
        llm_provider = "google"
    elif llm_provider == "google" and not google_key and or_key:
        llm_provider = "openrouter"

    api_key = or_key if llm_provider == "openrouter" else google_key
    if llm_provider == "openrouter" and not model_id:
        # Default to a known OpenRouter free model (avoid invalid/removed IDs)
        model_id = "meta-llama/llama-3.3-70b-instruct:free"

    # Build a textual summary of all specialist results for the LLM
    specialist_summary_parts = []
    all_flagged_persons = {}  # person_idx -> list of agents that flagged them
    # Restrict to agents corresponding to the aberrance_functions actually used.
    fn_selected = state.get("aberrance_functions") or []
    fn_to_agent = {
        "detect_nm": "nm_agent",
        "detect_pm": "pm_agent",
        "detect_ac": "ac_agent",
        "detect_as": "as_agent",
        "detect_rg": "rg_agent",
        "detect_tt": "tt_agent",
        "detect_pk": "pk_agent",
    }
    allowed_agents = {fn_to_agent[fn] for fn in fn_selected if fn in fn_to_agent} if fn_selected else set(flags.keys())

    for agent_name, data in flags.items():
        if allowed_agents and agent_name not in allowed_agents:
            continue
        if isinstance(data, dict):
            err = data.get("error")
            info = data.get("info")
            flagged = data.get("flagged", [])
            flagged_copiers = data.get("flagged_copiers", [])
            methods = data.get("methods", [])
            rte = data.get("rte", [])
            stat = data.get("stat", [])
            pairs = data.get("pairs", [])

            if err:
                specialist_summary_parts.append(f"**{agent_name}**: ERROR — {err}")
            elif info:
                specialist_summary_parts.append(f"**{agent_name}**: INFO — {info}")
            else:
                n_flagged = len(flagged) + len(flagged_copiers)
                parts = [f"**{agent_name}** (methods: {', '.join(methods)})"]
                if n_flagged > 0:
                    parts.append(f"  Flagged {n_flagged} person(s): {(flagged or flagged_copiers)[:20]}")
                else:
                    parts.append("  No persons flagged.")
                if rte:
                    mean_rte = sum(rte) / len(rte) if rte else 0
                    parts.append(f"  Mean RTE: {mean_rte:.3f}")
                if pairs:
                    sig_pairs = [p for p in pairs if any(v for k, v in p.items() if k not in ("Source", "Copier") and isinstance(v, (int, float)) and v > 2)]
                    if sig_pairs:
                        parts.append(f"  Significant copying pairs (top 10): {sig_pairs[:10]}")
                specialist_summary_parts.append("\n".join(parts))

            # Aggregate flagged persons
            for p in flagged:
                all_flagged_persons.setdefault(p, []).append(agent_name)
            for p in flagged_copiers:
                all_flagged_persons.setdefault(p, []).append(agent_name)

    # Top risk students
    risk_ranking = sorted(all_flagged_persons.items(), key=lambda x: -len(x[1]))
    top5 = risk_ranking[:5]
    top5_text = "\n".join([f"  Student {p+1}: flagged by {', '.join(agents)} ({len(agents)} agents)" for p, agents in top5]) if top5 else "  No students flagged across multiple agents."

    specialist_report = "\n\n".join(specialist_summary_parts) if specialist_summary_parts else "No specialist results available."

    # Human-readable list of agents used in this run for the prompt header.
    agent_labels = {
        "nm_agent": "Nonparametric misfit (detect_nm)",
        "pm_agent": "Parametric misfit (detect_pm)",
        "ac_agent": "Answer copying (detect_ac)",
        "as_agent": "Answer similarity / clusters (detect_as)",
        "rg_agent": "Rapid guessing / low effort (detect_rg)",
        "cp_agent": "Change-point in behavior (detect_cp)",
        "tt_agent": "Tampering / erasures (detect_tt)",
        "pk_agent": "Preknowledge on compromised items (detect_pk)",
    }
    used_agent_lines = [
        f"- {agent_labels[a]}" for a in agent_labels.keys() if a in allowed_agents
    ] or ["- (No specialist agents ran successfully.)"]

    prompt = f"""You are a test-security data forensics expert writing a short, professional report.

You are given the outputs from the following aberrance detectors for a single exam administration:
{chr(10).join(used_agent_lines)}

Below is a compact summary of the specialist agents' outputs and a ranking of high-risk examinees.

## Specialist Results (raw summary)

{specialist_report}

## Top High-Risk Students (by number of flags)

{top5_text}

## Task

Write a concise **Forensic Verdict** in Markdown format, following this structure EXACTLY:

1. **Case Overview**  
   - Briefly describe what was analyzed (generic exam description if unknown), what data were used (responses, response times, item parameters ψ, compromised items), and whether all eight detectors had usable data.

2. **Global Risk Assessment**  
   - Give an overall assessment of exam integrity using one of: **Low concern**, **Moderate concern**, or **High concern**.  
   - Justify this in 2–3 sentences, referring to the relative strength and consistency of the statistical evidence across detectors.

3. **Findings by Threat Dimension**  
   For each dimension below, include a short paragraph. If a dimension has essentially no evidence, say that explicitly and keep it brief.

   - **Model Misfit & Ability–Performance Anomalies** (detect_pm, detect_nm):  
     Summarize how many examinees are flagged (approximate counts or percentages if available), and whether misfit is scattered or concentrated.

   - **Collusion & Similarity** (detect_ac, detect_as):  
     Summarize whether there are suspicious source–copier pairs or clusters, and whether they exceed typical p-value thresholds (e.g., p < .05). If no convincing evidence, say so clearly.

   - **Speed / Effort Anomalies** (detect_rg, detect_cp):  
     Describe the presence/absence of large groups of rapid guessers and any clear change-points where behavior shifts from normal effort to low effort or vice versa.

   - **Tampering & Preknowledge** (detect_tt, detect_pk):  
     Describe whether tampering indicators or preknowledge on compromised items are observed, and at what rough magnitude.

4. **Watchlist (High-Risk Examinees)**  
   - Provide a short bullet list of up to 5–10 highest-risk examinees.  
   - Each bullet: “ID X: flagged by [detectors], brief 1-sentence rationale.”  
   - Only include examinees where the statistical evidence is clearly non-trivial.

5. **Interpretive Notes & Limitations**  
   - Emphasize that these detectors provide **statistical indicators, not proof** of misconduct.  
   - Mention any relevant limitations you can infer (e.g., no RT data, no compromised items specified, very few flags).

STYLE REQUIREMENTS:
- Use clear, neutral, professional language suitable for a psychometric / test-security audience.
- Be concise: total length **no more than ~500 words**.
- Do NOT invent exact numeric values that are not implied by the input. When necessary, use qualitative terms like “a small number”, “a moderate number”, or “a substantial number”.
- Do NOT speculate about motivations; focus only on patterns in the data.
- Use Markdown headers and bullet points, following the section titles above."""

    # Try LLM call
    report = None
    last_err: str | None = None
    try:
        if llm_provider == "openrouter" and api_key:
            messages = [{"role": "user", "content": prompt}]
            report, err = _call_openrouter(api_key, model_id, messages, timeout=60)
            if err:
                last_err = f"OpenRouter: {err}"
                report = None
        elif api_key:
            # Google Gemini
            variants = _get_llm_model_variants(api_key)
            body = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 4096},
            }
            for variant in variants:
                url = f"https://generativelanguage.googleapis.com/v1beta/{variant}:generateContent"
                try:
                    resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text.strip():
                        report = text.strip()
                        break
                except Exception as e:
                    last_err = f"Google: {type(e).__name__}: {e}"
                    continue
        else:
            # No API key available for either provider
            if or_key or google_key:
                # Provider selected but missing key (or empty after fallback)
                last_err = "Missing API key for selected LLM provider."
            else:
                last_err = "No LLM API key configured (OPENROUTER_API_KEY / GOOGLE_API_KEY missing)."
    except Exception as e:
        last_err = f"{type(e).__name__}: {e}"

    if not report:
        # Fallback: generate a structured report without LLM
        report_parts = ["# Forensic Verdict (Auto-Generated)\n"]
        if last_err:
            report_parts.append(f"*LLM was not available ({last_err}); this is a rule-based summary.*\n")
        else:
            report_parts.append("*LLM was not available; this is a rule-based summary.*\n")
        # Threat categories
        categories = {
            "Collusion": ["ac_agent", "as_agent"],
            "Speed Anomalies": ["rg_agent", "cp_agent"],
            "Model Misfit": ["pm_agent", "nm_agent"],
            "Tampering / Preknowledge": ["tt_agent", "pk_agent"],
        }
        for cat_name, agents in categories.items():
            cat_flagged = set()
            cat_errors = []
            for ag in agents:
                ag_data = flags.get(ag, {})
                if ag_data.get("error"):
                    cat_errors.append(f"{ag}: {ag_data['error']}")
                cat_flagged.update(ag_data.get("flagged", []))
                cat_flagged.update(ag_data.get("flagged_copiers", []))
            severity = "Critical" if len(cat_flagged) > 5 else ("Warning" if cat_flagged else "Clear")
            report_parts.append(f"## {cat_name} — {severity}")
            if cat_flagged:
                report_parts.append(f"- Flagged students: {sorted(p+1 for p in cat_flagged)}")
            if cat_errors:
                for ce in cat_errors:
                    report_parts.append(f"- *{ce}*")
            if not cat_flagged and not cat_errors:
                report_parts.append("- No issues detected.")
            report_parts.append("")
        report_parts.append("## Top 5 High-Risk Students\n")
        if top5:
            for p, agents_list in top5:
                report_parts.append(f"- **Student {p+1}**: flagged by {', '.join(agents_list)} ({len(agents_list)} agents)")
        else:
            report_parts.append("- No students flagged by multiple agents.")
        report = "\n".join(report_parts)

    return {
        "final_report": report,
    }


def forensic_reporter(state: State) -> dict:
    """Record a brief, deterministic audit summary after the LLM synthesizer (for traceability / UI)."""
    print("--- FORENSIC Reporter: compiling brief ---")
    flags = state.get("flags") or {}
    report = (state.get("final_report") or "").strip()
    n_keys = len(flags)
    n_err = sum(1 for v in flags.values() if isinstance(v, dict) and v.get("error"))
    n_info = sum(1 for v in flags.values() if isinstance(v, dict) and v.get("info") and not v.get("error"))
    wc = len(report.split()) if report else 0
    brief = (
        f"Specialist channels: {n_keys}; errors: {n_err}; info-only: {n_info}; "
        f"verdict ~{wc} words."
    )
    return {"reporter_brief": brief}


# ──────────────────────────────────────────────────────────────────────────────
# FORENSIC WORKFLOW (StateGraph)
# ──────────────────────────────────────────────────────────────────────────────

forensic_workflow = build_forensic_workflow(
    router=manager_router,
    specialists={
        "nm_agent": nm_agent,
        "pm_agent": pm_agent,
        "ac_agent": ac_agent,
        "as_agent": as_agent,
        "pk_agent": pk_agent,
        "rg_agent": rg_agent,
        "cp_agent": cp_agent,
        "tt_agent": tt_agent,
    },
    synthesizer=manager_synthesizer,
    reporter=forensic_reporter,
)


psych_workflow = app = build_psychometric_workflow(
    orchestrator=orchestrator_agent,
    irt=irt_agent,
    response_time=rt_agent,
    analyzer=analyze_agent,
)


if __name__ == "__main__":
    print("How to run the graph:")
    print("  1. Streamlit UI (recommended):  uv run streamlit run ui.py")
    print("     Then upload response + RT data and run the analysis.")
    print("  2. LangGraph server/API:       uv run langgraph dev")
    print("     Then POST to the API to invoke psych_workflow.")
    print("")
    print("Running a minimal test invoke (3 persons × 2 items)...")
    sample_state: State = {
        "responses": [{"0": 1, "1": 0}, {"0": 0, "1": 1}, {"0": 1, "1": 1}],
        "rt_data": [{"0": 1.2, "1": 0.8}, {"0": 2.1, "1": 1.5}, {"0": 0.9, "1": 1.1}],
        "theta": 0.0,
        "latency_flags": [],
        "next_step": "start",
    }
    try:
        result = psych_workflow.invoke(sample_state)
        print("Done. Keys in result:", list(result.keys()))
    except Exception as e:
        print(f"Test invoke failed (e.g. R/mirt not installed): {e}")
