"""
Streamlit UI for loading response and response-time datasets, then running the
psych_workflow graph.

Data loading lives here—not in the Orchestrator. The Orchestrator only routes;
the UI reads the two files and passes them as initial state into the graph.
"""

from pathlib import Path
import json
import math
import os
import re
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import requests
from dotenv import load_dotenv

from graph import analyze_prompt, psych_workflow


def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_col = df.columns[0]
    if str(first_col).strip() == "" or str(first_col).startswith("Unnamed"):
        return df.drop(columns=[first_col])

    maybe_index = pd.to_numeric(df[first_col], errors="coerce")
    if maybe_index.notna().all():
        vals = maybe_index.astype(int)
        if vals.is_unique:
            sorted_vals = sorted(vals.tolist())
            if sorted_vals == list(range(1, len(vals) + 1)) or sorted_vals == list(range(0, len(vals))):
                return df.drop(columns=[first_col])

    return df


def _coerce_numeric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = []
    for col in df.columns:
        if (df[col].notna() & numeric_df[col].isna()).any():
            non_numeric_cols.append(str(col))
    if non_numeric_cols:
        raise ValueError(
            f"{label} has non-numeric values in columns: {', '.join(non_numeric_cols)}."
        )
    return numeric_df


def _validate_binary_responses(resp_df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = _coerce_numeric(resp_df, "Response data")
    invalid_cols = []
    for col in numeric_df.columns:
        values = numeric_df[col].dropna().unique().tolist()
        if values and not set(values).issubset({0, 1}):
            invalid_cols.append(str(col))
    if invalid_cols:
        raise ValueError(
            "Response data must be binary (0/1). Invalid columns: "
            + ", ".join(invalid_cols)
        )
    return numeric_df


def _interpret_prompt(prompt: str) -> dict:
    return analyze_prompt(prompt)


def _paginate_df(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        st.info(f"No data for {label}.")
        return
    filtered_df = _filter_and_sort_df(df, label)
    if filtered_df.empty:
        st.info(f"No rows match the current filters for {label}.")
        return
    page_size = 10
    total_pages = math.ceil(len(filtered_df) / page_size)
    if total_pages > 1:
        page = st.number_input(
            f"{label} page",
            min_value=1,
            max_value=total_pages,
            step=1,
            value=1,
            key=f"{label}_page",
        )
    else:
        page = 1
    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(filtered_df.iloc[start:end], width="stretch")


def _filter_and_sort_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = [str(c) for c in df.columns]
    if not cols:
        return df

    with st.form(f"{label}_filter_form"):
        st.caption("Filter and sort")
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_col = st.selectbox(
                "Filter column",
                options=["(none)"] + cols,
                key=f"{label}_filter_col",
            )
        with c2:
            sort_col = st.selectbox(
                "Sort column",
                options=["(none)"] + cols,
                key=f"{label}_sort_col",
            )
        with c3:
            sort_dir = st.radio(
                "Order",
                options=["asc", "desc"],
                horizontal=True,
                key=f"{label}_sort_dir",
            )

        filtered_df = df
        if filter_col != "(none)":
            series = filtered_df[filter_col]
            if pd.api.types.is_numeric_dtype(series):
                min_val = st.number_input(
                    "Min",
                    value=float(series.min()) if series.notna().any() else 0.0,
                    key=f"{label}_min",
                )
                max_val = st.number_input(
                    "Max",
                    value=float(series.max()) if series.notna().any() else 0.0,
                    key=f"{label}_max",
                )
                filtered_df = filtered_df[series.between(min_val, max_val, inclusive="both")]
            else:
                text = st.text_input(
                    "Contains",
                    key=f"{label}_contains",
                )
                if text:
                    filtered_df = filtered_df[
                        series.astype(str).str.contains(text, case=False, na=False)
                    ]

        if sort_col != "(none)":
            ascending = sort_dir == "asc"
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

        apply_filters = st.form_submit_button("Apply")

    if not apply_filters and f"{label}_last_filtered" in st.session_state:
        return st.session_state[f"{label}_last_filtered"]

    st.session_state[f"{label}_last_filtered"] = filtered_df
    return filtered_df


@st.cache_data(show_spinner=False)
def _run_workflow_cached(payload_json: str) -> dict:
    payload = json.loads(payload_json)
    initial_state = {
        "responses": payload["responses"],
        "rt_data": payload["rt_data"],
        "theta": 0.0,
        "latency_flags": [],
        "next_step": "",
        "model_settings": payload["model_settings"],
        "is_verified": payload["is_verified"],
    }
    return psych_workflow.invoke(initial_state)


def _create_wright_map(item_params: pd.DataFrame, person_params: pd.DataFrame) -> str:
    """Create a Wright Map using R's WrightMap package."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
    except Exception as exc:
        print(f"Wright Map skipped; rpy2/R not available ({type(exc).__name__}: {exc})")
        return None
    
    # Extract item difficulty (b parameter) - try different possible column names
    b_col = None
    for col in ['b', 'b1', 'd', 'difficulty']:
        if col in item_params.columns:
            b_col = col
            break
    
    if b_col is None:
        return None
    
    # Extract person ability (F1 or theta) - try different possible column names
    ability_col = None
    for col in ['F1', 'theta', 'ability', 'F']:
        if col in person_params.columns:
            ability_col = col
            break
    
    if ability_col is None:
        return None
    
    # Get valid item data
    valid_mask = item_params[b_col].notna()
    item_df = item_params.loc[valid_mask].copy()
    if 'item' not in item_df.columns:
        item_df['item'] = item_df.index.astype(str)
    
    # Get person ability data
    person_df = person_params[[ability_col]].dropna().copy()
    
    if item_df.empty or person_df.empty:
        return None
    
    # Create output path
    wright_map_path = Path(tempfile.gettempdir()) / f"wright_map_{hash(str(item_df[b_col].values) + str(person_df[ability_col].values))}.png"
    wright_map_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with (ro.default_converter + pandas2ri.converter).context():
            # Convert DataFrames to R objects
            ro.globalenv["item_df"] = item_df
            ro.globalenv["person_df"] = person_df
            ro.globalenv["wright_map_path"] = str(wright_map_path)
            ro.globalenv["b_col"] = b_col
            ro.globalenv["ability_col"] = ability_col
            
            ro.r(
                """
                library(WrightMap)
                library(grDevices)
                
                # Extract item difficulties as a matrix/vector
                item_difficulty <- as.numeric(item_df[[b_col]])
                
                # Extract person abilities as a vector
                person_ability <- as.numeric(person_df[[ability_col]])
                
                # Create Wright Map
                # item.prop = 0.8 means 80% space for items, 20% for persons (histogram)
                png(wright_map_path, width=1200, height=1000, res=150, type="cairo")
                wrightMap(thetas = person_ability, 
                         thresholds = item_difficulty,
                         item.prop = 0.8,
                         main.title = "Wright Map: Person Ability Distribution and Item Difficulties")
                dev.off()
                """
            )
    except Exception as exc:
        print(f"Wright Map generation failed: {type(exc).__name__}: {exc}")
        return None
    
    if not wright_map_path.exists():
        return None
    
    return str(wright_map_path)


def _query_llm_analysis(question: str, analysis_context: dict) -> str:
    """Query LLM with a question about the psychometric analysis results."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    
    # Prepare context summary with key statistics
    context_summary = []
    
    if analysis_context.get("item_params"):
        item_df = pd.DataFrame(analysis_context["item_params"])
        context_summary.append(f"Item parameters: {len(item_df)} items analyzed.")
        if not item_df.empty:
            # Include key statistics for common columns
            if 'b' in item_df.columns or 'b1' in item_df.columns:
                b_col = 'b' if 'b' in item_df.columns else 'b1'
                b_vals = pd.to_numeric(item_df[b_col], errors='coerce').dropna()
                if len(b_vals) > 0:
                    context_summary.append(f"Item difficulty (b) range: {b_vals.min():.2f} to {b_vals.max():.2f}, mean: {b_vals.mean():.2f}")
            if 'a' in item_df.columns or 'a1' in item_df.columns:
                a_col = 'a' if 'a' in item_df.columns else 'a1'
                a_vals = pd.to_numeric(item_df[a_col], errors='coerce').dropna()
                if len(a_vals) > 0:
                    context_summary.append(f"Item discrimination (a) range: {a_vals.min():.2f} to {a_vals.max():.2f}, mean: {a_vals.mean():.2f}")
    
    if analysis_context.get("person_params"):
        person_df = pd.DataFrame(analysis_context["person_params"])
        context_summary.append(f"Person parameters: {len(person_df)} persons analyzed.")
        if not person_df.empty:
            # Include ability statistics
            for col in ['F1', 'theta', 'ability', 'F']:
                if col in person_df.columns:
                    ability_vals = pd.to_numeric(person_df[col], errors='coerce').dropna()
                    if len(ability_vals) > 0:
                        context_summary.append(f"Person ability ({col}) range: {ability_vals.min():.2f} to {ability_vals.max():.2f}, mean: {ability_vals.mean():.2f}")
                        break
    
    if analysis_context.get("item_fit"):
        fit_df = pd.DataFrame(analysis_context["item_fit"])
        context_summary.append(f"Item fit statistics available for {len(fit_df)} items.")
        if not fit_df.empty and 'p' in fit_df.columns:
            p_vals = pd.to_numeric(fit_df['p'], errors='coerce').dropna()
            if len(p_vals) > 0:
                context_summary.append(f"Item fit p-values: {sum(p_vals < 0.05)} items with p < 0.05")
    
    context_text = "\n".join(context_summary) if context_summary else "Analysis results are available."
    
    system = (
        "You are a psychometrics expert assistant helping users interpret IRT analysis results. "
        "You have access to item parameters, person parameters, and item fit statistics. "
        "Provide clear, helpful explanations and insights based on the analysis context provided. "
        "If the question is about specific values or patterns, refer to the available data. "
        "Be conversational and educational."
    )
    
    user_prompt = (
        f"Analysis Context:\n{context_text}\n\n"
        f"User Question: {question}\n\n"
        "Please provide a helpful analysis or answer to the user's question."
    )
    
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system}\n\n{user_prompt}"}]}
        ]
    }
    
    # Try different model name variations if the first one fails
    model_variants = [
        model,  # Try the configured/default model first
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-pro",
    ]
    
    resp = None
    last_error = None
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            resp.raise_for_status()
            # If successful, break
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                last_error = e
                resp = None
                continue
            else:
                # For non-404 errors, raise immediately
                raise
        except Exception as e:
            last_error = e
            resp = None
            continue
    
    # If all hardcoded models failed, try to discover available models dynamically
    if resp is None:
        try:
            list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            list_resp = requests.get(list_url, timeout=30)
            list_resp.raise_for_status()
            models_data = list_resp.json().get("models", [])
            
            # Find models that support generateContent
            candidates = []
            for m in models_data:
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    candidates.append(m["name"])
            
            # Prefer flash models, then pro models
            for name in candidates:
                if "gemini-1.5-flash" in name:
                    url = f"https://generativelanguage.googleapis.com/v1beta/{name}:generateContent"
                    try:
                        resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
                        resp.raise_for_status()
                        break
                    except:
                        continue
            
            # If flash didn't work, try any gemini model
            if resp is None:
                for name in candidates:
                    if "gemini" in name:
                        url = f"https://generativelanguage.googleapis.com/v1beta/{name}:generateContent"
                        try:
                            resp = requests.post(url, params={"key": api_key}, json=body, timeout=30)
                            resp.raise_for_status()
                            break
                        except:
                            continue
        except Exception as e:
            # If model discovery also fails, use the last error from hardcoded attempts
            pass
    
    if resp is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        return f"LLM analysis failed. Tried multiple models but none were available. Last error: {error_msg}. Please check your API key and model access."
    
    try:
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() if text else "No response from LLM."
    except Exception as e:
        return f"Error parsing LLM response: {type(e).__name__}: {e}"


def _analyze_wright_map_image(image_path: str) -> str:
    """Analyze Wright Map PNG image using LLM and generate a paper-ready summary."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
    
    if not api_key:
        return "LLM analysis unavailable: GOOGLE_API_KEY not set in .env file."
    
    if not Path(image_path).exists():
        return "Wright Map image not found."
    
    # Read and encode the image
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return f"Error reading image: {type(e).__name__}: {e}"
    
    system = (
        "You are a psychometrics expert. Analyze this Wright Map (item-person map) and provide "
        "a concise, two-paragraph summary suitable for a research paper. "
        "The Wright Map shows the distribution of person abilities (histogram on the left) and "
        "item difficulties (red lines on the right) on the same latent trait scale. "
        "Write EXACTLY TWO PARAGRAPHS: "
        "(1) First paragraph: Describe the patterns observed - the distribution of person abilities, "
        "the distribution and range of item difficulties, and the alignment between them. "
        "Include any gaps or clusters in the measurement. "
        "(2) Second paragraph: Present the findings and their implications for test design, "
        "measurement precision, and interpretation. "
        "Keep each paragraph concise (3-5 sentences). Write in a formal, academic style suitable for publication."
    )
    
    # Prepare the request body with image
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    }
                ]
            }
        ]
    }
    
    # First, try to discover available models (more reliable than hardcoded names)
    resp = None
    last_error = None
    discovered_models = []
    
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        list_resp = requests.get(list_url, timeout=30)
        list_resp.raise_for_status()
        models_data = list_resp.json().get("models", [])
        
        # Find models that support generateContent (vision-capable models)
        for m in models_data:
            methods = m.get("supportedGenerationMethods", [])
            model_name = m.get("name", "")
            if "generateContent" in methods and "gemini" in model_name.lower():
                discovered_models.append(model_name)
        
        # Prioritize: flash models first (faster), then pro models
        flash_models = [m for m in discovered_models if "flash" in m.lower()]
        pro_models = [m for m in discovered_models if "pro" in m.lower() and "flash" not in m.lower()]
        other_models = [m for m in discovered_models if m not in flash_models and m not in pro_models]
        
        # Try discovered models in priority order
        for model_name in flash_models + pro_models + other_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            try:
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_error = e
                    resp = None
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                resp = None
                continue
    except Exception as e:
        # If model discovery fails, fall back to hardcoded models
        last_error = e
    
    # If discovery failed or no models worked, try hardcoded variants
    if resp is None:
        model_variants = [
            model,
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash-002",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-pro",
        ]
        
        for model_variant in model_variants:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
            try:
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=60)
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    last_error = e
                    resp = None
                    continue
                else:
                    raise
            except Exception as e:
                last_error = e
                resp = None
                continue
    
    if resp is None:
        error_msg = str(last_error) if last_error else "Unknown error"
        discovered_info = f"Discovered {len(discovered_models)} models" if discovered_models else "No models discovered"
        return (
            f"LLM analysis failed. {discovered_info}. "
            f"Last error: {error_msg}. "
            f"Please check your API key and ensure you have access to Gemini models with vision capabilities."
        )
    
    try:
        data = resp.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() if text else "No response from LLM."
    except Exception as e:
        return f"Error parsing LLM response: {type(e).__name__}: {e}"


def _render_results(final: dict) -> None:
    tab1, tab2 = st.tabs(["📊 Response Results", "⏱️ RT Analysis"])

    with tab1:
        st.metric("θ (theta)", round(final.get("theta", 0), 4))
        if "responses" in final:
            st.dataframe(pd.DataFrame(final["responses"]))
        if final.get("icc_error"):
            st.warning(final["icc_error"])
        subtab_params, subtab_fit, subtab_icc, subtab_analysis = st.tabs(
            ["Parameters & Wright Map", "Item Fit", "ICC", "Analysis Chat"]
        )
        with subtab_params:
            # Create and display Wright Map first
            item_params_df = pd.DataFrame(final.get("item_params", []))
            person_params_df = pd.DataFrame(final.get("person_params", []))
            
            if not item_params_df.empty and not person_params_df.empty:
                wright_map_path = _create_wright_map(item_params_df, person_params_df)
                if wright_map_path and Path(wright_map_path).exists():
                    st.subheader("Wright Map")
                    st.image(wright_map_path)
                    st.caption("Person ability distribution (histogram) and item difficulties (red lines) on the same latent trait scale.")
                    
                    # LLM Analysis of Wright Map
                    wright_map_analysis_key = f"wright_map_analysis_{hash(wright_map_path)}"
                    
                    if wright_map_analysis_key not in st.session_state:
                        st.session_state[wright_map_analysis_key] = None
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("**LLM Analysis**")
                    with col2:
                        if st.button("Generate Analysis", key="generate_wright_analysis"):
                            with st.spinner("Analyzing Wright Map with LLM..."):
                                st.session_state[wright_map_analysis_key] = _analyze_wright_map_image(wright_map_path)
                            st.rerun()
                    
                    if st.session_state[wright_map_analysis_key]:
                        st.markdown("### Paper-Ready Summary")
                        st.markdown(st.session_state[wright_map_analysis_key])
                else:
                    st.info("Wright Map could not be generated. Check that item difficulty (b) and person ability (F1) columns are available.")
            
            # Person Parameters table with scrollbar
            st.subheader("Person Parameters")
            if final.get("person_params"):
                person_df = pd.DataFrame(final["person_params"])
                filtered_person_df = _filter_and_sort_df(person_df, "Person parameters")
                st.dataframe(filtered_person_df, height=300, use_container_width=True)
            else:
                st.info("No person parameters available.")
            
            # Item Parameters table with scrollbar
            st.subheader("Item Parameters")
            if final.get("item_params"):
                item_df = pd.DataFrame(final["item_params"])
                filtered_item_df = _filter_and_sort_df(item_df, "Item parameters")
                st.dataframe(filtered_item_df, height=300, use_container_width=True)
            else:
                st.info("No item parameters available.")
        
        with subtab_fit:
            if final.get("item_fit"):
                item_fit_df = pd.DataFrame(final["item_fit"])
                filtered_fit_df = _filter_and_sort_df(item_fit_df, "Item fit")
                st.dataframe(filtered_fit_df, height=400, use_container_width=True)
            else:
                st.info("No item fit data available.")
        
        with subtab_icc:
            if final.get("icc_plot_path") and Path(final["icc_plot_path"]).exists():
                st.image(final["icc_plot_path"])
            else:
                st.info("No ICC plot available.")
        
        with subtab_analysis:
            st.subheader("Ask LLM for Analysis")
            st.caption("Ask questions about your psychometric analysis results. The LLM has access to item parameters, person parameters, and item fit statistics.")
            
            # Initialize conversation history in session state
            if "analysis_chat_history" not in st.session_state:
                st.session_state.analysis_chat_history = []
            
            # Display conversation history
            if st.session_state.analysis_chat_history:
                st.markdown("### Conversation History")
                for i, (role, message) in enumerate(st.session_state.analysis_chat_history):
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(message)
                    else:
                        with st.chat_message("assistant"):
                            st.write(message)
            
            # Input form for new questions
            with st.form("analysis_query_form", clear_on_submit=True):
                user_question = st.text_area(
                    "Ask a question about your analysis:",
                    placeholder="e.g., 'Which items are the most difficult?', 'What does the item fit suggest?', 'Are there any problematic items?'",
                    height=100,
                    key="analysis_question_input"
                )
                submit_question = st.form_submit_button("Ask LLM", use_container_width=True)
                
                if submit_question and user_question.strip():
                    # Add user question to history
                    st.session_state.analysis_chat_history.append(("user", user_question.strip()))
                    
                    # Query LLM with analysis context
                    with st.spinner("Analyzing with LLM..."):
                        response = _query_llm_analysis(user_question.strip(), final)
                        st.session_state.analysis_chat_history.append(("assistant", response))
                    
                    # Rerun to show the new response
                    st.rerun()
            
            # Clear conversation button
            if st.session_state.analysis_chat_history:
                if st.button("Clear Conversation", key="clear_analysis_chat"):
                    st.session_state.analysis_chat_history = []
                    st.rerun()

    with tab2:
        if final.get("latency_flags"):
            st.write("Latency flags:", ", ".join(final["latency_flags"]))
        if final.get("rt_plot_path") and Path(final["rt_plot_path"]).exists():
            st.subheader("RT histograms")
            st.image(final["rt_plot_path"])

st.set_page_config(page_title="Psych MAS — Data & Run", layout="centered")
st.title("Psych MAS — Load data & run workflow")

st.markdown(
    "Upload **response** and **response-time** CSVs."
    #"workflow as initial state; the Orchestrator then dispatches to IRT, RT, and Analyze."
)

st.subheader("Input phase: conversation box")
if "model_settings" not in st.session_state:
    st.session_state.model_settings = _interpret_prompt("")
if "is_verified" not in st.session_state:
    st.session_state.is_verified = False
if "prompt_analyzed" not in st.session_state:
    st.session_state.prompt_analyzed = False
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

with st.form("prompt_form"):
    prompt = st.text_area(
        "Describe the analysis you want (e.g., guessing or 3PL).",
        placeholder='Hint: I think there is guessing on this test, use a better model.',
    )
    st.caption("Tip: press Ctrl+Enter to analyze")
    analyze = st.form_submit_button("Analyze prompt")
    if analyze:
        st.session_state.model_settings = _interpret_prompt(prompt)
        st.session_state.is_verified = False
        st.session_state.prompt_analyzed = True
        st.session_state.last_prompt = prompt

if st.session_state.prompt_analyzed:
    confirm_tab, = st.tabs(["Confirm settings"])
    with confirm_tab:
        feedback = st.session_state.model_settings.get("feedback", "")
        if feedback:
            st.success(f"Feedback: {feedback}")
        with st.form("confirm_settings"):
            st.markdown("**Interpretation phase: proposed settings**")
            suggestion = st.session_state.model_settings.get("suggestion", "")
            reason = st.session_state.model_settings.get("reason", "")
            note = st.session_state.model_settings.get("note", "")
            # If suggestion/reason are empty, feedback is the main message (off-topic prompt)
            if suggestion or reason:
                if reason:
                    st.info(f"Suggestion: {suggestion}  Reason: {reason}")
                elif suggestion:
                    st.info(f"Suggestion: {suggestion}")
            if note:
                st.warning(note)
            itemtype = st.selectbox(
                "Item model",
                options=["1PL", "2PL", "3PL", "4PL"],
                index=["1PL", "2PL", "3PL", "4PL"].index(
                    st.session_state.model_settings.get("itemtype", "2PL")
                ),
            )
            r_code = st.text_area(
                "R code preview",
                value=st.session_state.model_settings.get("r_code", ""),
                height=100,
            )
            confirmed = st.form_submit_button("Confirm settings")
            if confirmed:
                st.session_state.model_settings = {"itemtype": itemtype, "r_code": r_code}
                st.session_state.is_verified = True
else:
    st.info("Analyze the prompt to see settings.")

if not st.session_state.is_verified:
    st.info("Select settings and confirm to unlock IRT execution.")

col1, col2 = st.columns(2)

with col1:
    resp_file = st.file_uploader(
        "Response data (CSV)",
        type=["csv"],
        help="Item responses (e.g. rows = persons, columns = items, 0/1).",
    )

with col2:
    rt_file = st.file_uploader(
        "Response-time data (CSV)",
        type=["csv"],
        help="Response times in the same row/column layout as responses.",
    )

if resp_file is not None:
    with st.expander("Response data preview"):
        resp_file.seek(0)
        resp_df = pd.read_csv(resp_file)
        st.dataframe(resp_df.head(10), width="stretch")

if rt_file is not None:
    with st.expander("Response-time data preview"):
        rt_file.seek(0)
        rt_df = pd.read_csv(rt_file)
        st.dataframe(rt_df.head(10), width="stretch")

run = st.button("Run workflow")

if run:
    if resp_file is None or rt_file is None:
        st.error("Please upload both: response and response-time CSV.")
        st.stop()
    if not st.session_state.is_verified:
        st.error("Please confirm model settings before running the workflow.")
        st.stop()

    # Reset file position in case the preview already read the stream
    resp_file.seek(0)
    rt_file.seek(0)

    try:
        resp_df = pd.read_csv(resp_file)
        rt_df = pd.read_csv(rt_file)
    except pd.errors.EmptyDataError:
        st.error(
            "One or both files are empty or have no parseable columns. "
            "Re-upload valid CSVs and try again."
        )
        st.stop()

    resp_df = _drop_index_column(resp_df)
    rt_df = _drop_index_column(rt_df)

    try:
        resp_df = _validate_binary_responses(resp_df)
        rt_df = _coerce_numeric(rt_df, "Response-time data")
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if resp_df.shape[1] != rt_df.shape[1]:
        st.error(
            "Response and response-time files must have the same number of columns "
            f"after cleaning. Responses: {resp_df.shape[1]}, RT: {rt_df.shape[1]}."
        )
        st.stop()

    # Match graph expectations: list of dicts (DataFrame-friendly in irt_agent)
    responses = resp_df.to_dict(orient="records")
    rt_data = rt_df.to_dict(orient="records")

    payload_json = json.dumps(
        {
        "responses": responses,
        "rt_data": rt_data,
            "model_settings": st.session_state.model_settings,
            "is_verified": st.session_state.is_verified,
        },
        sort_keys=True,
    )

    with st.spinner("Running workflow…"):
        try:
            final = _run_workflow_cached(payload_json)
            st.session_state.last_result = final
            st.session_state.last_payload = payload_json
            st.success("Workflow finished.")
            _render_results(final)
        except Exception as e:
            st.exception(e)

if "last_result" in st.session_state and not run:
    st.subheader("Last results")
    _render_results(st.session_state.last_result)