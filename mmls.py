"""
Model lists for Psych-MAS (OpenRouter + Google Gemini).
Edit this file to add or remove models.

OpenRouter IDs must match https://openrouter.ai/docs (model slugs).
Gemini IDs use the Generative Language API form: models/gemini-...
"""

# ----- Google Gemini (display label, API model name) -----
GEMINI_MODEL_OPTIONS = [
    ("Gemini 1.5 Flash (latest)", "models/gemini-1.5-flash-latest"),
    ("Gemini 1.5 Flash 002", "models/gemini-1.5-flash-002"),
    ("Gemini 1.5 Flash 001", "models/gemini-1.5-flash-001"),
    ("Gemini 2.0 Flash", "models/gemini-2.0-flash"),
    ("Gemini 2.5 Flash", "models/gemini-2.5-flash"),
]
DEFAULT_GEMINI_MODEL_IDS = [api_id for _, api_id in GEMINI_MODEL_OPTIONS]

# ----- OpenRouter curated US-model list (display label, model_id) -----
OPENROUTER_FREE_MODELS = [
    ("OpenAI GPT-4o Mini — $0.15/$0.60 per 1M — default low-cost report drafting model", "openai/gpt-4o-mini"),
    ("OpenAI GPT-4.1 Mini — $0.40/$1.60 per 1M — stronger governed report writing", "openai/gpt-4.1-mini"),
    ("OpenAI GPT-4.1 Nano — $0.10/$0.40 per 1M — cheap extraction, classification, and short summaries", "openai/gpt-4.1-nano"),
    ("Anthropic Claude Haiku 4.5 — $1.00/$5.00 per 1M — natural reviewer-facing language", "anthropic/claude-haiku-4.5"),
    ("OpenAI GPT-4.1 — $2.00/$8.00 per 1M — high-stakes final polish and complex cases", "openai/gpt-4.1"),
]

OPENROUTER_FREE_MODEL_IDS = [mid for _, mid in OPENROUTER_FREE_MODELS]

# ----- Local Ollama placeholders (display label, local model name) -----
LOCAL_OLLAMA_MODELS = [
    ("Local Llama 3.1 8B — Ollama — private, low-cost draft/support model", "llama3.1:8b"),
    ("Local Llama 3.3 70B — Ollama — private, higher-quality local review model", "llama3.3:70b"),
]
LOCAL_OLLAMA_MODEL_IDS = [mid for _, mid in LOCAL_OLLAMA_MODELS]
