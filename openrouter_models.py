"""
OpenRouter free model list for Psych-MAS.
Edit this file to add or remove models; IDs must match openrouter.ai/api/v1/models.
Format: list of (display_label, model_id).
"""

OPENROUTER_FREE_MODELS = [
    ("Deepseek R1T2 Chimera (free)", "tngtech/deepseek-r1t2-chimera:free"),
    ("GLM-4.5 Air (free)", "z-ai/glm-4.5-air:free"),
    ("Deepseek R1T Chimera (free)", "tngtech/deepseek-r1t-chimera:free"),
    ("Trinity Large Preview (free)", "arcee-ai/trinity-large-preview:free"),
    ("Deepseek R1 0528 (free)", "deepseek/deepseek-r1-0528:free"),
    ("TNG R1T Chimera (free)", "tngtech/tng-r1t-chimera:free"),
    ("Nemotron 3 Nano 30B A3B (free)", "nvidia/nemotron-3-nano-30b-a3b:free"),
    ("Meta Llama 3.3 70B Instruct (free)", "meta-llama/llama-3.3-70b-instruct:free"),
    ("Google Gemma 3 27B IT (free)", "google/gemma-3-27b-it:free"),
    ("Qwen3 Coder (free)", "qwen/qwen3-coder:free"),
    ("OpenAI GPT-OSS 120B (free)", "openai/gpt-oss-120b:free"),
    ("Upstage Solar Pro 3 (free)", "upstage/solar-pro-3:free"),
    ("Trinity Mini (free)", "arcee-ai/trinity-mini:free"),
    ("OpenAI GPT-OSS 20B (free)", "openai/gpt-oss-20b:free"),
]

OPENROUTER_FREE_MODEL_IDS = [mid for _, mid in OPENROUTER_FREE_MODELS]
