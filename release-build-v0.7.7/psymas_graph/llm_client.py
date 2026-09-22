"""Small hosted-LLM clients used by graph-level prompt and report nodes."""

import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    api_key: str,
    model_id: str,
    messages: list[dict],
    timeout: int = 30,
) -> tuple[str | None, str | None]:
    """Call OpenRouter chat completions and return text or an error."""
    if not model_id or not messages:
        return None, "No model or messages."
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json={"model": model_id, "messages": messages},
            timeout=timeout,
        )
        response.raise_for_status()
        message = (response.json().get("choices") or [{}])[0].get("message", {})
        text = message.get("content") or ""
        return text.strip() or "No response from model.", None
    except requests.exceptions.HTTPError as error:
        try:
            body = error.response.json() if error.response is not None else {}
            message = body.get("error", {}).get("message", str(error))
        except Exception:
            message = str(error)
        status = error.response.status_code if error.response is not None else 0
        if status == 402:
            message += " Add credits at openrouter.ai/credits (free models require a non-negative balance)."
        return None, f"{status}: {message}"
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def google_model_variants(api_key: str) -> list[str]:
    """Discover compatible Gemini models, with stable fallback identifiers."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        candidates = []
        for model in response.json().get("models", []):
            methods = model.get("supportedGenerationMethods", [])
            name = model.get("name", "")
            if "generateContent" in methods and "gemini" in name.lower():
                candidates.append(name)
        flash = [name for name in candidates if "flash" in name.lower()]
        pro = [name for name in candidates if "pro" in name.lower() and name not in flash]
        other = [name for name in candidates if name not in flash and name not in pro]
        if flash or pro or other:
            return list(dict.fromkeys(flash + pro + other))
    except Exception:
        pass
    return [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
    ]


def generate_google_text(api_key: str, prompt_text: str, model_variants: list[str]) -> str | None:
    """Generate a short psychometric-assistant response with Gemini."""
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            'The user said: "{}"\n\n'
                            "Reply in one or two short sentences only. "
                            "If they ask about IRT models (1PL, 2PL, 3PL, 4PL), say what you recommend; "
                            "if they ask something else, answer directly then add: 'This app is for "
                            "psychometric analysis (IRT); I can help with models like 1PL, 2PL, 3PL, or 4PL.'"
                        ).format(prompt_text.replace('"', '\\"'))
                    }
                ],
            }
        ]
    }
    for model_variant in model_variants:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_variant}:generateContent"
        try:
            response = requests.post(url, params={"key": api_key}, json=body, timeout=30)
            response.raise_for_status()
            text = (
                response.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text.strip() if text else None
        except Exception:
            continue
    return None
