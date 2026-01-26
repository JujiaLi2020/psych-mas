import os
from dotenv import load_dotenv
import requests


def _get_key() -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GOOGLE_API_KEY in environment/.env")
    return key


def _list_models(key: str) -> list[dict]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("models", [])


def _pick_model(models: list[dict]) -> str:
    candidates = []
    for model in models:
        methods = model.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            candidates.append(model["name"])
    # Prefer a fast, text-capable model if available
    for name in candidates:
        if "gemini-1.5-flash" in name:
            return name
    for name in candidates:
        if "gemini-1.5-pro" in name:
            return name
    for name in candidates:
        if "gemini" in name:
            return name
    return candidates[0] if candidates else ""


def _test_generate(key: str, model: str) -> str:
    if not model:
        raise SystemExit("No models returned by ListModels.")
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": "ping"}]}]}
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )


def main() -> None:
    key = _get_key()
    models = _list_models(key)
    print(f"Models found: {len(models)}")
    # Set a model explicitly here if you want to override auto-pick.
    # model = "models/gemini-1.5-flash"
    model = _pick_model(models)
    print(f"Using model: {model}")
    text = _test_generate(key, model)
    print(f"Response: {text[:200]}")


if __name__ == "__main__":
    main()
