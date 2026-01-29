"""
Test OpenRouter API key and free models using OPENROUTER_API_KEY from .env.
Run: python test_openrouter.py
"""
import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODELS = [
    ("Meta Llama 3.2 3B (free)", "meta-llama/llama-3.2-3b-instruct:free"),
    ("Google Gemma 3 4B (free)", "google/gemma-3-4b-it:free"),
    ("Mistral Small 3.1 24B (free)", "mistralai/mistral-small-3.1-24b-instruct:free"),
    ("Google Gemma 3 12B (free)", "google/gemma-3-12b-it:free"),
    ("Qwen Qwen3 Next 80B A3B (free)", "qwen/qwen3-next-80b-a3b-instruct:free"),
]


def test_one(api_key: str, model_id: str, label: str, timeout: int = 25) -> tuple[bool, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    body = {"model": model_id, "messages": [{"role": "user", "content": "Reply with one word: OK"}]}
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            return True, content.strip() or "(empty)"
        err_body = {}
        try:
            err_body = r.json()
        except Exception:
            pass
        msg = (err_body.get("error") or {}).get("message", r.text or f"HTTP {r.status_code}")
        if r.status_code == 402:
            msg += " — Add credits at https://openrouter.ai/credits (free models still need a positive balance)."
        return False, f"{r.status_code}: {msg}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY not set in .env. Add your key from https://openrouter.ai/keys")
        sys.exit(1)
    print("Using OPENROUTER_API_KEY from .env")
    print("Testing free models (one short request per model)...\n")
    ok_count = 0
    for label, model_id in FREE_MODELS:
        ok, out = test_one(api_key, model_id, label)
        status = "OK" if ok else "FAIL"
        if ok:
            ok_count += 1
            print(f"  [{status}] {label}")
            print(f"         Response: {out[:80]}{'...' if len(out) > 80 else ''}")
        else:
            print(f"  [{status}] {label}")
            print(f"         {out}")
        print()
    print(f"Result: {ok_count}/{len(FREE_MODELS)} models responded successfully.")
    if ok_count == 0:
        print("\nTip: 402 = Insufficient credits. Add credits at https://openrouter.ai/credits (free models require a non-negative balance).")
    sys.exit(0 if ok_count > 0 else 1)


if __name__ == "__main__":
    main()
