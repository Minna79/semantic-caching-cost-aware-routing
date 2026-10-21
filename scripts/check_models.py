
from dotenv import load_dotenv
import os
import requests

load_dotenv()


def ask(model, prompt):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        },
        timeout=60,
    )

    data = r.json()

    # Show the actual API error instead of raising KeyError
    if r.status_code != 200 or "error" in data:
        raise RuntimeError(
            f"Model: {model}\n"
            f"HTTP status: {r.status_code}\n"
            f"API error: {data.get('error', data)}"
        )

    if "choices" not in data:
        raise RuntimeError(f"Unexpected API response: {data}")

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


models = [
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-sonnet-5.5",
]

for m in models:
    try:
        text, usage = ask(m, "Say hi in Persian.")
        print(f"\nModel: {m}")
        print("Usage:", usage)
        print("Response:", text)
    except Exception as e:
        print(f"\nRequest failed:\n{e}")
