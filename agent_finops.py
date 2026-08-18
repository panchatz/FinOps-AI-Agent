from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
DATA_FILE = Path(__file__).parent.parent / "data" / "sample_resources.csv"
OUTPUT_FILE = Path(__file__).parent / "output" / "agent_test_response.json"

def load_resources() -> list[dict[str, str]]:
    with DATA_FILE.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))

def query_ollama(resources: list[dict[str, str]]) -> str:
    prompt = f"""You are a conservative FinOps analyst. Analyse this cloud resource inventory:
{json.dumps(resources, indent=2)}
Identify sensible cost-optimisation opportunities. Do not recommend changes to
policy-locked resources. Think of the trade-offs as you would when
advising a team: focus on practical, low-risk changes that are worth making.
For each recommendation, explain what should change, why it would help, and
the likely monthly saving where it can be estimated. Return valid JSON with a
recommendations array and a short, clear summary.
"""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    
    try:
        response = query_ollama(load_resources())
    except requests.exceptions.RequestException as error:
        print(f"Ollama generation failed: {error}")
        sys.exit(1)

    print(response)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(response, encoding="utf-8")
    print(f"\nResponse saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()