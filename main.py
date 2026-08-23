from __future__ import annotations

import csv
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "sample_resources.csv"
OUTPUT_FILE = Path(__file__).parent / "output" / "recommendations.json"

@dataclass
class ResourceRecord:
    resource_id: str
    team: str
    environment: str
    service: str
    monthly_cost: float
    cpu_utilization: float
    memory_utilization: float
    uptime_required: str
    policy_locked: bool
    schedule_eligible: bool
    storage_tier_eligible: bool

@dataclass
class ActionCandidate:
    resource_id: str
    action: str
    estimated_monthly_saving: float
    reason: str
    risk: str
    source_rule: str

@dataclass
class VerificationResult:
    allowed: bool
    checks: list[str]

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"

def wait_for_ollama(max_retries: int = 30, timeout: int = 2) -> bool:
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
            if response.status_code == 200:
                print(f"Connected to Ollama at {OLLAMA_HOST}")
                return True
        except requests.exceptions.RequestException:
            pass
        if attempt < max_retries - 1:
            time.sleep(1)
    print(f"Failed to connect to Ollama after {max_retries} attempts")
    return False

def query_llm(prompt: str, stream: bool = False) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": stream},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "No response from LLM")
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
        return None

def enhance_candidate_with_llm(candidate: ActionCandidate, resource: ResourceRecord) -> dict:
    prompt = f"""
Analyze this cost optimization recommendation:
- Resource: {resource.resource_id} ({resource.service})
- Action: {candidate.action}
- Estimated Savings: ${candidate.estimated_monthly_saving}/month
- Current Risk Level: {candidate.risk}
- Reason: {candidate.reason}

Provide a brief risk assessment and any additional considerations (2-3 sentences).
"""
    llm_analysis = query_llm(prompt)
    return {**candidate.__dict__, "llm_analysis": llm_analysis}

def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"

def load_resources(path: Path) -> list[ResourceRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ResourceRecord(
                resource_id=row["resource_id"],
                team=row["team"],
                environment=row["environment"],
                service=row["service"],
                monthly_cost=float(row["monthly_cost"]),
                cpu_utilization=float(row["cpu_utilization"]),
                memory_utilization=float(row["memory_utilization"]),
                uptime_required=row["uptime_required"],
                policy_locked=parse_bool(row["policy_locked"]),
                schedule_eligible=parse_bool(row["schedule_eligible"]),
                storage_tier_eligible=parse_bool(row["storage_tier_eligible"]),
            )
            for row in reader
        ]