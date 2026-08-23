from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from main import INPUT_FILE, load_resources

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
OUTPUT_FILE = Path(__file__).parent / "output" / "agent_test_response.json"


class AgentState(TypedDict, total=False):
    resources: list[dict]
    response: str


def query_ollama(resources: list[dict]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a conservative FinOps analyst. Provide practical, "
                "low-risk cloud cost optimisation advice.",
            ),
            (
                "human",
                """Analyse this cloud resource inventory:
{resources}

Identify sensible cost-optimisation opportunities. Do not recommend changes to
policy-locked resources. Think of the trade-offs as you would when advising a
team: focus on practical, low-risk changes that are worth making. For each
recommendation, explain what should change, why it would help, and the likely
monthly saving where it can be estimated. Return valid JSON with a
recommendations array and a short, clear summary.""",
            ),
        ]
    )
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0)
    response = (prompt | llm).invoke({"resources": json.dumps(resources, indent=2)})
    return str(response.content)

def load_inventory(state: AgentState) -> AgentState:
    resources = [resource.__dict__ for resource in load_resources(INPUT_FILE)]
    return {"resources": resources}


def analyse_inventory(state: AgentState) -> AgentState:
    return {"response": query_ollama(state["resources"])}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_inventory", load_inventory)
    workflow.add_node("analyse_inventory", analyse_inventory)
    workflow.add_edge(START, "load_inventory")
    workflow.add_edge("load_inventory", "analyse_inventory")
    workflow.add_edge("analyse_inventory", END)
    return workflow.compile()


def main() -> None:
    try:
        result = build_graph().invoke({})
        response = result["response"]
    except Exception as error:
        print(f"Ollama generation failed: {error}")
        sys.exit(1)

    print(response)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(response, encoding="utf-8")
    print(f"\nResponse saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()