from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

from azure.identity import ClientSecretCredential
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.costmanagement.models import (QueryAggregation,QueryDataset,QueryDefinition,QueryGrouping,QueryTimePeriod,)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from main import INPUT_FILE, load_resources

load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "")
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
OUTPUT_FILE = Path(__file__).parent / "output" / "agent_test_response.json"
FOCUS_FILE = Path(__file__).parent / "output" / "focus_billed_costs.json"
DATABASE_FILE = Path(__file__).parent / "output" / "finops_costs.db"

class AgentState(TypedDict, total=False):
    resources: list[dict]
    response: str
    credential: Any
    azure_rows: list[dict]
    focus_records: list[dict]
    database_path: str
    historical_records: list[dict]
    inserted_records: int
    skipped_records: int

def query_ollama(
    resources: list[dict],
    focus_records: list[dict],
    historical_records: list[dict],
    database_path: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a conservative FinOps analyst. Provide practical, "
                "low-risk cloud cost optimisation advice.",
            ),
            (
                "human",
                """Analyse this cloud resource inventory and Azure cost data:
{resources}

FOCUS-formatted Azure records:
{focus_records}

Historical FOCUS records from previous months:
{historical_records}

Historical data is stored in SQLite at:
{database_path}

Identify sensible cost-optimisation opportunities. Do not recommend changes to
policy-locked resources. Think of the trade-offs as you would when advising a
team: focus on practical, low-risk changes that are worth making. For each
recommendation, explain what should change, why it would help, and the likely
monthly saving where it can be estimated. Return valid JSON with a
recommendations array and a short, clear summary. Take into consideration each 
resource's monthly cost, CPU and memory utilisation, uptime requirements and all 
the corresponding information in the database.""",
            ),
        ]
    )
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0)
    response = (prompt | llm).invoke(
        {
            "resources": json.dumps(resources, indent=2),
            "focus_records": json.dumps(focus_records, indent=2),
            "historical_records": json.dumps(historical_records, indent=2),
            "database_path": database_path,
        }
    )
    return str(response.content)

def load_inventory(state: AgentState) -> AgentState:
    resources = [resource.__dict__ for resource in load_resources(INPUT_FILE)]
    return {"resources": resources}


def analyse_inventory(state: AgentState) -> AgentState:
    return {
        "response": query_ollama(
            state["resources"],
            state["focus_records"],
            state["historical_records"],
            state["database_path"],
        )
    }

# Authenticate with Azure Entra ID using a service principal
def authenticate_service_principal(state: AgentState) -> AgentState:
    credential = ClientSecretCredential(tenant_id=AZURE_TENANT_ID,client_id=AZURE_CLIENT_ID,client_secret=AZURE_CLIENT_SECRET,)
    return {"credential": credential}

# Gather Azure cost data using the Azure Cost Management API
def gather_azure_costs(state: AgentState) -> AgentState:
    if not AZURE_SUBSCRIPTION_ID:
        raise RuntimeError("AZURE_SUBSCRIPTION_ID is not set; cannot query the Azure tenant.")

    client = CostManagementClient(state["credential"])
    scope = f"/subscriptions/{AZURE_SUBSCRIPTION_ID}"
    now = datetime.now(timezone.utc)

    query = QueryDefinition(
        type="ActualCost",
        timeframe="Custom",
        time_period=QueryTimePeriod(from_property=now - timedelta(days=30), to=now),
        dataset=QueryDataset(
            granularity="Daily",
            aggregation={"totalCost": QueryAggregation(name="PreTaxCost", function="Sum")},
            grouping=[
                QueryGrouping(type="Dimension", name="ServiceName"),
                QueryGrouping(type="Dimension", name="ResourceId"),
            ],
        ),
    )

    result = client.query.usage(scope, query)
    columns = [column.name for column in result.columns]
    azure_rows = [dict(zip(columns, row)) for row in result.rows]
    return {"azure_rows": azure_rows}


def format_focus_data(state: AgentState) -> AgentState:
    # Maps Azure Cost Management rows onto the FOCUS v1.0 (FinOps Open Cost and Usage Specification) schema.
    focus_records = [
        {
            "BilledCost": row.get("PreTaxCost", 0.0),
            "EffectiveCost": row.get("PreTaxCost", 0.0),
            "ChargePeriodStart": row.get("UsageDate"),
            "ChargeCategory": "Usage",
            "ProviderName": "Azure",
            "PublisherName": "Microsoft",
            "ServiceName": row.get("ServiceName", "Unknown"),
            "ResourceID": row.get("ResourceId", "N/A"),
            "SubscriptionID": AZURE_SUBSCRIPTION_ID,
        }
        for row in state["azure_rows"]
    ]

    FOCUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOCUS_FILE.write_text(json.dumps(focus_records, indent=2), encoding="utf-8")

    return {"focus_records": focus_records}


def store_focus_data(state: AgentState) -> AgentState:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    inserted_records = 0
    skipped_records = 0
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS focus_costs (
                id INTEGER PRIMARY KEY,
                billed_cost REAL NOT NULL,
                effective_cost REAL NOT NULL,
                charge_period_start TEXT,
                charge_category TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                publisher_name TEXT NOT NULL,
                service_name TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                subscription_id TEXT NOT NULL
            )
            """
        )
        # Insert each focus record into the database while checking for duplicates
        for record in state["focus_records"]:
            values = (
                record["BilledCost"],
                record["EffectiveCost"],
                record["ChargePeriodStart"],
                record["ChargeCategory"],
                record["ProviderName"],
                record["PublisherName"],
                record["ServiceName"],
                record["ResourceID"],
                record["SubscriptionID"],
            )
            connection.execute(
                """
                INSERT INTO focus_costs (
                    billed_cost, effective_cost, charge_period_start, charge_category,
                    provider_name, publisher_name, service_name, resource_id, subscription_id
                )
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM focus_costs
                    WHERE subscription_id = ?
                      AND resource_id = ?
                      AND charge_period_start IS ?
                      AND charge_category = ?
                )
                """,
                (*values, record["SubscriptionID"], record["ResourceID"],
                 record["ChargePeriodStart"], record["ChargeCategory"]),
            )
            if connection.execute("SELECT changes()").fetchone()[0]:
                inserted_records += 1
            else:
                skipped_records += 1
    return {
        "database_path": str(DATABASE_FILE),
        "inserted_records": inserted_records,
        "skipped_records": skipped_records,
    }

# Query historical cost data from the database
def query_historical_data(state: AgentState) -> AgentState:
    current_month = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")
    history_start = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT billed_cost,
                   effective_cost,
                   charge_period_start,
                   charge_category,
                   provider_name,
                   publisher_name,
                   service_name,
                   resource_id,
                   subscription_id
            FROM focus_costs
            WHERE subscription_id = ?
              AND charge_period_start >= ?
              AND charge_period_start < ?
            ORDER BY charge_period_start, resource_id
            """,
            (AZURE_SUBSCRIPTION_ID, history_start, current_month),
        ).fetchall()

    return {"historical_records": [dict(row) for row in rows]}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_inventory", load_inventory)
    workflow.add_node("analyse_inventory", analyse_inventory)
    workflow.add_node("authenticate_service_principal", authenticate_service_principal)
    workflow.add_node("gather_azure_costs", gather_azure_costs)
    workflow.add_node("format_focus_data", format_focus_data)
    workflow.add_node("store_focus_data", store_focus_data)
    workflow.add_node("query_historical_data", query_historical_data)
    workflow.add_edge(START, "load_inventory")
    workflow.add_edge("load_inventory", "authenticate_service_principal")
    workflow.add_edge("authenticate_service_principal", "gather_azure_costs")
    workflow.add_edge("gather_azure_costs", "format_focus_data")
    workflow.add_edge("format_focus_data", "store_focus_data")
    workflow.add_edge("store_focus_data", "query_historical_data")
    workflow.add_edge("query_historical_data", "analyse_inventory")
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
    print(f"FOCUS data saved to {FOCUS_FILE}")
    print(f"FOCUS records stored in {result['database_path']}")
    print(f"SQLite records inserted: {result['inserted_records']}")
    print(f"SQLite records skipped: {result['skipped_records']}")
    

if __name__ == "__main__":
    main()