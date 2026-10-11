from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from azure.identity import ClientSecretCredential
except ModuleNotFoundError:
    ClientSecretCredential = None

try:
    from azure.mgmt.costmanagement import CostManagementClient
    from azure.mgmt.costmanagement.models import (
        QueryAggregation,
        QueryDataset,
        QueryDefinition,
        QueryGrouping,
        QueryTimePeriod,
    )
except ModuleNotFoundError:
    CostManagementClient = None
    QueryAggregation = QueryDataset = QueryDefinition = QueryGrouping = QueryTimePeriod = None

try:
    from azure.monitor.query import MetricAggregationType, MetricsQueryClient
except ModuleNotFoundError:
    MetricAggregationType = None
    MetricsQueryClient = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from main import INPUT_FILE, load_resources

load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.1:8b"
AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.environ.get("AZURE_CLIENT_SECRET", "")
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
SUPPORTED_SERVICES = {
    service.strip().casefold()
    for service in os.environ.get("FINOPS_SERVICES", "Virtual Machines,Storage").split(",")
    if service.strip()
}
AKS_POOL_MARKERS = ("aks", "agentpool", "nodepool") # used for identifying Azure Kubernetes resources (and excludee them from certain operations)
OUTPUT_FILE = Path(__file__).parent / "output" / "agent_test_response.json"
FOCUS_FILE = Path(__file__).parent / "output" / "focus_billed_costs.json"
ACTION_FILE = Path(__file__).parent / "output" / "recommendation_actions.json"
DATABASE_FILE = Path("output") / "finops_costs.db"
MAX_EMPTY_RESULT_RETRIES = 2
RETRY_DELAY_SECONDS = 2
USE_SAMPLE_DATA_ONLY = True
AZURE_CPU_METRIC = os.environ.get("AZURE_CPU_METRIC", "Percentage CPU")
AZURE_MEMORY_METRIC = os.environ.get("AZURE_MEMORY_METRIC", "Available Memory Percentage")
AZURE_CPU_METRIC_NAMESPACE = os.environ.get("AZURE_CPU_METRIC_NAMESPACE", "microsoft.compute/virtualmachines")
AZURE_MEMORY_METRIC_NAMESPACE = os.environ.get("AZURE_MEMORY_METRIC_NAMESPACE", "microsoft.compute/virtualmachines")
AZURE_METRICS_LOOKBACK_DAYS = 30

class AgentState(TypedDict, total=False):
    resources: list[dict]
    response: str
    credential: Any
    azure_rows: list[dict]
    excluded_azure_rows: list[dict]
    using_sample_data: bool
    azure_metrics: list[dict]
    focus_records: list[dict]
    database_path: str
    historical_records: list[dict]
    inserted_records: int
    skipped_records: int
    azure_error: str
    retry_count: int
    recommendations: list[dict]
    approved_recommendations: list[dict]
    rejected_recommendations: list[dict]
    human_reviewed_recommendations: list[dict]
    hard_rejected_recommendations: list[dict]
    reviewable_recommendations: list[dict]
    action_results: list[dict]


# FOCUS v1.0 (FinOps Open Cost and Usage Specification) record schema.
class FocusRecord(BaseModel):
    BilledCost: float = Field(ge=0)
    EffectiveCost: float = Field(ge=0)
    ConsumedQuantity: float = Field(ge=0)
    ChargePeriodStart: str | None
    ChargeCategory: str
    ProviderName: str
    PublisherName: str
    ServiceName: str
    ResourceID: str
    SubscriptionID: str


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resource_id: str = Field(min_length=1)
    action: Literal["rightsize_vm", "schedule_shutdown", "move_to_cool_tier"]
    estimated_monthly_saving: float = Field(ge=0, allow_inf_nan=False)
    reason: str = Field(min_length=1)
    proposed_sku: str | None
    proposed_sku_reason: str
    sku_recommendation_confidence: Literal["high", "medium", "low"]

    @model_validator(mode="after")
    def validate_sku_fields(self) -> Recommendation:
        if self.action == "rightsize_vm":
            if not self.proposed_sku:
                raise ValueError("rightsize_vm recommendations require proposed_sku.")
            if not self.proposed_sku_reason:
                raise ValueError("rightsize_vm recommendations require proposed_sku_reason.")
        elif (
            self.proposed_sku is not None
            or self.proposed_sku_reason
            or self.sku_recommendation_confidence != "low"
        ):
            raise ValueError("Non-VM recommendations must not include a SKU proposal.")
        return self


def validate_focus_records(records: list[dict]) -> list[dict]:
    try:
        return [FocusRecord(**record).model_dump() for record in records]
    except ValidationError as error:
        raise ValueError(f"Invalid FOCUS record: {error}") from error


def empty_cost_data_state() -> AgentState:
    return {
        "focus_records": [],
        "historical_records": [],
        "database_path": "",
        "inserted_records": 0,
        "skipped_records": 0,
    }


def ensure_database() -> None:
    """Create the SQLite database and schema if they are missing."""
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS focus_costs (
                id INTEGER PRIMARY KEY,
                billed_cost REAL NOT NULL,
                effective_cost REAL NOT NULL,
                consumed_quantity REAL NOT NULL DEFAULT 0,
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
        focus_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(focus_costs)")
        }
        if "consumed_quantity" not in focus_columns:
            connection.execute(
                "ALTER TABLE focus_costs ADD COLUMN consumed_quantity REAL NOT NULL DEFAULT 0"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_utilization (
                resource_id TEXT PRIMARY KEY,
                cpu_utilization REAL,
                memory_utilization REAL,
                recorded_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]: row[3]
            for row in connection.execute("PRAGMA table_info(resource_utilization)")
        }
        if columns["cpu_utilization"] or columns["memory_utilization"]:
            connection.execute("ALTER TABLE resource_utilization RENAME TO resource_utilization_old")
            connection.execute(
                """
                CREATE TABLE resource_utilization (
                    resource_id TEXT PRIMARY KEY,
                    cpu_utilization REAL,
                    memory_utilization REAL,
                    recorded_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO resource_utilization (
                    resource_id, cpu_utilization, memory_utilization, recorded_at
                ) SELECT resource_id, cpu_utilization, memory_utilization, recorded_at
                FROM resource_utilization_old
                """
            )
            connection.execute("DROP TABLE resource_utilization_old")


def query_ollama(
    resources: list[dict],
    focus_records: list[dict],
    historical_records: list[dict],
    azure_metrics: list[dict],
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

Live Azure Monitor metrics, averaged across the last 30 days:
{azure_metrics}

Historical data is stored in SQLite at:
{database_path}

Identify sensible cost-optimisation opportunities. Do not recommend changes to
policy-locked resources. Think of the trade-offs as you would when advising a
team: focus on practical, low-risk changes that are worth making. For each
recommendation, explain what should change, why it would help, and the likely
monthly saving where it can be estimated. Return valid JSON with a
recommendations array and a short, clear summary. Take into consideration each 
resource's monthly cost, CPU and memory utilisation, uptime requirements and all 
the corresponding information in the database. Prefer matching live Azure Monitor
metrics over local inventory utilisation values. Do not treat missing or null
Azure Monitor values as low utilisation, and do not recommend rightsizing when
live CPU or memory metrics are unavailable.

For any recommendation where action is "rightsize_vm", include:
- proposed_sku: a suggested Azure VM SKU such as Standard_B2s, Standard_B4ms, Standard_D4s_v5, or Standard_D8s_v5
- proposed_sku_reason: a brief explanation for the proposed SKU
- sku_recommendation_confidence: one of "high", "medium", or "low"

For non-VM actions, set proposed_sku to null, proposed_sku_reason to an empty string, and sku_recommendation_confidence to "low".

Each recommendation must include resource_id, action (rightsize_vm, schedule_shutdown, or move_to_cool_tier), estimated_monthly_saving, reason, proposed_sku, proposed_sku_reason, and sku_recommendation_confidence. Return JSON only.""",
            ),
        ]
    )
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0)
    response = (prompt | llm).invoke(
        {
            "resources": json.dumps(resources, indent=2),
            "focus_records": json.dumps(focus_records, indent=2),
            "historical_records": json.dumps(historical_records, indent=2),
            "azure_metrics": json.dumps(azure_metrics, indent=2),
            "database_path": database_path,
        }
    )
    return str(response.content)

# Load the resource inventory from the input file
def load_inventory(state: AgentState) -> AgentState:
    resources = [resource.__dict__ for resource in load_resources(INPUT_FILE)]
    return {"resources": resources}


def enrich_resources_with_azure_metrics(resources: list[dict], azure_metrics: list[dict]) -> list[dict]:
    """Use live Azure Monitor values when the inventory and Azure IDs match."""
    metrics_by_id = {
        metric["resource_id"].casefold(): metric
        for metric in azure_metrics
    }
    enriched_resources = []
    for resource in resources:
        metric = metrics_by_id.get(str(resource["resource_id"]).casefold())
        if metric is None:
            enriched_resources.append(resource)
            continue
        enriched_resources.append(
            {
                **resource,
                "cpu_utilization": metric.get("cpu_utilization") if metric.get("cpu_utilization") is not None else resource["cpu_utilization"],
                "memory_utilization": metric.get("memory_utilization") if metric.get("memory_utilization") is not None else resource["memory_utilization"],
                "utilization_source": "azure_monitor",
            }
        )
    return enriched_resources

# Analyse the resource inventory and Azure cost data using the LLM
def analyse_inventory(state: AgentState) -> AgentState:
    resources = enrich_resources_with_azure_metrics(
        state["resources"],
        state.get("azure_metrics", []),
    )
    return {
        "response": query_ollama(
            resources,
            state["focus_records"],
            state["historical_records"],
            state.get("azure_metrics", []),
            state["database_path"],
        )
    }

# Parse the LLM output to extract recommendations
def parse_llm_output(state: AgentState) -> AgentState:
    response = state["response"]
    start = response.find("{")
    if start == -1:
        raise ValueError("The LLM response does not contain a JSON object.")

    payload = json.JSONDecoder().raw_decode(response[start:])[0]
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("The LLM response does not contain a recommendations array.")
    try:
        validated_recommendations = [
            Recommendation.model_validate(recommendation).model_dump()
            for recommendation in recommendations
        ]
    except ValidationError as error:
        raise ValueError(f"Invalid recommendation output: {error}") from error
    return {"recommendations": validated_recommendations}


def azure_resource_name(resource_id: str) -> str:
    """Return the resource name from an Azure Resource Manager resource ID."""
    return resource_id.rstrip("/").rsplit("/", 1)[-1]


def is_dev_or_test_vm(resource: dict | None, azure_rows: list[dict]) -> bool:
    """Approve the name heuristic only when Azure confirms the exact resource ID."""
    if resource is None or resource.get("service") != "vm":
        return False

    resource_id = str(resource.get("resource_id", "")).casefold()
    for row in azure_rows:
        azure_id = str(row.get("ResourceId", ""))
        if azure_id.casefold() != resource_id:
            continue
        azure_name = azure_resource_name(azure_id).casefold()
        return "test" in azure_name or "dev" in azure_name
    return False


def is_aks_pool_resource(row: dict) -> bool:
    """Exclude Azure VM and disk resources identifiable as Azure Kubernetes pool resources."""
    resource_id = str(row.get("ResourceId", "")).casefold()
    is_vm_scale_set_or_disk = (
        "/virtualmachinescalesets/" in resource_id
        or "/disks/" in resource_id
    )
    return is_vm_scale_set_or_disk and any(marker in resource_id for marker in AKS_POOL_MARKERS)


def validate_policy_constraints(state: AgentState) -> AgentState:
    """Validate policy constraints for the recommended actions."""
    resources = {resource["resource_id"]: resource for resource in state["resources"]}
    approved = []
    hard_rejected = []
    reviewable = []

    for recommendation in state["recommendations"]:
        resource_id = recommendation.get("resource_id")
        action = recommendation.get("action")
        resource = resources.get(resource_id)
        violations = []
        hard_violations = []
        dev_test_vm_approved = is_dev_or_test_vm(resource, state.get("azure_rows", []))

        if resource is None:
            hard_violations.append("Resource is not present in the approved inventory.")
        else:
            if resource["policy_locked"]:
                hard_violations.append("Resource is policy-locked.")
            if action == "schedule_shutdown":
                if resource["service"] != "vm":
                    hard_violations.append("Shutdown scheduling applies only to VMs.")
                if not dev_test_vm_approved and resource["environment"] == "prod":
                    violations.append("Production resources cannot be scheduled for shutdown.")
                if not dev_test_vm_approved and resource["uptime_required"] == "24x7":
                    violations.append("24x7 resources cannot be scheduled for shutdown.")
                if not resource["schedule_eligible"]:
                    violations.append("Resource is not schedule-eligible.")
            elif action == "rightsize_vm":
                if resource["service"] != "vm":
                    hard_violations.append("Rightsizing applies only to VMs.")
                if not dev_test_vm_approved and (resource["cpu_utilization"] >= 25 or resource["memory_utilization"] >= 40):
                    violations.append("Resource does not meet the low-utilisation threshold.")
            elif action == "move_to_cool_tier":
                if resource["service"] != "storage":
                    hard_violations.append("Storage tiering applies only to storage resources.")
                if not resource["storage_tier_eligible"]:
                    hard_violations.append("Resource is not storage-tier eligible.")
            else:
                hard_violations.append("Action is not an approved FinOps action.")

        audited_recommendation = {
            **recommendation,
            "policy_violations": [*hard_violations, *violations],
            "hard_policy_violations": hard_violations,
        }
        if hard_violations:
            hard_rejected.append({
                **audited_recommendation,
                "status": "blocked",
                "human_review": "not_permitted",
            })
        elif violations:
            reviewable.append({
                **audited_recommendation,
                "status": "requires_human_review",
            })
        else:
            approved.append({**audited_recommendation, "status": "approved"})

    return {
        "approved_recommendations": approved,
        "rejected_recommendations": [*hard_rejected, *reviewable],
        "hard_rejected_recommendations": hard_rejected,
        "reviewable_recommendations": reviewable,
    }


def review_rejected_recommendations(state: AgentState) -> AgentState:
    """Let a human approve or reject recommendations blocked by policy checks."""
    approved = list(state["approved_recommendations"])
    hard_rejected = list(state.get("hard_rejected_recommendations", []))
    reviewed = []

    for recommendation in state.get("reviewable_recommendations", []):
        print("\nRecommendation requires human review:")
        print(json.dumps(recommendation, indent=2))
        decision = input("Approve this recommendation anyway? [y/N]: ").strip().casefold()
        human_approved = decision in {"y", "yes"}
        reviewed_recommendation = {
            **recommendation,
            "human_review": "approved" if human_approved else "rejected",
            "status": "approved" if human_approved else "rejected",
        }
        reviewed.append(reviewed_recommendation)
        if human_approved:
            approved.append(reviewed_recommendation)

    return {
        "approved_recommendations": approved,
        "rejected_recommendations": hard_rejected + [
            recommendation
            for recommendation in reviewed
            if recommendation["human_review"] == "rejected"
        ],
        "human_reviewed_recommendations": reviewed,
    }


def execute_mock_actions(state: AgentState) -> AgentState:
    """Simulate approved actions without actually making any changes"""
    action_descriptions = {
        "rightsize_vm": lambda recommendation: (
            f"Would resize the VM to {recommendation.get('proposed_sku')}."
            if recommendation.get("proposed_sku")
            else "Cannot simulate VM resizing because no proposed SKU was provided."
        ),
        "schedule_shutdown": lambda recommendation: "Would apply the approved VM shutdown schedule.",
        "move_to_cool_tier": lambda recommendation: "Would move the storage resource to the cool access tier.",
    }
    action_results = []
    for recommendation in state.get("approved_recommendations", []):
        action = recommendation.get("action")
        describe_action = action_descriptions.get(action)
        result = {
            "resource_id": recommendation.get("resource_id"),
            "action": action,
            "execution_mode": "mock",
            "execution_status": "simulated" if describe_action else "not_supported",
            "details": (
                describe_action(recommendation)
                if describe_action
                else "No mock executor is defined for this action."
            ),
        }
        action_results.append(result)

    ACTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTION_FILE.write_text(json.dumps(action_results, indent=2), encoding="utf-8")
    return {"action_results": action_results}

def save_audited_report(state: AgentState) -> AgentState:
    """Save the audited report to a JSON file."""
    report = {
        "approved_recommendations": state["approved_recommendations"],
        "rejected_recommendations": state["rejected_recommendations"],
        "action_results": state.get("action_results", []),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {}

def authenticate_service_principal(state: AgentState) -> AgentState:
    """Authenticate with Azure Entra ID using a service principal."""
    if USE_SAMPLE_DATA_ONLY:
        return {"credential": None}
    if ClientSecretCredential is None:
        raise RuntimeError("Azure SDK is not installed. Enable Azure or set USE_SAMPLE_DATA_ONLY=True.")
    credential = ClientSecretCredential(tenant_id=AZURE_TENANT_ID,client_id=AZURE_CLIENT_ID,client_secret=AZURE_CLIENT_SECRET,)
    return {"credential": credential}


def average_azure_metric(
    client: MetricsQueryClient,
    resource_id: str,
    metric_name: str,
    metric_namespace: str,
) -> float | None:
    result = client.query_resource(
        resource_id,
        metric_names=[metric_name],
        timespan=timedelta(days=AZURE_METRICS_LOOKBACK_DAYS),
        granularity=timedelta(hours=1),
        aggregations=[MetricAggregationType.AVERAGE],
        metric_namespace=metric_namespace,
    )
    values = [
        data.average
        for metric in result.metrics
        for timeseries in metric.timeseries
        for data in timeseries.data
        if data.average is not None
    ]
    return sum(values) / len(values) if values else None


def collect_azure_metrics(state: AgentState) -> AgentState:
    """Collect 30-day average VM CPU and memory usage from Azure Monitor."""
    if USE_SAMPLE_DATA_ONLY:
        return {"azure_metrics": []}
    if MetricsQueryClient is None:
        return {"azure_metrics": []}

    metrics = []
    client = MetricsQueryClient(state["credential"])
    resource_ids = {
        str(row.get("ResourceId"))
        for row in state.get("azure_rows", [])
        if str(row.get("ResourceId", "")).casefold().find("/virtualmachines/") >= 0
    }

    for resource_id in resource_ids:
        cpu = None
        memory = None
        try:
            cpu = average_azure_metric(
                client,
                resource_id,
                AZURE_CPU_METRIC,
                AZURE_CPU_METRIC_NAMESPACE,
            )
        except Exception as error:
            print(f"Azure Monitor metric '{AZURE_CPU_METRIC}' unavailable for {resource_id}: {error}")
        try:
            available_memory = average_azure_metric(
                client,
                resource_id,
                AZURE_MEMORY_METRIC,
                AZURE_MEMORY_METRIC_NAMESPACE,
            )
            memory = 100 - available_memory if available_memory is not None else None
        except Exception as error:
            print(f"Azure Monitor metric '{AZURE_MEMORY_METRIC}' unavailable for {resource_id}: {error}")

        if cpu is None:
            print(f"Azure Monitor metric '{AZURE_CPU_METRIC}' returned no data for {resource_id}.")
        if memory is None:
            print(f"Azure Monitor metric '{AZURE_MEMORY_METRIC}' returned no data for {resource_id}.")

        metrics.append(
            {
                "resource_id": resource_id,
                "cpu_utilization": cpu,
                "memory_utilization": memory,
                "source": "azure_monitor",
            }
        )

    return {"azure_metrics": metrics}

def gather_azure_costs(state: AgentState) -> AgentState:
    """Gather Azure cost data using the Azure Cost Management API."""
    if USE_SAMPLE_DATA_ONLY:
        rows = sample_cost_rows(state["resources"])
        return {"azure_rows": rows, "excluded_azure_rows": [], "using_sample_data": True}
    if CostManagementClient is None:
        return {"azure_rows": [], "azure_error": "Azure Cost Management SDK is not installed; running sample-data-only mode."}

    if not AZURE_SUBSCRIPTION_ID:
        return {"azure_rows": [], "azure_error": "AZURE_SUBSCRIPTION_ID is not set; cannot query the Azure tenant."}

    scope = f"/subscriptions/{AZURE_SUBSCRIPTION_ID}"
    now = datetime.now(timezone.utc)

    query = QueryDefinition(
        type="ActualCost",
        timeframe="Custom",
        time_period=QueryTimePeriod(from_property=now - timedelta(days=30), to=now),
        dataset=QueryDataset(
            granularity="Daily",
            aggregation={
                "totalCost": QueryAggregation(name="PreTaxCost", function="Sum"),
                "totalUsageQuantity": QueryAggregation(name="UsageQuantity", function="Sum"),
            },
            grouping=[
                QueryGrouping(type="Dimension", name="ServiceName"),
                QueryGrouping(type="Dimension", name="ResourceId"),
            ],
        ),
    )

    try:
        client = CostManagementClient(state["credential"])
        result = client.query.usage(scope, query)
    except Exception as error:
        return {"azure_rows": [], "azure_error": f"Azure Cost Management query failed: {error}"}

    columns = [column.name for column in result.columns]
    azure_rows = []
    excluded_azure_rows = []
    for row in result.rows:
        azure_row = dict(zip(columns, row))
        if str(azure_row.get("ServiceName", "")).casefold() not in SUPPORTED_SERVICES:
            continue
        if is_aks_pool_resource(azure_row):
            excluded_azure_rows.append(azure_row)
            continue
        azure_rows.append(azure_row)
    print(f"Excluded AKS pool cost rows: {len(excluded_azure_rows)}")
    return {"azure_rows": azure_rows, "excluded_azure_rows": excluded_azure_rows}


# Route after gather_azure_costs based on whether the query failed or returned no data.
def route_after_gather_azure_costs(state: AgentState) -> str:
    if USE_SAMPLE_DATA_ONLY:
        return "format_focus_data"
    if state.get("azure_error"):
        return "handle_azure_failure"
    if not state["azure_rows"]:
        if state.get("retry_count", 0) < MAX_EMPTY_RESULT_RETRIES:
            return "retry_gather_azure_costs"
        return "handle_empty_cost_data"
    return "format_focus_data"


# retry the Azure query after an empty result, waiting briefly before each attempt.
def retry_gather_azure_costs(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count", 0) + 1
    print(f"No cost data returned; retrying Azure query (attempt {retry_count} of {MAX_EMPTY_RESULT_RETRIES})...")
    time.sleep(RETRY_DELAY_SECONDS)
    return {"retry_count": retry_count}



def handle_azure_failure(state: AgentState) -> AgentState:
    print(f"Azure query failure: {state['azure_error']}")
    return {
        "azure_rows": sample_cost_rows(state["resources"]),
        "using_sample_data": True,
    }



def handle_empty_cost_data(state: AgentState) -> AgentState:
    print("Azure Cost Management returned no cost data after all retry attempts.")
    return {
        "azure_rows": sample_cost_rows(state["resources"]),
        "using_sample_data": True,
    }


def sample_cost_rows(resources: list[dict]) -> list[dict]:
    """Convert the local inventory into cost rows when Azure cost data is unavailable."""
    usage_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    service_names = {"vm": "Virtual Machines", "storage": "Storage"}
    return [
        {
            "PreTaxCost": resource["monthly_cost"],
            "UsageQuantity": 1.0,
            "UsageDate": usage_date,
            "ServiceName": service_names[resource["service"]],
            "ResourceId": resource["resource_id"],
        }
        for resource in resources
        if resource["service"] in service_names
    ]


def normalize_usage_date(value: Any) -> str | None:
    """Convert Azure UsageDate values to ISO date text."""
    if value is None:
        return None

    date_text = str(value)
    if len(date_text) == 8 and date_text.isdigit():
        return datetime.strptime(date_text, "%Y%m%d").date().isoformat()
    return date_text


# Maps Azure Cost Management rows onto the FOCUS v1.0 (FinOps Open Cost and Usage Specification) schema.
def format_focus_data(state: AgentState) -> AgentState:
    records = [
        {
            "BilledCost": row.get("PreTaxCost", 0.0),
            "EffectiveCost": row.get("PreTaxCost", 0.0),
            "ConsumedQuantity": row.get("UsageQuantity", 0.0),
            "ChargePeriodStart": normalize_usage_date(row.get("UsageDate")),
            "ChargeCategory": "Usage",
            "ProviderName": "Sample" if state.get("using_sample_data") else "Azure",
            "PublisherName": "Microsoft",
            "ServiceName": row.get("ServiceName", "Unknown"),
            "ResourceID": row.get("ResourceId", "N/A"),
            "SubscriptionID": "sample" if state.get("using_sample_data") else AZURE_SUBSCRIPTION_ID,
        }
        for row in state["azure_rows"]
    ]
    validated_records = validate_focus_records(records)

    FOCUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FOCUS_FILE.write_text(json.dumps(validated_records, indent=2), encoding="utf-8")

    return {"focus_records": validated_records}


def store_focus_data(state: AgentState) -> AgentState:
    ensure_database()
    inserted_records = 0
    skipped_records = 0
    with sqlite3.connect(DATABASE_FILE) as connection:
        recorded_at = datetime.now(timezone.utc).isoformat()
        utilization_by_id = {
            metric["resource_id"]: metric
            for metric in state.get("azure_metrics", [])
        }
        for resource in state.get("resources", []):
            metric = utilization_by_id.get(resource["resource_id"], resource)
            connection.execute(
                """
                INSERT INTO resource_utilization (
                    resource_id, cpu_utilization, memory_utilization, recorded_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(resource_id) DO UPDATE SET
                    cpu_utilization = excluded.cpu_utilization,
                    memory_utilization = excluded.memory_utilization,
                    recorded_at = excluded.recorded_at
                """,
                (
                    resource["resource_id"],
                    metric.get("cpu_utilization") if metric.get("cpu_utilization") is not None else resource["cpu_utilization"],
                    metric.get("memory_utilization") if metric.get("memory_utilization") is not None else resource["memory_utilization"],
                    recorded_at,
                ),
            )

        for metric in state.get("azure_metrics", []):
            if not any(resource["resource_id"] == metric["resource_id"] for resource in state.get("resources", [])):
                connection.execute(
                    """
                    INSERT INTO resource_utilization (
                        resource_id, cpu_utilization, memory_utilization, recorded_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(resource_id) DO UPDATE SET
                        cpu_utilization = excluded.cpu_utilization,
                        memory_utilization = excluded.memory_utilization,
                        recorded_at = excluded.recorded_at
                    """,
                    (
                        metric["resource_id"],
                        metric["cpu_utilization"],
                        metric["memory_utilization"],
                        recorded_at,
                    ),
                )

        # Insert each focus record into the database while checking for duplicates
        for record in state["focus_records"]:
            values = (
                record["BilledCost"],
                record["EffectiveCost"],
                record["ConsumedQuantity"],
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
                    billed_cost, effective_cost, consumed_quantity, charge_period_start, charge_category,
                    provider_name, publisher_name, service_name, resource_id, subscription_id
                )
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
    subscription_id = "sample" if state.get("using_sample_data") else AZURE_SUBSCRIPTION_ID

    ensure_database()
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT billed_cost AS BilledCost,
                   effective_cost AS EffectiveCost,
                     consumed_quantity AS ConsumedQuantity,
                   charge_period_start AS ChargePeriodStart,
                   charge_category AS ChargeCategory,
                   provider_name AS ProviderName,
                   publisher_name AS PublisherName,
                   service_name AS ServiceName,
                   resource_id AS ResourceID,
                   subscription_id AS SubscriptionID
            FROM focus_costs
            WHERE subscription_id = ?
              AND charge_period_start >= ?
              AND charge_period_start < ?
            ORDER BY charge_period_start, resource_id
            """,
            (subscription_id, history_start, current_month),
        ).fetchall()

    return {"historical_records": validate_focus_records([dict(row) for row in rows])}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_inventory", load_inventory)
    workflow.add_node("analyse_inventory", analyse_inventory)
    workflow.add_node("parse_llm_output", parse_llm_output)
    workflow.add_node("validate_policy_constraints", validate_policy_constraints)
    workflow.add_node("review_rejected_recommendations", review_rejected_recommendations)
    workflow.add_node("execute_mock_actions", execute_mock_actions)
    workflow.add_node("save_audited_report", save_audited_report)
    workflow.add_node("authenticate_service_principal", authenticate_service_principal)
    workflow.add_node("gather_azure_costs", gather_azure_costs)
    workflow.add_node("collect_azure_metrics", collect_azure_metrics)
    workflow.add_node("retry_gather_azure_costs", retry_gather_azure_costs)
    workflow.add_node("handle_azure_failure", handle_azure_failure)
    workflow.add_node("handle_empty_cost_data", handle_empty_cost_data)
    workflow.add_node("format_focus_data", format_focus_data)
    workflow.add_node("store_focus_data", store_focus_data)
    workflow.add_node("query_historical_data", query_historical_data)
    workflow.add_edge(START, "load_inventory")
    workflow.add_edge("load_inventory", "authenticate_service_principal")
    workflow.add_edge("authenticate_service_principal", "gather_azure_costs")
    workflow.add_edge("gather_azure_costs", "collect_azure_metrics")
    workflow.add_conditional_edges(
        "collect_azure_metrics",
        route_after_gather_azure_costs,
        {
            "handle_azure_failure": "handle_azure_failure",
            "retry_gather_azure_costs": "retry_gather_azure_costs",
            "handle_empty_cost_data": "handle_empty_cost_data",
            "format_focus_data": "format_focus_data",
        },
    )
    workflow.add_edge("retry_gather_azure_costs", "gather_azure_costs")
    workflow.add_edge("handle_azure_failure", "format_focus_data")
    workflow.add_edge("handle_empty_cost_data", "format_focus_data")
    workflow.add_edge("format_focus_data", "store_focus_data")
    workflow.add_edge("store_focus_data", "query_historical_data")
    workflow.add_edge("query_historical_data", "analyse_inventory")
    workflow.add_edge("analyse_inventory", "parse_llm_output")
    workflow.add_edge("parse_llm_output", "validate_policy_constraints")
    workflow.add_edge("validate_policy_constraints", "review_rejected_recommendations")
    workflow.add_edge("review_rejected_recommendations", "execute_mock_actions")
    workflow.add_edge("execute_mock_actions", "save_audited_report")
    workflow.add_edge("save_audited_report", END)
    return workflow.compile()


def main() -> None: 
    try:
        result = build_graph().invoke({})
        response = result["response"]
    except Exception as error:
        print(f"Ollama generation failed: {error}")
        sys.exit(1)

    print(response)
    print(f"\nAudited recommendations saved to {OUTPUT_FILE}")
    print(f"Approved recommendations: {len(result['approved_recommendations'])}")
    print(f"Rejected recommendations: {len(result['rejected_recommendations'])}")
    print(f"Mock actions simulated: {sum(action['execution_status'] == 'simulated' for action in result['action_results'])}")
    print(f"Mock action results saved to {ACTION_FILE}")
    print(f"FOCUS data saved to {FOCUS_FILE}")
    print(f"FOCUS records stored in {result['database_path']}")
    print(f"SQLite records inserted: {result['inserted_records']}")
    print(f"SQLite records skipped: {result['skipped_records']}")
    

if __name__ == "__main__":
    main()