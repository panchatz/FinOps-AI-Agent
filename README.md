# Constrained Agentic FinOps for Cloud Cost Optimization

A research prototype that uses a locally hosted large language model (LLM) to generate Azure cost-optimisation recommendations. Recommendations are checked against resource metadata and policy rules. Approved actions are simulated; the prototype does not automatically modify Azure resources.

## Features

- Reads resource inventory from a CSV file.
- Generates recommendations for VM rightsizing, shutdown scheduling, and storage-tier changes.
- Uses Llama 3.1 8B through Ollama, with LangGraph for workflow orchestration.
- Validates recommendations against configured resource and policy constraints.
- Supports terminal-based human review for reviewable recommendations.
- Stores cost and utilisation records in SQLite.
- Saves recommendation and mock-execution results as JSON.
- Includes a Jupyter notebook with a deterministic rule-based baseline for comparison.

## Project structure

```text
.
├── agent_finops.py
├── main.py
├── data/
│   └── sample_resources.csv
├── requirements-agent.txt
├── requirements-notebook.txt
├── Baseline Notebook/
│   └── Baseline.ipynb
├── output/                 # Generated files; excluded from Git
└── README.md
```
## Configure the inventory

Place the sample inventory at:

```text
data/sample_resources.csv
```

The CSV must include these columns:

```text
resource_id,team,environment,service,monthly_cost,cpu_utilization,memory_utilization,uptime_required,policy_locked,schedule_eligible,storage_tier_eligible
```

Update each script’s input path to point to this file. For example, when running from the project root, use `data/sample_resources.csv`. The baseline notebook may use a different working directory, so confirm its relative path resolves to the project’s `data` folder before running it.

Use `true` and `false` for Boolean fields. The sample workflow expects service values such as `vm` and `storage`.
## Requirements

- Python 3.10 or later
- Ollama installed and running
- Llama 3.1 8B available in Ollama
- Azure credentials and permissions only if live Azure mode is enabled

The project has separate dependency files for the agent and notebook.

### Install agent dependencies

From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-agent.txt
```

### Install notebook dependencies

```powershell
python -m pip install -r requirements-notebook.txt
```

The notebook can then be opened and run in VS Code or Jupyter.

## Set up Ollama

Install Ollama, then pull the model:

```powershell
ollama pull llama3.1:8b
```

Ensure Ollama is running before starting the agent. The project is configured to use `http://localhost:11434` and the model name `llama3.1:8b`.

## Configure the inventory

The inventory CSV must include these columns:

```text
resource_id,team,environment,service,monthly_cost,cpu_utilization,memory_utilization,uptime_required,policy_locked,schedule_eligible,storage_tier_eligible
```

The current `main.py` configuration points to:

```text
C:\temp\data\sample_resources.csv
```

Create the directory and place the CSV there, or update the input path in the relevant script. Check the notebook’s configured data path as well; its path may be relative to the notebook’s working directory.

Use `true` and `false` for Boolean fields. The sample workflow expects service values such as `vm` and `storage`.

## Run the agent

From the project directory, with the virtual environment active:

```powershell
python agent_finops.py
```

Sample-data mode is enabled by default in the current configuration. In this mode, costs are derived from the inventory rather than retrieved from Azure, and live Azure Monitor metrics are not collected. Check `agent_finops.py` before running if you have changed this configuration.

If a recommendation is routed for human review, the agent prompts in the terminal. Approval proceeds to mock execution only; it does not apply changes to Azure. Policy-locked resources and hard policy violations are blocked rather than approved through this review step.

## Run the baseline notebook

Open `Baseline Notebook/Baseline.ipynb` in VS Code or Jupyter and run its cells in order. The notebook applies deterministic rules to the sample inventory and compares its recommendations with the agent’s approved recommendations.

The baseline uses fixed savings multipliers. These are heuristic estimates, not measured or guaranteed savings.

## Output and generated files

The agent writes generated reports and data under `output/`, including recommendation JSON, simulated action results, and FOCUS-like cost records. It also uses a local SQLite database for cost and utilisation records.

Generated output files and the `.env` file are excluded from Git by `.gitignore`. Do not commit credentials or sensitive cloud data.

## Evaluation scope and limitations

The evaluation uses sample inventory and inventory-derived cost records. It compares recommendation counts, action agreement, and estimated savings against a rule-based baseline. It does not establish realised savings, recommendation accuracy in production, or effectiveness using live Azure billing and utilisation data.

The cost-record structure is project-defined and FOCUS-like; do not describe it as fully FOCUS-compliant unless it has been validated against the applicable specification.
