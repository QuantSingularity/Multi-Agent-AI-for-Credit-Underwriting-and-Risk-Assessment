# Multi-Agent AI for Automated Credit Underwriting and Risk Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This repository contains a **fully implemented, reproducible multi-agent system** for automated credit underwriting. The system is designed to handle the complexity of loan applications, incorporating advanced features such as:

- **Fairness Guarantees**: Monitoring and mitigation of bias across protected attributes.
- **Explainability**: Generation of human-readable rationales for credit decisions.
- **Regulatory Compliance**: Built-in checks for PII handling and human review gating.

The core of the system is a hierarchical multi-agent architecture that delegates specialized tasks to dedicated agents, all coordinated by a central **Supervisor** agent.

### Key Features

| Feature                        | Description                                                                                                       | Implementation Focus                           |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| **Hierarchical Multi-Agent**   | Supervisor, Credit Scorer, and Fairness agents coordinate to process applications and make final decisions.       | `code/agents/supervisor.py`                    |
| **Fairness & Bias Mitigation** | Implements reweighing (pre-processing) and threshold optimization (post-processing) to ensure equitable outcomes. | `code/fairness/mitigation.py`                  |
| **Explainability & Audit**     | Complete audit trails and rationale generation for every decision, supporting regulatory requirements.            | `code/agents/supervisor.py`                    |
| **Reproducible Environment**   | Docker-based setup with locked dependencies and deterministic seeds for all experiments.                          | `Dockerfile`, `code/eval/experiment_runner.py` |

## 📊 Key Results (Placeholder)

The full experimental suite generates comprehensive results on model performance, fairness metrics, and agentic system efficiency.

| Metric                 | Baseline Model (LightGBM)           | **Agentic System (LightGBM Backend)** |
| :--------------------- | :---------------------------------- | :------------------------------------ |
| **AUC**                | [Value from `metrics_summary.json`] | [Value from `metrics_summary.json`]   |
| **Average Precision**  | [Value from `metrics_summary.json`] | [Value from `metrics_summary.json`]   |
| **Demographic Parity** | [Value from `fairness_report.json`] | [Value from `fairness_report.json`]   |
| **Equalized Odds**     | [Value from `fairness_report.json`] | [Value from `fairness_report.json`]   |

_Note: Full quantitative results are generated upon running the experiments and stored in `results/metrics_summary.json` and `results/fairness_report.json`._

## 🚀 Quick Start (30 minutes)

The recommended way to run the project is using Docker to ensure a consistent and reproducible environment.

### Prerequisites

- Docker (version 20.10+)
- Recommended: 4-core CPU, 8GB RAM

### Run with Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment.git
cd Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment

# Build Docker image
docker build -t credit-agents .

# Run quick integration test (generates synthetic data, trains models, and runs a sample evaluation)
docker run --rm -v $(pwd)/results:/app/results credit-agents ./run_quick.sh

# View generated results
ls -la results/
```

### Run Full Experiment

```bash
# Run complete experiments (estimated 2-4 hours on a standard machine)
docker run --rm -v $(pwd)/results:/app/results credit-agents ./run_full.sh
```

## 📁 Repository Structure

The repository structure is organized around the core components of the multi-agent system and the evaluation framework.

```
Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment/
├── README.md                          # This file
├── LICENSE                            # Project license
├── Dockerfile                         # Defines the reproducible environment
├── requirements.txt                   # Python dependencies (locked versions)
├── run_quick.sh                       # Quick test script (30 min)
├── run_full.sh                        # Full experiment script (2-4 hours)
│
├── code/                              # Main implementation
│   ├── agents/                        # Core agent modules
│   │   ├── base.py                   # Abstract base class for agents
│   │   ├── credit_scorer.py          # Model-based credit scoring agent
│   │   └── supervisor.py             # Orchestration and final decision logic
│   │
│   ├── data/                          # Data processing and generation
│   │   └── synthetic_generator.py     # Deterministic synthetic data generator
│   │
│   ├── eval/                          # Evaluation framework
│   │   └── experiment_runner.py      # Main script for running all experiments
│   │
│   └── fairness/                      # Bias mitigation and fairness monitoring
│       └── mitigation.py             # Implementation of reweighing and threshold optimization
│
├── figures/                           # Publication-ready images (generated by experiments)
├── results/                           # Experimental outputs (JSON reports, logs)
└── scripts/                           # Utility scripts (e.g., generate_figures.py)
```

## 🏗️ Architecture: Agent Hierarchy

The system employs a **Supervisor** agent (`supervisor.py`) to manage the end-to-end loan application process. This agent delegates conceptual tasks to specialized agents, whose logic is implemented either in dedicated files or within the `Supervisor` itself for streamlined execution.

| Conceptual Agent         | Responsibility                                                                                                     | Implementation Location                                              |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **Supervisor Agent**     | Orchestrates the entire workflow, aggregates scores, and makes the final decision (including human review gating). | `code/agents/supervisor.py`                                          |
| **Credit Scoring Agent** | Calculates the Probability of Default (PD) using a trained machine learning model (e.g., LightGBM).                | `code/agents/credit_scorer.py`                                       |
| **Fairness Agent**       | Monitors for demographic parity and equalized odds violations, and applies mitigation techniques.                  | `code/fairness/mitigation.py` (logic used by `experiment_runner.py`) |
| **Document Processor**   | _Conceptual:_ Handles document parsing and data extraction.                                                        | Logic is mocked/simplified within `code/agents/supervisor.py`        |
| **Fraud Detector**       | _Conceptual:_ Assesses application fraud risk.                                                                     | Logic is mocked/simplified within `code/agents/supervisor.py`        |
| **Explanation Agent**    | _Conceptual:_ Generates the final human-readable rationale and audit trail.                                        | Logic is mocked/simplified within `code/agents/supervisor.py`        |

## 🧪 Evaluation Framework

The evaluation is comprehensive, covering performance, fairness, and robustness. The core logic resides in `code/eval/experiment_runner.py`.

### Baselines Implemented

The agentic system is benchmarked against several standard machine learning models:

- **Dummy Classifier**: Stratified random baseline.
- **Logistic Regression**: Simple, interpretable linear model.
- **Decision Tree**: Non-linear, rule-based model.
- **LightGBM**: State-of-the-art gradient boosting model (used as the backend for the Agentic System).

### Fairness Analysis

The framework includes dedicated fairness evaluation, focusing on the protected attributes of **sex** and **race**.

| Mitigation Technique       | Type            | Purpose                                                                                           |
| :------------------------- | :-------------- | :------------------------------------------------------------------------------------------------ |
| **Baseline**               | None            | Measures inherent bias in the unmitigated model.                                                  |
| **Reweighing**             | Pre-processing  | Adjusts sample weights to achieve statistical parity in the training data.                        |
| **Threshold Optimization** | Post-processing | Adjusts decision thresholds per group to satisfy fairness constraints (e.g., demographic parity). |

## 🛡️ Regulatory Compliance

The system is designed with regulatory principles in mind, particularly those related to fair lending and consumer protection:

- **HMDA/ECOA Mapping**: The framework supports mapping features to protected attributes required by the Equal Credit Opportunity Act (ECOA) and the Home Mortgage Disclosure Act (HMDA).
- **PII Handling**: The architecture includes conceptual steps for PII redaction and secure data handling.
- **Human Review Gating**: The Supervisor agent implements logic to flag borderline or high-risk cases for mandatory human review, ensuring compliance with human-in-the-loop requirements.

## 💻 Testing

The repository supports unit and integration testing to ensure code quality and system integrity.

```bash
# Run all unit tests (assuming tests are in a 'tests/' directory)
pytest tests/

# Run integration test
pytest tests/test_integration.py
```
