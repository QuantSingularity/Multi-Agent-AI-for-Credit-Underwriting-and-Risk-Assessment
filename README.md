# Multi-Agent AI for Credit Underwriting and Risk Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This repository provides a **fully implemented, reproducible multi-agent system** for automated credit underwriting and risk assessment. The system is designed to modernize the loan application process by integrating advanced machine learning models with a robust, auditable agentic workflow. The core innovation lies in the hierarchical multi-agent architecture, which ensures **regulatory compliance**, **fairness guarantees**, and **explainability** for every credit decision.

The central **Supervisor Agent** orchestrates the entire process, delegating specialized tasks to agents responsible for credit scoring, fraud detection, fairness checks, and compliance, ultimately leading to a final, justified decision.

---

## 🔑 Key Features and Compliance Focus

The system is built with a strong emphasis on regulatory and ethical requirements, making it suitable for deployment in regulated financial environments.

| Feature                             | Category              | Key Capabilities                                                                                                                                                 |
| :---------------------------------- | :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hierarchical Multi-Agent System** | Core Architecture     | Central `Supervisor` agent coordinates specialized agents for scoring, fraud, and compliance, including a negotiation loop for borderline cases.                 |
| **Fairness & Bias Mitigation**      | Ethical AI            | Implements **Reweighing** (pre-processing) and **Threshold Optimization** (post-processing) to ensure **Demographic Parity** and **Equalized Odds**.             |
| **Explainability & Auditability**   | Regulatory Compliance | Generates human-readable rationales for all decisions, including Adverse Action Notices (`code/compliance/adverse_action.py`), and maintains a full audit trail. |
| **Model Monitoring**                | MLOps                 | Dedicated module for monitoring model performance, data drift, and concept drift in a production environment.                                                    |
| **Document Processing**             | Data Ingestion        | Includes a conceptual framework for **OCR processing** of application documents (e.g., pay stubs, bank statements).                                              |
| **Reproducible Evaluation**         | Experimentation       | Dockerized environment with deterministic synthetic data generation (`code/data/synthetic_generator.py`) for consistent and verifiable results.                  |

---

## 🤖 Multi-Agent Architecture: The Underwriting Workflow

The system operates as a sequential workflow orchestrated by the `LoanSupervisor` (`code/agents/supervisor.py`), which delegates tasks to specialized agents.

| Step                              | Conceptual Agent     | Implementation Location                     | Function                                                                                                              |
| :-------------------------------- | :------------------- | :------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------- |
| **1. Application Intake**         | Document Processor   | `code/document_processing/ocr_processor.py` | Extracts and verifies data from application documents (e.g., income, employment).                                     |
| **2. Credit Scoring**             | Credit Scorer Agent  | `code/agents/credit_scorer.py`              | Calculates the **Probability of Default (PD)** and a FICO-like score using a trained ML model (e.g., LightGBM).       |
| **3. Fraud & Risk Assessment**    | Fraud Detector Agent | Logic within `supervisor.py`                | Assesses the application for potential fraud risks and red flags.                                                     |
| **4. Fairness Check**             | Fairness Agent       | `code/fairness/mitigation.py`               | Monitors the decision for bias across protected attributes and applies mitigation techniques if necessary.            |
| **5. Decision Aggregation**       | Supervisor Agent     | `code/agents/supervisor.py`                 | Aggregates scores, handles the negotiation loop for borderline cases, and applies human review gating.                |
| **6. Final Decision & Rationale** | Compliance Agent     | `code/compliance/adverse_action.py`         | Generates the final decision (Approve/Deny/Review) and the legally required Adverse Action Notice (AAN) or rationale. |

---

## 📁 Repository Structure and Component Breakdown

The repository is structured to clearly separate the agent logic, compliance modules, and evaluation framework.

### Top-Level Structure

| Path                   | Description                                                                        |
| :--------------------- | :--------------------------------------------------------------------------------- |
| `code/`                | Contains all Python source code for the agents, models, and system components.     |
| `docs/`                | Contains supplementary documentation, including `ENTERPRISE_FEATURES.md`.          |
| `figures/`             | Stores generated plots and visualizations (e.g., fairness trade-offs, ROC curves). |
| `results/`             | Output directory for experiment metrics, fairness reports, and synthetic data.     |
| `scripts/`             | Utility scripts for running the quick demo and generating figures.                 |
| `Dockerfile`           | Defines the reproducible environment for the system.                               |
| `README_ENTERPRISE.md` | Documentation for advanced, enterprise-level features.                             |

### Detailed `code/` Directory Breakdown

| Directory                   | Key File(s)                         | Detailed Function                                                                                       |
| :-------------------------- | :---------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `code/agents/`              | `supervisor.py`, `credit_scorer.py` | Core multi-agent logic and the central orchestration of the underwriting process.                       |
| `code/compliance/`          | `adverse_action.py`                 | Module for generating legally compliant Adverse Action Notices (AANs).                                  |
| `code/data/`                | `synthetic_generator.py`            | Deterministic generation of synthetic credit application data for reproducible experiments.             |
| `code/document_processing/` | `ocr_processor.py`                  | Conceptual module for Optical Character Recognition (OCR) and document data extraction.                 |
| `code/eval/`                | `experiment_runner.py`              | Main script for training models, running the agentic system, and executing the full evaluation suite.   |
| `code/fairness/`            | `mitigation.py`                     | Implementation of fairness metrics and bias mitigation techniques (reweighing, threshold optimization). |
| `code/monitoring/`          | `model_monitoring.py`               | Framework for detecting data drift, concept drift, and performance degradation in production.           |
| `code/visualization/`       | `fairness_plots.py`                 | Scripts to generate visualizations of fairness trade-offs and model performance.                        |

---

## 🚀 Quick Start

The project uses Docker to ensure a consistent and reproducible environment for all experiments.

### Prerequisites

- Docker (version 20.10+)
- Recommended: 4-core CPU, 8GB RAM

### Run Quick Integration Test (Recommended)

This runs a streamlined version of the experiment, generating synthetic data, training the model, and running a sample evaluation.

```bash
# 1. Clone repository
git clone https://github.com/quantsingularity/Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment.git
cd Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment

# 2. Build Docker image
docker build -t credit-agents .

# 3. Run quick test script
./scripts/run_quick.sh

# The script executes:
# docker run --rm -v $(pwd)/results:/app/results credit-agents python code/eval/experiment_runner.py --quick

# 4. View generated results
ls -la results/
```

### Run Full Experiment

This executes the complete experimental suite, including all fairness mitigation techniques and baselines (estimated 2-4 hours).

```bash
# Execute the full experiment runner
docker run --rm -v $(pwd)/results:/app/results credit-agents python code/eval/experiment_runner.py --full
```

---

## 🧪 Evaluation Framework

The evaluation framework is designed for rigorous, multi-faceted assessment of the system's performance and fairness.

### Baselines and Models

The agentic system (which uses LightGBM as its backend) is benchmarked against several standard models:

| Model                   | Type              | Purpose                                                             |
| :---------------------- | :---------------- | :------------------------------------------------------------------ |
| **Dummy Classifier**    | Baseline          | Measures performance against a random baseline.                     |
| **Logistic Regression** | Linear            | Simple, interpretable model for comparison.                         |
| **Decision Tree**       | Non-linear        | Rule-based model for capturing non-linear relationships.            |
| **LightGBM**            | Gradient Boosting | State-of-the-art model used as the core of the `CreditScorerAgent`. |

### Fairness Mitigation Techniques

The framework focuses on mitigating bias related to the protected attributes of **sex** and **race**.

| Technique                  | Type            | Goal                                                                                              | Implementation                |
| :------------------------- | :-------------- | :------------------------------------------------------------------------------------------------ | :---------------------------- |
| **Baseline**               | None            | Measures inherent bias in the unmitigated model.                                                  | `code/fairness/mitigation.py` |
| **Reweighing**             | Pre-processing  | Adjusts sample weights to achieve statistical parity in the training data.                        | `code/fairness/mitigation.py` |
| **Threshold Optimization** | Post-processing | Adjusts decision thresholds per group to satisfy fairness constraints (e.g., demographic parity). | `code/fairness/mitigation.py` |

---

## 📄 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.
