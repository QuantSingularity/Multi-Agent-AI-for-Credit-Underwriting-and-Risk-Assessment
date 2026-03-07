# Multi-Agent AI for Credit Underwriting and Risk Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](requirements.txt)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A fully implemented multi-agent system for automated credit underwriting and risk assessment. A central Supervisor Agent orchestrates a pipeline of specialized agents for credit scoring, fraud detection, fairness checking, and compliance, producing a justified, auditable credit decision with regulatory-grade Adverse Action Notices.

---

## Table of Contents

- [Overview](#overview)
- [Agentic Workflow](#agentic-workflow)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [License](#license)

---

## Overview

| Feature                             | Description                                                                                                        |
| :---------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Hierarchical Multi-Agent System** | Supervisor coordinates scoring, fraud, and compliance agents with a negotiation loop for borderline cases          |
| **Fairness and Bias Mitigation**    | Reweighing (pre-processing) and Threshold Optimization (post-processing) for Demographic Parity and Equalized Odds |
| **Explainability and Audit Trail**  | Human-readable rationales and legally compliant Adverse Action Notices for every decision                          |
| **Model Monitoring**                | Production module for data drift, concept drift, and performance degradation detection                             |
| **Document Processing**             | OCR framework for extracting data from application documents such as pay stubs and bank statements                 |
| **Reproducibility**                 | Dockerized environment with deterministic synthetic data generation for consistent experiments                     |

---

## Agentic Workflow

The `LoanSupervisor` (`code/agents/supervisor.py`) orchestrates six sequential steps.

| Step                    | Agent              | Function                                                                          |
| :---------------------- | :----------------- | :-------------------------------------------------------------------------------- |
| 1. Application Intake   | Document Processor | Extracts and verifies income, employment, and identity from application documents |
| 2. Credit Scoring       | Credit Scorer      | Computes Probability of Default and a FICO-like score using LightGBM              |
| 3. Fraud Assessment     | Fraud Detector     | Assesses application for fraud risk and red flags                                 |
| 4. Fairness Check       | Fairness Agent     | Monitors for bias across protected attributes and applies mitigation if needed    |
| 5. Decision Aggregation | Supervisor         | Aggregates scores, runs negotiation loop for borderline cases, gates human review |
| 6. Final Decision       | Compliance Agent   | Issues Approve/Deny/Review decision and generates the Adverse Action Notice       |

---

## Repository Structure

| Path                        | Description                                                  |
| :-------------------------- | :----------------------------------------------------------- |
| `code/agents/`              | Supervisor and credit scorer agent logic                     |
| `code/compliance/`          | Adverse Action Notice generation                             |
| `code/fairness/`            | Fairness metrics, reweighing, threshold optimization         |
| `code/data/`                | Deterministic synthetic credit application data generator    |
| `code/document_processing/` | OCR module for application document extraction               |
| `code/eval/`                | Model training, agentic system runner, full evaluation suite |
| `code/monitoring/`          | Data drift and concept drift detection                       |
| `code/visualization/`       | Fairness trade-off and model performance plots               |
| `figures/`                  | Generated plots and visualizations                           |
| `results/`                  | Experiment metrics, fairness reports, synthetic data outputs |
| `scripts/`                  | Quick demo and figure generation scripts                     |

---

## Quick Start

### Prerequisites

- Docker 20.10+
- Recommended: 4-core CPU, 8GB RAM

```bash
git clone https://github.com/quantsingularity/Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment.git
cd Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment

docker build -t credit-agents .
./scripts/run_quick.sh
```

Generates synthetic data, trains the model, and runs a sample evaluation. Results saved to `results/`.

### Full Experiment

```bash
docker run --rm -v $(pwd)/results:/app/results credit-agents python code/eval/experiment_runner.py --full
```

Estimated runtime: 2 to 4 hours. Runs all fairness mitigation techniques and baselines.

---

## Evaluation

### Model Baselines

| Model               | Type              | Purpose                              |
| :------------------ | :---------------- | :----------------------------------- |
| Dummy Classifier    | Baseline          | Random performance reference         |
| Logistic Regression | Linear            | Simple interpretable comparison      |
| Decision Tree       | Non-linear        | Rule-based non-linear comparison     |
| **LightGBM**        | Gradient Boosting | Core model used by CreditScorerAgent |

### Fairness Mitigation Techniques

Applied across protected attributes of sex and race.

| Technique              | Type            | Goal                                                           |
| :--------------------- | :-------------- | :------------------------------------------------------------- |
| Baseline               | None            | Measures inherent bias in unmitigated model                    |
| Reweighing             | Pre-processing  | Adjusts sample weights for statistical parity in training data |
| Threshold Optimization | Post-processing | Adjusts per-group thresholds to satisfy demographic parity     |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
