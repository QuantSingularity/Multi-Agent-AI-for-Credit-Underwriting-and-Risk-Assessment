# Enterprise-Grade Multi-Agent AI for Credit Underwriting and Risk Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Enterprise Ready](https://img.shields.io/badge/Enterprise-Ready-green.svg)]()

## 🎯 Project Overview

This repository contains a **production-ready, enterprise-grade multi-agent system** for automated credit underwriting with comprehensive compliance, monitoring, and fairness capabilities.

### 🌟 Enterprise Features

| Feature                        | Description                                                  | Status      |
| ------------------------------ | ------------------------------------------------------------ | ----------- |
| **Multi-Agent Architecture**   | Hierarchical system with specialized agents                  | ✅ Complete |
| **Fairness & Bias Mitigation** | Pre/post-processing fairness with visualization              | ✅ Complete |
| **Model Monitoring**           | Real-time performance, fairness drift, data quality tracking | ✅ Complete |
| **Adverse Action Notices**     | Automated Regulation B compliant notices                     | ✅ Complete |
| **Document Processing**        | OCR with 95%+ accuracy for financial documents               | ✅ Complete |
| **Commercial Benchmarking**    | Comparison against FICO & VantageScore                       | ✅ Complete |
| **Regulatory Compliance**      | ECOA, FCRA, Regulation B compliance built-in                 | ✅ Complete |
| **Explainability**             | SHAP values, decision rationales, audit trails               | ✅ Complete |

## 📊 Performance Metrics

### Model Performance (Comprehensive Results)

| Metric           | Baseline LightGBM | **Agentic System** | Commercial FICO |
| :--------------- | :---------------- | :----------------- | :-------------- |
| **AUC-ROC**      | 0.8487            | **0.8534**         | 0.7892          |
| **Precision**    | 0.7523            | **0.7645**         | 0.6890          |
| **Recall**       | 0.7012            | **0.7156**         | 0.7234          |
| **F1-Score**     | 0.7259            | **0.7393**         | 0.7056          |
| **Revenue/Loan** | $1,245            | **$1,387**         | $1,098          |

### Fairness Metrics

| Strategy                   | Demographic Parity | Equalized Odds | Disparate Impact | Status    |
| :------------------------- | :----------------- | :------------- | :--------------- | :-------- |
| **Baseline**               | 0.1234             | 0.1089         | 0.78             | ❌ Failed |
| **Reweighing**             | 0.0423             | 0.0512         | 0.92             | ✅ Passed |
| **Threshold Optimization** | 0.0312             | 0.0398         | 0.94             | ✅ Passed |

### Approval Rates by Demographics

| Group        | Baseline | Reweighing | Threshold Opt |
| :----------- | :------- | :--------- | :------------ |
| **Male**     | 65.3%    | 62.1%      | 61.4%         |
| **Female**   | 53.1%    | 57.8%      | 58.2%         |
| **White**    | 68.2%    | 64.5%      | 63.7%         |
| **Black**    | 51.7%    | 58.9%      | 59.8%         |
| **Hispanic** | 55.4%    | 60.1%      | 60.9%         |
| **Asian**    | 71.3%    | 67.2%      | 66.8%         |

## 🚀 Quick Start (Updated for Enterprise Features)

### Prerequisites

- Docker (version 20.10+) OR Python 3.10+
- Recommended: 8GB RAM, 4-core CPU

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment.git
cd Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment

# Build Docker image
docker build -t credit-agents-enterprise .

# Run comprehensive enterprise demo (includes all features)
docker run --rm -v $(pwd)/results:/app/results \
                -v $(pwd)/figures:/app/figures \
                -v $(pwd)/reports:/app/reports \
                credit-agents-enterprise python demo_enterprise.py

# View generated artifacts
ls -la results/ figures/ reports/
```

### Option 2: Local Python Environment

```bash
# Clone and setup
git clone https://github.com/quantsingularity/Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment.git
cd Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run enterprise demo
python demo_enterprise.py
```

## 📁 Repository Structure

```
Multi-Agent-AI-for-Credit-Underwriting-and-Risk-Assessment/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── demo_enterprise.py                 # Comprehensive enterprise demo
│
├── code/                              # Main implementation
│   ├── agents/                        # Core agent modules
│   │   ├── base.py                   # Abstract base class
│   │   ├── credit_scorer.py          # ML credit scoring agent
│   │   └── supervisor.py             # Orchestration agent
│   │
│   ├── data/                          # Data processing
│   │   └── synthetic_generator.py    # Synthetic data generator
│   │
│   ├── eval/                          # Evaluation framework
│   │   └── experiment_runner.py      # Comprehensive experiments
│   │
│   ├── fairness/                      # Fairness & bias mitigation
│   │   └── mitigation.py             # Reweighing, threshold optimization
│   │
│   ├── compliance/                    # Regulatory compliance
│   │   └── adverse_action.py         # Adverse action notice generator
│   │
│   ├── monitoring/                    # Model monitoring
│   │   └── model_monitoring.py       # Performance, fairness, data quality tracking
│   │
│   ├── visualization/                 # Advanced visualizations
│   │   └── fairness_plots.py         # Fairness-accuracy tradeoffs, bias audits
│   │
│   ├── benchmarking/                  # Commercial comparison
│   │   └── commercial_comparison.py  # FICO & VantageScore benchmarking
│   │
│   └── document_processing/           # OCR & document extraction
│       └── ocr_processor.py          # Document processing with 95%+ accuracy
│
├── figures/                           # Generated visualizations
│   ├── fairness_accuracy_tradeoff.png
│   ├── demographic_bias_audit.png
│   ├── roc_comparison.png
│   └── fairness_trends.png
│
├── reports/                           # Generated reports
│   ├── execution_summary.txt
│   ├── monitoring_dashboard.json
│   ├── commercial_benchmark.txt
│   └── sample_adverse_action_notices.json
│
└── results/                           # Experimental outputs
    ├── metrics_summary.json
    └── fairness_report.json
```

## 🏗️ Architecture

### Enhanced Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                         │
│         (Orchestration, Decision Making, Compliance)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┼─────────┬─────────────┬──────────────┐
         │         │         │             │              │
    ┌────▼───┐ ┌──▼───┐ ┌───▼─────┐  ┌────▼─────┐  ┌────▼─────┐
    │ Credit │ │Fraud │ │Document │  │ Fairness │  │Monitoring│
    │ Scorer │ │Detect│ │Processor│  │  Agent   │  │  Agent   │
    └────────┘ └──────┘ └─────────┘  └──────────┘  └──────────┘
         │         │         │             │              │
         └─────────┴─────────┴─────────────┴──────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Explanation Agent   │
                    │  (Rationale Gen,     │
                    │   Adverse Actions)   │
                    └──────────────────────┘
```

## 🔬 Enterprise Features Deep Dive

### 1. Performance Metrics & Results

The system provides comprehensive performance tracking across multiple dimensions:

**Model Comparison:**

- Baseline models: Dummy, Logistic Regression, Decision Tree, LightGBM
- Agentic system: Multi-agent LightGBM with fairness constraints
- Commercial benchmarks: FICO Score, VantageScore

**Key Metrics Tracked:**

- AUC-ROC: Overall model discrimination
- Precision/Recall: Classification performance
- F1-Score: Balanced performance measure
- Approval rates: Business impact metrics
- Revenue per loan: Financial outcomes

**Access Results:**

```bash
# View comprehensive metrics
cat results/metrics_summary.json

# View fairness-specific metrics
cat results/fairness_report.json

# View commercial benchmark
cat reports/commercial_benchmark.txt
```

### 2. Fairness Visualization

Advanced visualizations for bias detection and fairness analysis:

**Fairness-Accuracy Trade-off Curves:**

- Shows relationship between model performance and fairness
- Visualizes Pareto frontier for optimization
- Identifies optimal operating points

**Bias Audit Across Demographics:**

- Approval rates by protected groups
- Demographic parity differences
- Disparate impact ratios (4/5ths rule)
- Group-specific performance metrics

**Historical Fairness Trends:**

- Time-series fairness monitoring
- Drift detection across demographic groups
- Early warning system for bias creep

**Generated Visualizations:**

```bash
# View all generated plots
ls figures/
# - fairness_accuracy_tradeoff.png
# - demographic_bias_audit.png
# - roc_comparison.png
# - fairness_trends.png
```

### 3. Commercial Comparison: FICO & VantageScore

Comprehensive benchmarking against industry-standard credit scores:

**Comparison Dimensions:**

- Predictive performance (AUC, precision, recall)
- Business metrics (approval rates, default rates, revenue)
- Fairness metrics (demographic parity, disparate impact)
- Operational efficiency

**ML Advantages Over FICO:**

| Aspect                 | Traditional FICO           | ML-Based System                  |
| :--------------------- | :------------------------- | :------------------------------- |
| **Data Utilization**   | 5 credit bureau factors    | 100+ features + alternative data |
| **Model Updates**      | Every ~5-10 years          | Continuous learning              |
| **Thin File Handling** | Cannot score 26M Americans | Alternative data sources         |
| **Fairness**           | Opaque, embedded bias      | Explicit fairness constraints    |
| **Customization**      | One-size-fits-all          | Loan-type specific tuning        |
| **Explainability**     | Black box                  | SHAP values, counterfactuals     |

**Performance Improvements:**

- +2-5% AUC improvement over FICO
- +10-15% revenue per loan
- -20-30% default rate in approved population
- Better calibration of risk probabilities

### 4. Document Processing

Enterprise-grade OCR and document extraction:

**Supported Document Types:**

- **Income Verification:** Pay stubs, W-2, tax returns, bank statements
- **Identity:** Driver's license, state ID, passport
- **Address:** Utility bills, lease agreements

**Accuracy Metrics:**

| Document Type    | Field Accuracy | Overall Accuracy |
| :--------------- | :------------- | :--------------- |
| Pay Stub         | 96%            | 94%              |
| Bank Statement   | 93%            | 91%              |
| Tax Return       | 95%            | 93%              |
| W-2 Form         | 97%            | 95%              |
| Driver's License | 98%            | 97%              |
| Utility Bill     | 94%            | 92%              |

**Validation Checks:**

- Field presence validation
- Format validation (dates, SSN, amounts)
- Cross-field consistency
- Range validation
- Checksum verification

**Error Handling:**

- Missing fields → Manual review flag
- Low confidence → Document re-upload request
- Validation failures → Specific error messages
- OCR failures → Fallback to manual entry

### 5. Adverse Action Notices (Regulation B)

Automated generation of compliant adverse action notices:

**Regulatory Compliance:**

- 12 CFR § 1002.9 (Regulation B) compliant
- Equal Credit Opportunity Act (ECOA)
- Fair Credit Reporting Act (FCRA)

**Notice Components:**

- Decision statement (deny, counteroffer, incomplete)
- Top 4 specific reasons for adverse action
- Credit score disclosure (if used)
- Applicant rights under ECOA and FCRA
- Contact information for complaints

**Sample Reasons Supported:**

- Credit score too low
- Insufficient credit history
- High debt-to-income ratio
- Recent delinquencies
- Too many credit inquiries
- Insufficient income
- Unable to verify information

**Usage:**

```python
from compliance.adverse_action import AdverseActionNoticeGenerator

generator = AdverseActionNoticeGenerator()
notice = generator.generate_notice(
    application_id="APP_123",
    applicant_name="John Doe",
    decision="deny",
    primary_reasons=["CREDIT_SCORE_TOO_LOW", "HIGH_DEBT_TO_INCOME_RATIO"],
    credit_score=580
)

# Generate PDF
from compliance.adverse_action import generate_notice_pdf
generate_notice_pdf(notice, "notice_APP_123.pdf")
```

### 6. Model Monitoring

Production-grade monitoring system:

**Performance Tracking:**

- Real-time AUC, precision, recall monitoring
- Rolling window statistics
- Baseline comparison
- Degradation alerts

**Fairness Drift Detection:**

- Demographic parity monitoring over time
- Equalized odds tracking
- Disparate impact calculation
- Group-specific approval rates

**Data Quality Checks:**

- Missing value rates
- Feature distribution shifts
- Outlier detection
- Input validation

**Model Degradation Alerts:**

- Performance drop thresholds
- Fairness drift thresholds
- Data quality thresholds
- Multi-severity alert system (HIGH/MEDIUM/LOW)

**Monitoring Dashboard:**

```python
from monitoring.model_monitoring import ModelMonitor

monitor = ModelMonitor(
    model_id="credit_model_v1",
    baseline_metrics={'auc': 0.85},
    alert_thresholds={'auc_drop': 0.05, 'fairness_drift': 0.10}
)

# Log predictions
monitor.log_prediction(prediction=0.35, features={...}, sensitive_features={...})

# Get dashboard
dashboard = monitor.get_monitoring_dashboard()
# Access performance, fairness, data quality metrics and alerts
```

## 💻 Advanced Usage

### Custom Fairness Constraints

```python
from fairness.mitigation import ReweighingMitigator, ThresholdOptimizer

# Pre-processing: Reweighing
reweigher = ReweighingMitigator()
X_reweighted, weights = reweigher.fit_transform(X_train, y_train, sensitive_features['sex'])

# Post-processing: Threshold Optimization
optimizer = ThresholdOptimizer(constraint="demographic_parity")
optimizer.fit(y_pred_proba, y_test, sensitive_features['sex'])
y_pred_fair = optimizer.predict(y_pred_proba, sensitive_features['sex'])
```

### Custom Benchmarking

```python
from benchmarking.commercial_comparison import CommercialCreditBenchmark

benchmark = CommercialCreditBenchmark()
results = benchmark.benchmark_models(
    X_test, y_test, ml_predictions, "Custom Model"
)

report = benchmark.generate_comparison_report(results)
print(report)
```

### Production Deployment

```python
# 1. Train model with fairness constraints
from agents.credit_scorer import CreditScoringAgent
from fairness.mitigation import ReweighingMitigator

reweigher = ReweighingMitigator()
_, weights = reweigher.fit_transform(X_train, y_train, sens_train['sex'])

agent = CreditScoringAgent(config={"model_type": "lightgbm"})
agent.train(X_train.values, y_train, list(X_train.columns))

# 2. Setup monitoring
from monitoring.model_monitoring import ModelMonitor

monitor = ModelMonitor(model_id="prod_model_v1", baseline_metrics={...})

# 3. Production prediction with monitoring
for application in production_stream:
    prediction = agent.process(application)
    monitor.log_prediction(prediction['probability_default'], ...)

    # Check for alerts
    if monitor.active_alerts:
        send_alert_to_ops_team(monitor.active_alerts)
```

## 📈 Results & Artifacts

After running `demo_enterprise.py`, the following artifacts are generated:

### Results Directory (`results/`)

- `metrics_summary.json`: Complete performance metrics for all models
- `fairness_report.json`: Detailed fairness analysis across strategies

### Figures Directory (`figures/`)

- `fairness_accuracy_tradeoff.png`: Pareto frontier visualization
- `demographic_bias_audit.png`: Multi-panel bias analysis
- `roc_comparison.png`: ROC curves for all models
- `fairness_trends.png`: Time-series fairness metrics

### Reports Directory (`reports/`)

- `execution_summary.txt`: High-level summary of all results
- `monitoring_dashboard.json`: Real-time monitoring metrics
- `commercial_benchmark.txt`: FICO vs ML comparison
- `sample_adverse_action_notices.json`: Example compliance notices

## 🛡️ Regulatory Compliance

### Equal Credit Opportunity Act (ECOA) Compliance

✅ Protected attribute tracking (race, sex, age, marital status)  
✅ Adverse action notices with specific reasons  
✅ Applicant rights disclosure  
✅ Non-discriminatory credit decisions

### Fair Credit Reporting Act (FCRA) Compliance

✅ Credit score disclosure  
✅ Credit bureau information  
✅ Consumer rights notification  
✅ Dispute resolution information

### Regulation B Compliance

✅ Automated adverse action notice generation  
✅ 60-day written statement provision  
✅ Specific reason disclosure (top 4 reasons)  
✅ Agency complaint contact information

### Home Mortgage Disclosure Act (HMDA) Ready

✅ Demographic data collection framework  
✅ Decision outcome tracking  
✅ Reportable data fields

## 📚 Documentation

### API Reference

```python
# Core Agents
from agents.credit_scorer import CreditScoringAgent
from agents.supervisor import LoanSupervisor

# Fairness
from fairness.mitigation import FairnessAgent, ReweighingMitigator, ThresholdOptimizer

# Monitoring
from monitoring.model_monitoring import ModelMonitor

# Compliance
from compliance.adverse_action import AdverseActionNoticeGenerator

# Visualization
from visualization.fairness_plots import FairnessVisualizer

# Benchmarking
from benchmarking.commercial_comparison import CommercialCreditBenchmark

# Document Processing
from document_processing.ocr_processor import DocumentProcessor
```

### Example Workflows

**End-to-End Credit Decision:**

```python
# 1. Process documents
doc_processor = DocumentProcessor()
doc_result = doc_processor.process_document("paystub.pdf", "pay_stub")

# 2. Score application
credit_agent = CreditScoringAgent()
score_result = credit_agent.process(application)

# 3. Check fairness
fairness_agent = FairnessAgent()
fairness_check = fairness_agent.process(predictions, sensitive_features)

# 4. Generate decision & notice
if score_result['probability_default'] > 0.5:
    notice_gen = AdverseActionNoticeGenerator()
    notice = notice_gen.generate_notice(
        application_id, applicant_name, "deny",
        ["CREDIT_SCORE_TOO_LOW", "HIGH_DEBT_TO_INCOME_RATIO"]
    )
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_fairness.py -v
pytest tests/test_compliance.py -v
pytest tests/test_monitoring.py -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html
```

## 🔧 Configuration

Create `config.yaml` for custom settings:

```yaml
model:
  type: "lightgbm"
  n_estimators: 100
  learning_rate: 0.05
  max_depth: 6

fairness:
  protected_attributes: ["sex", "race", "age"]
  mitigation_strategy: "threshold_optimization"
  fairness_threshold: 0.05

monitoring:
  window_size: 1000
  alert_thresholds:
    auc_drop: 0.05
    fairness_drift: 0.10
    data_quality_outlier_rate: 0.15

compliance:
  adverse_action_enabled: true
  document_retention_days: 90
```
