# Multi-Agent AI for Automated Credit Underwriting and Risk Assessment

## Complete Research Implementation

This repository contains a fully implemented, reproducible multi-agent system for automated credit underwriting with fairness guarantees, explainability, and regulatory compliance.

## Quick Start (30-minute review mode)

```bash
# Build Docker environment
docker build -t credit-agents .

# Run quick integration test (≤30 minutes on 4-core CPU)
docker run --rm credit-agents ./run_quick.sh

# View results
docker run --rm -v $(pwd)/results:/app/results credit-agents ls -la /app/results
```

## Full Experiment

```bash
# Run complete experiments (estimated 2-4 hours on 4-core CPU)
docker run --rm -v $(pwd)/results:/app/results credit-agents ./run_full.sh
```

## Repository Structure

```
├── code/                    # Main implementation
│   ├── agents/             # Multi-agent modules
│   ├── orchestrator/       # Supervisor logic
│   ├── models/             # Credit scoring models
│   ├── data/               # Data processing
│   ├── ocr/                # Document processing
│   ├── fairness/           # Bias mitigation
│   ├── eval/               # Evaluation framework
│   ├── ui/                 # CLI interface
│   └── tests/              # Unit and integration tests
├── data/                   # Dataset fetchers and preprocessors
├── figures/                # Publication-ready images (generated)
├── results/                # Experimental outputs
├── ethics/                 # Compliance documentation
├── CI/                     # GitHub Actions workflows
├── Dockerfile              # Reproducible environment
├── requirements.txt        # Python dependencies (locked versions)
├── run_quick.sh            # Quick test script
├── run_full.sh             # Full experiment script
```

## System Requirements

- **Quick mode**: 4-core CPU, 8GB RAM, ~5GB disk
- **Full mode**: 8-core CPU (or 4-core with GPU), 16GB RAM, ~20GB disk
- **OS**: Linux (Ubuntu 20.04+), macOS, or WSL2 on Windows
- **Docker**: 20.10+

## Estimated Compute & Cost

- **Quick integration test**: 20-30 minutes, $0.00 (local compute)
- **Full experiment**: 2-4 hours, $0.00 (local) or ~$2-5 on cloud (AWS t3.xlarge)

## Key Features

✅ **Hierarchical Multi-Agent Architecture**: Supervisor, Document Processor, Income Verifier, Credit Scoring, Fraud Detection, Fairness, Explanation & Audit agents

✅ **Real Experimental Results**: All metrics from actual runs on LendingClub + deterministic synthetic data

✅ **Fairness & Bias Mitigation**: Implemented reweighing, constrained optimization, demographic parity/equalized odds monitoring

✅ **Explainability & Audit**: Complete audit trails, LIME/SHAP explanations, human-readable rationales

✅ **Reproducible Environment**: Docker + locked dependencies + deterministic seeds

✅ **Regulatory Compliance**: HMDA/ECOA mapping, PII redaction, human review gating

✅ **Publication-Ready Artifacts**: 2 papers (ML + Industry), 5+ figures, statistical tests

## Running Experiments

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t credit-agents .

# Quick test (30 min)
docker run --rm -v $(pwd)/results:/app/results credit-agents ./run_quick.sh

# Full experiment
docker run --rm -v $(pwd)/results:/app/results credit-agents ./run_full.sh
```

### Option 2: Local Python Environment

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run quick test
bash run_quick.sh

# Run full experiment
bash run_full.sh
```

## Viewing Results

After running experiments:

```bash
# View metrics
cat results/metrics_summary.json

# View fairness report
cat results/fairness_report.json

# View audit logs (sample)
head -5 results/logs/audit_trail.jsonl

# View figures
ls -la figures/
```

## Configuration

Set environment variables in `.env` (optional):

```bash
# Data sources (fallback to open/synthetic if not provided)
LENDING_CLUB_API_KEY=your_key_here

# Model parameters
RANDOM_SEED=42
QUICK_MODE=true  # Reduced dataset size
```

## Testing

```bash
# Run all unit tests
pytest code/tests/

# Run integration test
pytest code/tests/test_integration.py

# Check code quality
flake8 code/
black --check code/
```
