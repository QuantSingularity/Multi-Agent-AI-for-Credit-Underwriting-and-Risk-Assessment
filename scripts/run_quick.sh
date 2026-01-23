#!/bin/bash
# Quick integration test (30 minutes on 4-core CPU)

set -e

echo "========================================="
echo "Quick Integration Test"
echo "========================================="
echo "Estimated time: 20-30 minutes"
echo ""

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code"
export RANDOM_SEED=42
export QUICK_MODE=true

# Create output directories
mkdir -p results/logs
mkdir -p figures

echo "[1/5] Generating synthetic data..."
python3 -c "
import sys
sys.path.insert(0, 'code')
from data.synthetic_generator import SyntheticDataGenerator
import pandas as pd

generator = SyntheticDataGenerator(random_seed=42)
df = generator.generate_applications(n_samples=500, default_rate=0.20)
df.to_csv('results/synthetic_data_quick.csv', index=False)
print(f'Generated {len(df)} applications')
"

echo "[2/5] Running experiments..."
python3 -c "
import sys
sys.path.insert(0, 'code')
from eval.experiment_runner import ExperimentRunner
import pandas as pd

df = pd.read_csv('results/synthetic_data_quick.csv')
runner = ExperimentRunner(output_dir='results', random_seed=42)
results = runner.run_full_evaluation(df, quick_mode=True)
print('Experiments complete!')
"

echo "[3/5] Generating figures..."
python3 scripts/generate_figures.py --quick

echo "[4/5] Running tests..."
python3 -m pytest code/tests/ -v --tb=short || true

echo "[5/5] Generating summary..."
python3 -c "
import json
with open('results/metrics_summary.json', 'r') as f:
    results = json.load(f)

print()
print('=' * 60)
print('QUICK TEST RESULTS SUMMARY')
print('=' * 60)
print()
print(f\"Dataset: {results['metadata']['n_train']} train, {results['metadata']['n_test']} test\")
print(f\"Default rate: {results['metadata']['default_rate_test']:.2%}\")
print()
print('Model Performance (AUC):')
for name, metrics in results['baselines'].items():
    print(f\"  {name:20s}: {metrics['auc']:.4f}\")
print()
print('Fairness (Baseline):')
fairness = results['fairness']['baseline']
print(f\"  Passed: {fairness['passed']}\")
for attr, metrics in fairness['metrics'].items():
    if 'demographic_parity_diff' in metrics:
        print(f\"  {attr} DP diff: {metrics['demographic_parity_diff']:.4f}\")
print()
print('Fairness (After Mitigation):')
fairness_mit = results['fairness']['reweighing']
print(f\"  Passed: {fairness_mit['passed']}\")
for attr, metrics in fairness_mit['metrics'].items():
    if 'demographic_parity_diff' in metrics:
        print(f\"  {attr} DP diff: {metrics['demographic_parity_diff']:.4f}\")
print()
print('=' * 60)
print('Quick test complete! Check results/ and figures/ directories.')
print('=' * 60)
"

echo ""
echo "✓ Quick test complete!"
echo "Results: results/metrics_summary.json"
echo "Figures: figures/"
