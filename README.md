# Responsible AI Governance & Model Risk Assessment

RAI-AuditUK v1 is a portfolio Responsible AI audit project for loan approval and credit-risk style machine-learning models. It generates performance, fairness, explainability, robustness and governance-readiness evidence with an understandable audit report.

## Implemented features

- Synthetic credit/loan dataset generator for safe local review.
- Random Forest classifier for an educational loan-approval task.
- Performance metrics: accuracy, precision, recall, F1 and ROC-AUC.
- Fairness review by age group using selection-rate gaps and recall.
- Explainability using permutation importance.
- Robustness check using small synthetic perturbations.
- Markdown audit report with Green/Amber-style risk framing.
- Streamlit dashboard assets from the original project are preserved.
- pytest tests and GitHub Actions workflow.

## Architecture

```text
synthetic credit data
  -> train/test split
  -> model training
  -> performance metrics
  -> fairness by group
  -> permutation importance
  -> perturbation robustness check
  -> audit report
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_dashboard.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Run audit

```bash
python run_audit.py
```

Generated outputs:

- `outputs/performance_metrics.csv`
- `outputs/fairness_by_age_group.csv`
- `outputs/permutation_importance.csv`
- `outputs/audit_report.md`

These generated sample outputs from the synthetic audit workflow are committed for review.

## Run dashboard

```bash
streamlit run dashboard_app.py
```

## Run tests

```bash
pytest -q
```

## Data provenance

The testable audit module uses synthetic data generated in code. The repository also preserves the original loan-approval CSV/PDF artefacts from the existing project. Before any public or CV use, review those artefacts for licensing, privacy and source documentation.

## Assumptions and limitations

- This is an educational portfolio implementation, not a legal compliance certification.
- Fairness groups and thresholds are simplified.
- Fairness metrics must be selected with domain, legal and stakeholder input in real use.
- Permutation importance is a useful signal but not a complete explanation.
- Robustness testing here uses narrow perturbations and should not be treated as exhaustive.
- No production deployment, client use or regulatory approval is claimed.
