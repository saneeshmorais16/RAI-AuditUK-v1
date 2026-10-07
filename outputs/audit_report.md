# Responsible AI audit report

## Scope

Educational model-risk assessment for a synthetic loan approval classifier. This is guidance for portfolio demonstration, not legal compliance certification.

## Performance

- Accuracy: 0.825
- Precision: 0.849
- Recall: 0.882
- F1: 0.865
- ROC-AUC: 0.920

## Fairness

Fairness is assessed with group selection-rate gaps and recall by age group. These metrics are inspectable but incomplete; real deployment would require legally reviewed protected-characteristic handling and domain-specific thresholds.

Largest selection-rate gap versus the highest group: 0.164

## Explainability

Permutation importance highlights these leading features: credit_score, employment_years, income, age_group_under_30.

## Robustness

Prediction agreement after a small synthetic income and credit-score perturbation: 0.925

## Governance mapping

Rating: Amber

This maps findings to educational governance themes: data limitations, model performance, fairness review, explainability evidence, robustness checks, human oversight and monitoring. It is not a certification of legal or regulatory compliance.

## Limitations

The dataset is synthetic, fairness groups are simplified, explainability is model-specific, and robustness checks are narrow. Results should be treated as evidence for learning and portfolio review only.
