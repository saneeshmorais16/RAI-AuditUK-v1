from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def make_synthetic_credit_data(rows: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.normal(52000, 16000, rows).clip(15000, 130000)
    debt_ratio = rng.uniform(0.05, 0.75, rows)
    credit_score = rng.normal(660, 80, rows).clip(420, 840)
    age_group = rng.choice(["under_30", "30_to_50", "over_50"], rows, p=[0.25, 0.55, 0.2])
    employment_years = rng.integers(0, 18, rows)
    logits = (credit_score - 620) / 80 + (income - 45000) / 30000 - debt_ratio * 2 + employment_years / 12
    approved = (logits + rng.normal(0, 0.65, rows) > 0).astype(int)
    return pd.DataFrame({"income": income.round(2), "debt_ratio": debt_ratio.round(3), "credit_score": credit_score.round(0), "age_group": age_group, "employment_years": employment_years, "approved": approved})


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"income", "debt_ratio", "credit_score", "age_group", "employment_years", "approved"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    x = pd.get_dummies(data.drop(columns=["approved"]), columns=["age_group"], drop_first=False)
    y = data["approved"].astype(int)
    return x, y


def fairness_by_group(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for group in sorted(groups.unique()):
        mask = groups == group
        rows.append({"group": group, "selection_rate": float(np.mean(y_pred[mask])), "recall": float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), "count": int(mask.sum())})
    result = pd.DataFrame(rows)
    result["selection_rate_gap_vs_max"] = result["selection_rate"].max() - result["selection_rate"]
    return result


def robustness_check(model: RandomForestClassifier, x_test: pd.DataFrame) -> float:
    perturbed = x_test.copy()
    if "income" in perturbed:
        perturbed["income"] = perturbed["income"] * 0.95
    if "credit_score" in perturbed:
        perturbed["credit_score"] = perturbed["credit_score"] - 10
    return float(np.mean(model.predict(x_test) == model.predict(perturbed)))


def run_audit(output_dir: Path = Path("outputs")) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = make_synthetic_credit_data()
    x, y = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=11, stratify=y)
    model = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=4, random_state=11)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "robustness_agreement_after_small_perturbation": robustness_check(model, x_test),
    }
    groups = data.loc[x_test.index, "age_group"]
    fairness = fairness_by_group(y_test, pred, groups)
    importance = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=11)
    explainability = pd.DataFrame({"feature": x_test.columns, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    metrics_path = output_dir / "performance_metrics.csv"
    fairness_path = output_dir / "fairness_by_age_group.csv"
    explainability_path = output_dir / "permutation_importance.csv"
    report_path = output_dir / "audit_report.md"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    fairness.to_csv(fairness_path, index=False)
    explainability.to_csv(explainability_path, index=False)
    report_path.write_text(_report(metrics, fairness, explainability), encoding="utf-8")
    return {"metrics": metrics, "metrics_path": metrics_path, "fairness_path": fairness_path, "explainability_path": explainability_path, "report_path": report_path}


def _report(metrics: dict, fairness: pd.DataFrame, explainability: pd.DataFrame) -> str:
    largest_gap = fairness["selection_rate_gap_vs_max"].max()
    top_features = ", ".join(explainability.head(4)["feature"].tolist())
    risk_rating = "Amber" if largest_gap > 0.15 or metrics["robustness_agreement_after_small_perturbation"] < 0.9 else "Green"
    return f"""# Responsible AI audit report

## Scope

Educational model-risk assessment for a synthetic loan approval classifier. This is guidance for portfolio demonstration, not legal compliance certification.

## Performance

- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- F1: {metrics['f1']:.3f}
- ROC-AUC: {metrics['roc_auc']:.3f}

## Fairness

Fairness is assessed with group selection-rate gaps and recall by age group. These metrics are inspectable but incomplete; real deployment would require legally reviewed protected-characteristic handling and domain-specific thresholds.

Largest selection-rate gap versus the highest group: {largest_gap:.3f}

## Explainability

Permutation importance highlights these leading features: {top_features}.

## Robustness

Prediction agreement after a small synthetic income and credit-score perturbation: {metrics['robustness_agreement_after_small_perturbation']:.3f}

## Governance mapping

Rating: {risk_rating}

This maps findings to educational governance themes: data limitations, model performance, fairness review, explainability evidence, robustness checks, human oversight and monitoring. It is not a certification of legal or regulatory compliance.

## Limitations

The dataset is synthetic, fairness groups are simplified, explainability is model-specific, and robustness checks are narrow. Results should be treated as evidence for learning and portfolio review only.
"""
