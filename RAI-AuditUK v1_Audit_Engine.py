from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    PageBreak,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(r"C:\Users\sanee\AppData\Local\Programs\Python\Python313\RAI-Audit UK v1 - Copy")

DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset_B.csv"

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

REPORT_PATH = BASE_DIR / "RAI_Audit_UK_v1_Loan_Approval_Report.pdf"
SUMMARY_CSV = OUT_DIR / "model_summary_fixed.csv"
RANKING_CSV = OUT_DIR / "model_rankings_fixed.csv"
TRACEABILITY_CSV = OUT_DIR / "traceability_summary_fixed.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SPD_THRESHOLD = 0.20
DI_LOW, DI_HIGH = 0.80, 1.25
EOD_THRESHOLD = 0.20
ROBUST_FLIP_ALERT = 0.10
MAX_EXPLAIN_ROWS = 400
MAX_LOCAL_REASON_CASES = 8


# ==========================================================
# HELPERS
# ==========================================================
def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text.lower().replace(" ", "_"))


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def fmt_metric(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.3f}"


def ensure_dataset_exists() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            f"Expected dataset at: {DATA_PATH}"
        )


def wrapped_table(
    data: List[List[Any]],
    col_widths: List[float],
    styles,
    repeat_rows: int = 1
) -> Table:
    wrapped = []

    body_style = ParagraphStyle(
        "WrappedBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
    )
    header_style = ParagraphStyle(
        "WrappedHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
    )

    for r, row in enumerate(data):
        current = []
        for cell in row:
            current.append(Paragraph(str(cell), header_style if r == 0 else body_style))
        wrapped.append(current)

    t = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=repeat_rows)
    t.setStyle([
        ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    return t


# ==========================================================
# DATA
# ==========================================================
def detect_target_column(df: pd.DataFrame) -> str:
    for col in [
        "loan_status",
        "Loan_Status",
        "TARGET",
        "target",
        "status",
        "approved",
        "approval_status",
    ]:
        if col in df.columns:
            return col
    raise ValueError("No recognised target column found.")


def map_target_to_binary(df: pd.DataFrame, target_col: str) -> pd.Series:
    raw = df[target_col].astype(str).str.strip()

    mapping = {
        "Approved": 0,
        "Rejected": 1,
        "Y": 0,
        "N": 1,
        "Yes": 0,
        "No": 1,
        "accepted": 0,
        "declined": 1,
        "approve": 0,
        "reject": 1,
        "0": 0,
        "1": 1,
    }

    if raw.nunique() <= 2 and set(raw.unique()).issubset({"0", "1"}):
        y = raw.astype(int)
    elif target_col in {"TARGET", "target"}:
        y = pd.to_numeric(df[target_col], errors="coerce")
    else:
        y = raw.map(mapping)

    if y.isna().any():
        raise ValueError(f"Unexpected values in target column '{target_col}'.")

    return y.astype(int)


def detect_protected_proxy(df: pd.DataFrame) -> Tuple[str, np.ndarray, Tuple[str, str], bool]:
    """
    Fairness audit group encoding:
    0 = unprivileged group
    1 = privileged group

    Dataset B:
    0 = AGE_UNDER_30
    1 = AGE_30_PLUS

    These groups are used for fairness audit evidence only.
    They are not proof of legal discrimination or legal compliance.
    """

    age_under_cols = [
        "age_group_audit_AGE_UNDER_30",
        "AGE_UNDER_30",
        "Age_Under_30",
        "age_under_30",
    ]

    age_plus_cols = [
        "age_group_audit_AGE_30_PLUS",
        "AGE_30_PLUS",
        "Age_30_Plus",
        "age_30_plus",
    ]

    under_col = next((c for c in age_under_cols if c in df.columns), None)
    plus_col = next((c for c in age_plus_cols if c in df.columns), None)

    if plus_col is not None:
        p_group = pd.to_numeric(df[plus_col], errors="coerce").fillna(0).astype(int).values

        if len(np.unique(p_group)) >= 2:
            return (
                "age_group_audit",
                p_group,
                ("AGE_UNDER_30 (unprivileged)", "AGE_30_PLUS (privileged)"),
                True,
            )

    if under_col is not None:
        under_30 = pd.to_numeric(df[under_col], errors="coerce").fillna(0).astype(int).values
        p_group = np.where(under_30 == 1, 0, 1).astype(int)

        if len(np.unique(p_group)) >= 2:
            return (
                "age_group_audit",
                p_group,
                ("AGE_UNDER_30 (unprivileged)", "AGE_30_PLUS (privileged)"),
                True,
            )

    options = [
        ("education", {"Graduate": 1, "Not Graduate": 0}, ("Not Graduate", "Graduate")),
        ("Education", {"Graduate": 1, "Not Graduate": 0}, ("Not Graduate", "Graduate")),
        ("self_employed", {"No": 0, "Yes": 1}, ("Not Self-Employed", "Self-Employed")),
        ("Self_Employed", {"No": 0, "Yes": 1}, ("Not Self-Employed", "Self-Employed")),
        ("Gender", {"Male": 1, "Female": 0}, ("Female", "Male")),
        ("gender", {"Male": 1, "Female": 0}, ("Female", "Male")),
    ]

    for col, mapping, labels in options:
        if col in df.columns:
            g = df[col].astype(str).str.strip().map(mapping)

            if g.notna().sum() > 0 and g.nunique(dropna=True) >= 2:
                return col, g.fillna(0).astype(int).values, labels, True

    return "proxy_none", np.zeros(len(df), dtype=int), ("Group 0", "Group 1"), False


def select_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    preferred = [
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    available = [c for c in preferred if c in df.columns]

    if len(available) < 4:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        audit_only_cols = {
            "age_group_audit_AGE_UNDER_30",
            "age_group_audit_AGE_30_PLUS",
            "AGE_UNDER_30",
            "AGE_30_PLUS",
            "Age_Under_30",
            "Age_30_Plus",
            "age_under_30",
            "age_30_plus",
        }

        drop_cols = {
            target_col,
            "loan_id",
            "Loan_ID",
            "id",
            "ID",
            *audit_only_cols,
        }

        available = [c for c in numeric_cols if c not in drop_cols]

    if len(available) < 3:
        raise ValueError("Not enough usable numeric features were found in the dataset.")

    X = df[available].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X.fillna(X.median(numeric_only=True))


def make_splits(X: pd.DataFrame, y: pd.Series, p: np.ndarray) -> Dict[str, Any]:
    X_train, X_temp, y_train, y_temp, p_train, p_temp = train_test_split(
        X,
        y,
        p,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test, p_val, p_test = train_test_split(
        X_temp,
        y_temp,
        p_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "p_train": p_train,
        "p_val": p_val,
        "p_test": p_test,
        "scaler": scaler,
    }


# ==========================================================
# MODELS
# ==========================================================
SCALED_MODELS = {"Logistic Regression", "Linear SVM"}


def build_models() -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

        "Linear SVM": LinearSVC(
            max_iter=15000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    return models


def fit_predict_models(models: Dict[str, Any], split_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = {}

    for name, model in models.items():
        if name in SCALED_MODELS:
            X_train = split_data["X_train_scaled"]
            X_val = split_data["X_val_scaled"]
            X_test = split_data["X_test_scaled"]
        else:
            X_train = split_data["X_train"]
            X_val = split_data["X_val"]
            X_test = split_data["X_test"]

        y_train = split_data["y_train"]
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

        score_val = None
        score_test = None

        if hasattr(model, "predict_proba"):
            score_val = model.predict_proba(X_val)[:, 1]
            score_test = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            score_val = model.decision_function(X_val)
            score_test = model.decision_function(X_test)

        results[name] = {
            "model": model,
            "pred_val": pred_val.astype(int),
            "pred_test": pred_test.astype(int),
            "score_val": score_val,
            "score_test": score_test,
        }

    return results


# ==========================================================
# METRICS
# ==========================================================
def compute_performance(y_true, y_pred, y_score=None) -> Dict[str, float]:
    metrics = {
        "accuracy": as_float(accuracy_score(y_true, y_pred)),
        "precision": as_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": as_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": as_float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = as_float(roc_auc_score(y_true, y_score)) if y_score is not None else float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def approval_from_bad_pred(y_pred_bad: np.ndarray) -> np.ndarray:
    """
    Converts model prediction into favourable approval outcome.

    Target meaning:
    0 = Approved / Good
    1 = Rejected / Bad

    Favourable outcome:
    approved = 1 when prediction is class 0.
    """
    return (1 - y_pred_bad).astype(int)


def statistical_parity_difference(approved: np.ndarray, p_group: np.ndarray) -> float:
    """
    SPD = approval_rate_unprivileged - approval_rate_privileged

    Group encoding:
    0 = unprivileged
    1 = privileged

    If AGE_UNDER_30 has a lower approval rate than AGE_30_PLUS,
    SPD should be negative.
    """
    p_group = np.asarray(p_group).astype(int)
    approved = np.asarray(approved).astype(int)

    if len(np.unique(p_group)) < 2:
        return float("nan")

    mask_unpriv = p_group == 0
    mask_priv = p_group == 1

    if mask_unpriv.sum() == 0 or mask_priv.sum() == 0:
        return float("nan")

    approval_unpriv = approved[mask_unpriv].mean()
    approval_priv = approved[mask_priv].mean()

    return float(approval_unpriv - approval_priv)


def disparate_impact(approved: np.ndarray, p_group: np.ndarray) -> float:
    """
    DI = approval_rate_unprivileged / approval_rate_privileged

    Group encoding:
    0 = unprivileged
    1 = privileged

    If AGE_UNDER_30 has a lower approval rate than AGE_30_PLUS,
    DI should be below 1.
    """
    p_group = np.asarray(p_group).astype(int)
    approved = np.asarray(approved).astype(int)

    if len(np.unique(p_group)) < 2:
        return float("nan")

    mask_unpriv = p_group == 0
    mask_priv = p_group == 1

    if mask_unpriv.sum() == 0 or mask_priv.sum() == 0:
        return float("nan")

    approval_unpriv = approved[mask_unpriv].mean()
    approval_priv = approved[mask_priv].mean()

    if approval_priv == 0:
        return float("nan")

    return float(approval_unpriv / approval_priv)


def equal_opportunity_difference(y_true_bad, y_pred_bad, p_group) -> float:
    """
    EOD = TPR_unprivileged - TPR_privileged

    Target meaning:
    0 = Approved / Good
    1 = Rejected / Bad

    Since approval is the favourable outcome, TPR is calculated using class 0:
    TPR = correctly predicted approved among actually approved applicants.
    """
    p_group = np.asarray(p_group).astype(int)
    y_true_bad = np.asarray(y_true_bad).astype(int)
    y_pred_bad = np.asarray(y_pred_bad).astype(int)

    if len(np.unique(p_group)) < 2:
        return float("nan")

    mask_unpriv = p_group == 0
    mask_priv = p_group == 1

    y_true_approved = y_true_bad == 0
    y_pred_approved = y_pred_bad == 0

    actual_approved_unpriv = y_true_approved & mask_unpriv
    actual_approved_priv = y_true_approved & mask_priv

    denom_unpriv = actual_approved_unpriv.sum()
    denom_priv = actual_approved_priv.sum()

    if denom_unpriv == 0 or denom_priv == 0:
        return float("nan")

    tpr_unpriv = (y_pred_approved & actual_approved_unpriv).sum() / denom_unpriv
    tpr_priv = (y_pred_approved & actual_approved_priv).sum() / denom_priv

    return float(tpr_unpriv - tpr_priv)


def fairness_risk_score(spd: float, di: float, eod: float) -> float:
    scores = []

    if not np.isnan(spd):
        scores.append(min(abs(spd) / SPD_THRESHOLD, 1.0))

    if not np.isnan(di):
        if DI_LOW <= di <= DI_HIGH:
            scores.append(0.0)
        elif di < DI_LOW:
            scores.append(min((DI_LOW - di) / DI_LOW, 1.0))
        else:
            scores.append(min((di - DI_HIGH) / DI_HIGH, 1.0))

    if not np.isnan(eod):
        scores.append(min(abs(eod) / EOD_THRESHOLD, 1.0))

    return float(max(scores)) if scores else 0.0


# ==========================================================
# ROBUSTNESS
# ==========================================================
def get_income_like_feature(columns: List[str]) -> Optional[str]:
    for c in ["income_annum", "AMT_INCOME_TOTAL", "ApplicantIncome", "person_income"]:
        if c in columns:
            return c
    return columns[0] if columns else None


def prediction_flip_rate(
    model_name: str,
    model: Any,
    X_ref_df: pd.DataFrame,
    base_pred: np.ndarray,
    scaler: StandardScaler,
    noise: float = 0.10,
    miss_rate: float = 0.10
) -> Tuple[float, float]:
    feature = get_income_like_feature(list(X_ref_df.columns))
    if feature is None:
        return 0.0, 0.0

    X_noise = X_ref_df.copy()
    X_noise[feature] = X_noise[feature] * (1 + np.random.normal(0, noise, len(X_noise)))

    if model_name in SCALED_MODELS:
        X_noise_scaled = pd.DataFrame(scaler.transform(X_noise), columns=X_ref_df.columns, index=X_ref_df.index)
        y_noise = model.predict(X_noise_scaled)
    else:
        y_noise = model.predict(X_noise)

    flip_noise = float(np.mean(y_noise.astype(int) != base_pred.astype(int)))

    X_miss = X_ref_df.copy()
    median_value = float(X_miss[feature].median())
    mask = np.random.rand(len(X_miss)) < miss_rate
    X_miss.loc[mask, feature] = median_value

    if model_name in SCALED_MODELS:
        X_miss_scaled = pd.DataFrame(scaler.transform(X_miss), columns=X_ref_df.columns, index=X_ref_df.index)
        y_miss = model.predict(X_miss_scaled)
    else:
        y_miss = model.predict(X_miss)

    flip_miss = float(np.mean(y_miss.astype(int) != base_pred.astype(int)))
    return flip_noise, flip_miss


def robustness_risk_score(flip_noise: float, flip_miss: float) -> float:
    return float(min((flip_noise + flip_miss) / (2 * ROBUST_FLIP_ALERT), 1.0))


# ==========================================================
# EXPLAINABILITY
# ==========================================================
def plot_importance_bar(feature_names, importances, title, out_path):
    order = np.argsort(importances)
    plt.figure(figsize=(6.5, 4.5))
    plt.barh(np.array(feature_names)[order], np.array(importances)[order])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def stable_global_importance(model_name, model, X_bg, X_eval, feature_names):
    img_path = OUT_DIR / f"{safe_filename(model_name)}_explainability.png"

    if model_name == "Linear SVM" and hasattr(model, "coef_"):
        imp = np.abs(model.coef_).reshape(-1)
        plot_importance_bar(feature_names, imp, f"Coefficient Importance - {model_name}", img_path)
        return imp, img_path

    if model_name == "Logistic Regression" and hasattr(model, "coef_") and not SHAP_AVAILABLE:
        imp = np.abs(model.coef_).reshape(-1)
        plot_importance_bar(feature_names, imp, f"Coefficient Importance - {model_name}", img_path)
        return imp, img_path

    if not SHAP_AVAILABLE:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).reshape(-1)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_)).reshape(-1)
        else:
            imp = np.ones(len(feature_names))

        plot_importance_bar(feature_names, imp, f"Feature Importance - {model_name}", img_path)
        return imp, img_path

    X_eval_local = X_eval.copy()
    X_bg_local = X_bg.copy()

    if len(X_eval_local) > MAX_EXPLAIN_ROWS:
        X_eval_local = X_eval_local.sample(MAX_EXPLAIN_ROWS, random_state=RANDOM_STATE)
    if len(X_bg_local) > MAX_EXPLAIN_ROWS:
        X_bg_local = X_bg_local.sample(MAX_EXPLAIN_ROWS, random_state=RANDOM_STATE)

    try:
        if model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_bg_local)
            shap_values = explainer.shap_values(X_eval_local)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_eval_local)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        imp = np.abs(np.array(shap_values)).mean(axis=0).reshape(-1)
    except Exception:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).reshape(-1)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_)).reshape(-1)
        else:
            imp = np.ones(len(feature_names))

    plot_importance_bar(feature_names, imp, f"Explainability Evidence - {model_name}", img_path)
    return imp, img_path


def explainability_risk_score(model_name: str, importance: np.ndarray) -> Tuple[float, float]:
    base = {
        "Logistic Regression": 0.20,
        "Linear SVM": 0.25,
        "Random Forest": 0.35,
    }.get(model_name, 0.40)

    total = float(np.sum(importance)) if float(np.sum(importance)) > 0 else 1.0
    top_share = float(np.max(importance) / total) if total > 0 else 1.0
    extra = max(0.0, min((top_share - 0.40) / 0.40, 0.35))
    return float(min(base + extra, 1.0)), top_share


# ==========================================================
# PLOTS
# ==========================================================
def confusion_matrix_plot(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    img_path = OUT_DIR / f"{safe_filename(model_name)}_confusion_matrix.png"

    plt.figure(figsize=(4.8, 3.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Good(0)", "Bad(1)"])
    plt.yticks(ticks, ["Good(0)", "Bad(1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(img_path, dpi=180)
    plt.close()
    return img_path, cm


def fairness_plot(approved_pred, p_group, group_labels, model_name):
    img_path = OUT_DIR / f"{safe_filename(model_name)}_fairness.png"

    rate0 = float(approved_pred[p_group == 0].mean()) if np.any(p_group == 0) else float("nan")
    rate1 = float(approved_pred[p_group == 1].mean()) if np.any(p_group == 1) else float("nan")

    plt.figure(figsize=(4.8, 3.2))
    plt.bar([group_labels[0], group_labels[1]], [rate0, rate1])
    plt.ylabel("Predicted Approval Rate")
    plt.title(f"Approval Rates by Group - {model_name}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(img_path, dpi=180)
    plt.close()
    return img_path, rate0, rate1


# ==========================================================
# GOVERNANCE READY SCORING ENGINE (GRSE)
# ==========================================================
WEIGHTS = {
    "fairness": 0.35,
    "robustness": 0.20,
    "explainability": 0.25,
    "governance": 0.20,
}


def governance_risk_base(protected_is_proxy: bool) -> float:
    return 0.55 if protected_is_proxy else 0.45


def grse_breakdown(
    fairness_risk: float,
    robustness_risk: float,
    explainability_risk: float,
    protected_is_proxy: bool
) -> Dict[str, float]:
    governance_base = governance_risk_base(protected_is_proxy)

    fairness_component = fairness_risk * WEIGHTS["fairness"] * 100
    robustness_component = robustness_risk * WEIGHTS["robustness"] * 100
    explainability_component = explainability_risk * WEIGHTS["explainability"] * 100
    governance_component = governance_base * WEIGHTS["governance"] * 100

    total = fairness_component + robustness_component + explainability_component + governance_component

    return {
        "fairness_component": round(fairness_component, 2),
        "robustness_component": round(robustness_component, 2),
        "explainability_component": round(explainability_component, 2),
        "governance_component": round(governance_component, 2),
        "governance_base": round(governance_base, 3),
        "total_score": round(total, 2),
    }


def gar_band(score: float) -> str:
    if score <= 30:
        return "Green"
    if score <= 60:
        return "Amber"
    return "Red"


def deployment_decision(score: float) -> str:
    if score <= 30:
        return "Low Risk - Deployable"
    if score <= 60:
        return "Medium Risk - Mitigation Required"
    if score <= 80:
        return "High Risk - Restricted Use"
    return "Critical Risk - Not Deployable"


def mitigation_actions(
    fairness_risk,
    spd,
    di,
    eod,
    robustness_risk,
    flip_noise,
    flip_miss,
    explainability_risk,
    top_share,
    protected_name
):
    actions = []

    if fairness_risk > 0.4:
        actions.append({
            "Risk Area": "Fairness and Discrimination",
            "Factors": [
                f"Outcome disparity across groups using proxy/protected feature '{protected_name}'",
                f"SPD={fmt_metric(spd)}",
                f"DI={fmt_metric(di)}",
                f"EOD={fmt_metric(eod)}",
            ],
            "Actions": [
                "Test bias mitigation and retrain under documented settings",
                "Review proxy-sensitive features and threshold policy",
                "Introduce human review for borderline or adverse decisions",
                "Monitor group-level disparities after deployment",
            ]
        })

    if robustness_risk > 0.4:
        actions.append({
            "Risk Area": "Robustness and Data Quality",
            "Factors": [
                f"Flip-rate under noise stress={flip_noise:.3f}",
                f"Flip-rate under missingness stress={flip_miss:.3f}",
                "Potential production drift or input instability",
            ],
            "Actions": [
                "Add stronger input validation and missingness controls",
                "Retrain with drift-aware or noise-augmented settings",
                "Implement scheduled stress testing and drift alerts",
            ]
        })

    if explainability_risk > 0.4:
        actions.append({
            "Risk Area": "Transparency and Explainability",
            "Factors": [
                f"Top-feature dominance share={top_share:.3f}",
                "Possible over-reliance on a small set of drivers",
            ],
            "Actions": [
                "Provide model-level and case-level explanation summaries",
                "Review dominant features for governance suitability",
                "Translate technical signals into reason-code style outputs",
            ]
        })

    if not actions:
        actions.append({
            "Risk Area": "Overall Governance",
            "Factors": ["No material governance threshold was exceeded in this run"],
            "Actions": [
                "Proceed only with controlled monitoring",
                "Repeat the audit on schedule and after material model changes",
            ]
        })

    return actions


def build_traceability(fairness_risk, robustness_risk, explainability_risk):
    def status(risk, strict=False):
        if risk >= 0.70:
            return "FAIL"
        if risk >= 0.40:
            return "FAIL" if strict else "PARTIAL"
        return "PASS"

    return [
        ["EU AI Act Art. 9 Risk Management", "Risk score and mitigations generated", "PASS"],
        ["EU AI Act Art. 10 Data Governance / Bias", "SPD, DI, EOD fairness evidence", status(fairness_risk, strict=True)],
        ["EU AI Act Art. 13 Transparency", "Explainability evidence generated", status(explainability_risk)],
        ["EU AI Act Art. 14 Human Oversight", "Human-review mitigation included", "PASS" if fairness_risk >= 0.4 else "PARTIAL"],
        ["EU AI Act Art. 15 Robustness", "Noise and missingness stress tests", status(robustness_risk, strict=True)],
        ["UK FCA Principle 6", "Group-level outcome monitoring and mitigations", status(fairness_risk)],
        ["UK FCA Principle 7", "Reason-code style explanation artefacts", status(explainability_risk)],
        ["UK PRA Model Risk", "Stress testing and governance documentation", status(robustness_risk)],
        ["NIST AI RMF", "Measure and Manage evidence included", "PASS"],
        ["ISO/IEC 42001", "Governance and monitoring control suggestions", "PARTIAL"],
    ]


# ==========================================================
# REJECTION REASONING
# ==========================================================
REASON_TEMPLATES = {
    "cibil_score": ("Low credit score (CIBIL)", "Improve credit score by paying dues on time and reducing utilisation."),
    "income_annum": ("Lower income relative to requested loan", "Increase verifiable income or reduce requested exposure."),
    "loan_amount": ("High requested loan amount", "Reduce the requested loan amount or strengthen collateral."),
    "loan_term": ("Unfavourable or long loan term", "Adjust loan term to improve affordability and risk profile."),
    "residential_assets_value": ("Low residential asset value", "Provide stronger collateral or better asset evidence."),
    "commercial_assets_value": ("Low commercial asset value", "Strengthen financial backing or additional guarantees."),
    "luxury_assets_value": ("Low total asset strength", "Improve overall asset support or reduce exposure."),
    "bank_asset_value": ("Low bank assets or liquidity", "Show stronger savings or liquidity evidence."),
    "ApplicantIncome": ("Lower applicant income", "Increase income evidence or apply for a smaller amount."),
    "LoanAmount": ("High requested loan amount", "Reduce exposure or improve repayment support."),
    "Credit_History": ("Weak credit history signal", "Improve repayment history and reduce prior delinquencies."),
}


def dataset_rejection_drivers(X_eval, y_pred_bad):
    approved_mask = y_pred_bad == 0
    rejected_mask = y_pred_bad == 1

    if rejected_mask.sum() < 5 or approved_mask.sum() < 5:
        return []

    means_rej = X_eval.loc[rejected_mask].mean()
    means_app = X_eval.loc[approved_mask].mean()
    std_all = X_eval.std().replace(0, 1.0)

    effect = (means_rej - means_app) / std_all
    ranked = effect.reindex(effect.abs().sort_values(ascending=False).index)

    return [(feat, float(eff), "higher" if eff > 0 else "lower") for feat, eff in ranked.items()]


def sample_linear_reason_strings(model, X_scaled_df, row_index, feature_names, topk=3):
    if not hasattr(model, "coef_"):
        return []

    weights = np.asarray(model.coef_).reshape(-1)
    x = X_scaled_df.iloc[row_index].values.reshape(-1)
    contrib = weights * x
    order = np.argsort(-contrib)

    items = []
    for idx in order[:topk]:
        if contrib[idx] > 0:
            items.append((feature_names[idx], float(contrib[idx])))
    return items


def build_rejection_reason_report(model_name, model, X_eval_df, X_eval_scaled_df, y_pred_bad, feature_names):
    common_drivers = dataset_rejection_drivers(X_eval_df, y_pred_bad)

    steps = []
    for feat, _, _ in common_drivers[:5]:
        if feat in REASON_TEMPLATES:
            advice = REASON_TEMPLATES[feat][1]
            if advice not in steps:
                steps.append(advice)

    rejected_indices = np.where(y_pred_bad == 1)[0]
    if len(rejected_indices) == 0:
        return common_drivers, [], steps

    sample_idx = rejected_indices[:MAX_LOCAL_REASON_CASES]
    sample_cases = []

    for idx in sample_idx:
        reasons = []
        row = X_eval_df.iloc[idx]

        if model_name in {"Logistic Regression", "Linear SVM"}:
            local = sample_linear_reason_strings(model, X_eval_scaled_df, idx, feature_names, topk=3)
            for feat, _ in local:
                reasons.append(REASON_TEMPLATES.get(feat, (feat, ""))[0])
        else:
            for feat, _, direction in common_drivers[:3]:
                label = REASON_TEMPLATES.get(feat, (feat, ""))[0]
                value = row.get(feat, None)
                if value is not None and np.isfinite(value):
                    reasons.append(f"{label} ({direction}; value={value:.0f})")
                else:
                    reasons.append(f"{label} ({direction})")

        if not reasons:
            for feat, _, _ in common_drivers[:3]:
                reasons.append(REASON_TEMPLATES.get(feat, (feat, ""))[0])

        sample_cases.append({
            "case_index": int(idx),
            "reasons": reasons[:3]
        })

    return common_drivers, sample_cases, steps


# ==========================================================
# RANKING
# ==========================================================
def safe_auc(metrics):
    auc = metrics.get("roc_auc", float("nan"))
    return 0.0 if np.isnan(auc) else auc


def performance_score(metrics):
    score = (
        0.25 * metrics["accuracy"] +
        0.20 * metrics["precision"] +
        0.20 * metrics["recall"] +
        0.20 * metrics["f1"] +
        0.15 * safe_auc(metrics)
    ) * 100
    return float(round(score, 2))

def build_fairness_audit_note(protected_name: str, group_labels: tuple[str, str]) -> str:
    """
    Builds the correct fairness note depending on the detected audit group.

    group_labels[0] = unprivileged group
    group_labels[1] = privileged group
    """

    unprivileged_label = group_labels[0]
    privileged_label = group_labels[1]

    if protected_name == "education":
        group_context = (
            "For Dataset A education-based proxy auditing, "
            f"{unprivileged_label} is treated as the unprivileged audit group "
            f"and {privileged_label} as the privileged audit group."
        )

    elif protected_name == "age_group_audit":
        group_context = (
            "For Dataset B age-group proxy auditing, "
            f"{unprivileged_label} is treated as the unprivileged audit group "
            f"and {privileged_label} as the privileged audit group."
        )

    else:
        group_context = (
            "For this dataset, the detected audit group is used as a proxy-sensitive "
            f"comparison variable, with {unprivileged_label} treated as the "
            f"unprivileged audit group and {privileged_label} as the privileged audit group."
        )

    return (
        "<b>Fairness audit note:</b> "
        "The favourable outcome is class 0, meaning Approved/Good. "
        f"{group_context} "
        "SPD is calculated as the approval rate of the unprivileged group minus the "
        "approval rate of the privileged group. DI is calculated as the approval rate "
        "of the unprivileged group divided by the approval rate of the privileged group. "
        "EOD is calculated as the favourable-class true positive rate of the "
        "unprivileged group minus the favourable-class true positive rate of the "
        "privileged group. These metrics are fairness audit evidence only and do not "
        "prove legal discrimination or certify legal compliance."
    )


# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dataset_exists()

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    target_col = detect_target_column(df)
    y = map_target_to_binary(df, target_col)
    protected_name, p, group_labels, protected_is_proxy = detect_protected_proxy(df)
    X = select_features(df, target_col)
    feature_names = list(X.columns)

    split_data = make_splits(X, y, p)
    models = build_models()
    fitted = fit_predict_models(models, split_data)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    usable_width = A4[0] - doc.leftMargin - doc.rightMargin
    elements = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elements.append(Paragraph("<b>RAI-Audit UK v1 - Loan Approval Governance Audit Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        f"<b>Generated:</b> {now}<br/>"
        f"<b>Dataset:</b> {DATA_PATH.name}<br/>"
        f"<b>Target:</b> {target_col} (0=Approved/Good, 1=Rejected/Bad)<br/>"
        f"<b>Protected / proxy feature:</b> {protected_name} (proxy={protected_is_proxy})<br/>"
        f"<b>Feature set:</b> {', '.join(feature_names)}<br/>"
        f"<b>Splits:</b> train={len(split_data['X_train'])}, validation={len(split_data['X_val'])}, test={len(split_data['X_test'])}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    fairness_note = build_fairness_audit_note(protected_name, group_labels)

    elements.append(Paragraph(
        fairness_note,
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "<b>GRSE interpretation note:</b> "
        "The Governance Ready Scoring Engine is a risk score. A lower GRSE score indicates stronger "
        "governance readiness under the tested audit conditions.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    model_rows = []
    ranking_rows = []
    traceability_rows_all = []
    end_report = {"per_model_drivers": {}, "per_model_cases": {}, "per_model_steps": {}}

    for model_name, payload in fitted.items():
        model = payload["model"]
        pred_val = payload["pred_val"]
        pred_test = payload["pred_test"]
        score_val = payload["score_val"]
        score_test = payload["score_test"]

        perf_val = compute_performance(split_data["y_val"], pred_val, score_val)
        perf_test = compute_performance(split_data["y_test"], pred_test, score_test)

        approved_pred = approval_from_bad_pred(pred_test)
        spd = statistical_parity_difference(approved_pred, split_data["p_test"])
        di = disparate_impact(approved_pred, split_data["p_test"])
        eod = equal_opportunity_difference(split_data["y_test"], pred_test, split_data["p_test"])
        fair_risk = fairness_risk_score(spd, di, eod)

        print("\n" + "=" * 80)
        print(f"FAIRNESS AUDIT EVIDENCE - {model_name}")
        print("=" * 80)
        print("Target meaning:")
        print("  0 = Approved / Good outcome")
        print("  1 = Rejected / Bad outcome")
        print("Favourable outcome:")
        print("  Predicted class 0 = Approved / Good")
        print("Sensitive/proxy attribute:")
        print(f"  {protected_name}")
        print("Group direction:")
        print(f"  {group_labels[0]} = unprivileged/reference group")
        print(f"  {group_labels[1]} = privileged/comparison group")
        print("Metric direction:")
        print("  SPD = approval_rate_unprivileged - approval_rate_privileged")
        print("  DI  = approval_rate_unprivileged / approval_rate_privileged")
        print("  EOD = TPR_unprivileged - TPR_privileged, using approval/class 0")
        print(f"Statistical Parity Difference (SPD): {fmt_metric(spd)}")
        print(f"Disparate Impact / Demographic Parity Ratio (DI): {fmt_metric(di)}")
        print(f"Equal Opportunity Difference (EOD): {fmt_metric(eod)}")
        print(f"Fairness risk score: {fmt_metric(fair_risk)}")
        print("Interpretation note:")
        print("  These values are fairness audit evidence only.")
        print("  They are not proof of legal discrimination or legal compliance.")

        flip_noise, flip_miss = prediction_flip_rate(
            model_name,
            model,
            split_data["X_test"],
            pred_test,
            split_data["scaler"]
        )
        robust_risk = robustness_risk_score(flip_noise, flip_miss)

        if model_name in SCALED_MODELS:
            importance, exp_img = stable_global_importance(
                model_name,
                model,
                split_data["X_train_scaled"],
                split_data["X_test_scaled"],
                feature_names
            )
        else:
            importance, exp_img = stable_global_importance(
                model_name,
                model,
                split_data["X_train"],
                split_data["X_test"],
                feature_names
            )

        expl_risk, top_share = explainability_risk_score(model_name, importance)
        grse = grse_breakdown(fair_risk, robust_risk, expl_risk, protected_is_proxy)
        governance_score = grse["total_score"]
        gar = gar_band(governance_score)
        decision = deployment_decision(governance_score)

        cm_img, _ = confusion_matrix_plot(split_data["y_test"], pred_test, model_name)
        fair_img, rate0, rate1 = fairness_plot(approved_pred, split_data["p_test"], group_labels, model_name)
        trace_rows = build_traceability(fair_risk, robust_risk, expl_risk)
        mitigations = mitigation_actions(
            fair_risk, spd, di, eod, robust_risk, flip_noise, flip_miss, expl_risk, top_share, protected_name
        )
        common_drivers, sample_cases, steps = build_rejection_reason_report(
            model_name,
            model,
            split_data["X_test"],
            split_data["X_test_scaled"],
            pred_test,
            feature_names
        )

        end_report["per_model_drivers"][model_name] = common_drivers
        end_report["per_model_cases"][model_name] = sample_cases
        end_report["per_model_steps"][model_name] = steps

        perf_score = performance_score(perf_test)

        model_rows.append({
            "model": model_name,
            "test_accuracy": perf_test["accuracy"],
            "test_precision": perf_test["precision"],
            "test_recall": perf_test["recall"],
            "test_f1": perf_test["f1"],
            "test_roc_auc": perf_test["roc_auc"],
            "spd_unpriv_minus_priv": spd,
            "di_unpriv_div_priv": di,
            "eod_unpriv_minus_priv": eod,
            "flip_noise": flip_noise,
            "flip_missingness": flip_miss,
            "fairness_risk": fair_risk,
            "robustness_risk": robust_risk,
            "explainability_risk": expl_risk,
            "grse_fairness_component": grse["fairness_component"],
            "grse_robustness_component": grse["robustness_component"],
            "grse_explainability_component": grse["explainability_component"],
            "grse_governance_component": grse["governance_component"],
            "governance_score_lower_is_better": governance_score,
            "gar": gar,
            "deployment_decision": decision,
            "performance_score": perf_score,
        })

        ranking_rows.append({
            "model": model_name,
            "performance_score": perf_score,
            "governance_score": governance_score,
        })

        for framework, evidence, status_value in trace_rows:
            traceability_rows_all.append({
                "model": model_name,
                "framework_requirement": framework,
                "evidence": evidence,
                "status": status_value,
            })

        elements.append(PageBreak())
        elements.append(Paragraph(f"<b>Model Card and Audit - {model_name}</b>", styles["Heading1"]))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph(
            f"<b>Performance Score:</b> {perf_score:.2f}<br/>"
            f"<b>Governance Ready Scoring Engine (GRSE) Score:</b> {governance_score:.2f} "
            f"(lower = stronger governance readiness)<br/>"
            f"<b>GAR Classification:</b> {gar}<br/>"
            f"<b>Deployment Decision:</b> {decision}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 10))

        perf_table = wrapped_table([
            ["Metric", "Validation", "Test"],
            ["Accuracy", f"{perf_val['accuracy']:.3f}", f"{perf_test['accuracy']:.3f}"],
            ["Precision", f"{perf_val['precision']:.3f}", f"{perf_test['precision']:.3f}"],
            ["Recall", f"{perf_val['recall']:.3f}", f"{perf_test['recall']:.3f}"],
            ["F1", f"{perf_val['f1']:.3f}", f"{perf_test['f1']:.3f}"],
            ["ROC-AUC", fmt_metric(perf_val["roc_auc"]), fmt_metric(perf_test["roc_auc"])],
        ], [usable_width * 0.34, usable_width * 0.33, usable_width * 0.33], styles)

        elements.append(Paragraph("<b>Predictive Performance</b>", styles["Heading2"]))
        elements.append(perf_table)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
        elements.append(Image(str(cm_img), width=320, height=250))
        elements.append(Spacer(1, 12))

        evidence_table = wrapped_table([
            ["Metric", "Value"],
            [f"Approval rate ({group_labels[0]})", fmt_metric(rate0)],
            [f"Approval rate ({group_labels[1]})", fmt_metric(rate1)],
            ["SPD: unprivileged - privileged", fmt_metric(spd)],
            ["DI: unprivileged / privileged", fmt_metric(di)],
            ["EOD: unprivileged - privileged", fmt_metric(eod)],
            ["Fairness risk", fmt_metric(fair_risk)],
            ["Flip-rate (noise)", fmt_metric(flip_noise)],
            ["Flip-rate (missingness)", fmt_metric(flip_miss)],
            ["Robustness risk", fmt_metric(robust_risk)],
            ["Explainability risk", fmt_metric(expl_risk)],
            ["Top feature dominance", fmt_metric(top_share)],
        ], [usable_width * 0.58, usable_width * 0.42], styles)

        elements.append(Paragraph("<b>Governance Evidence</b>", styles["Heading2"]))
        elements.append(evidence_table)
        elements.append(Spacer(1, 12))

        grse_table = wrapped_table([
            ["GRSE Component", "Risk Score Contribution"],
            ["Fairness component", f"{grse['fairness_component']:.2f}"],
            ["Robustness component", f"{grse['robustness_component']:.2f}"],
            ["Explainability component", f"{grse['explainability_component']:.2f}"],
            ["Governance baseline component", f"{grse['governance_component']:.2f}"],
            ["Total GRSE score (lower = stronger governance readiness)", f"{grse['total_score']:.2f}"],
        ], [usable_width * 0.68, usable_width * 0.32], styles)

        elements.append(Paragraph(
            "<b>Governance Ready Scoring Engine (GRSE: lower score = stronger governance readiness)</b>",
            styles["Heading2"]
        ))
        elements.append(grse_table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Fairness Evidence</b>", styles["Heading2"]))
        elements.append(Image(str(fair_img), width=320, height=220))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Explainability Evidence</b>", styles["Heading2"]))
        elements.append(Image(str(exp_img), width=360, height=250))
        elements.append(Spacer(1, 10))

        topk = min(6, len(feature_names))
        top_idx = np.argsort(-importance)[:topk]
        top_features = [["Feature", "Importance"]] + [
            [feature_names[i], f"{float(importance[i]):.6f}"] for i in top_idx
        ]

        elements.append(Paragraph("<b>Top Features</b>", styles["Heading2"]))
        elements.append(wrapped_table(top_features, [usable_width * 0.65, usable_width * 0.35], styles))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Governance Mitigation Factors and Recommended Actions</b>", styles["Heading2"]))
        for m in mitigations:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"<b>Risk Area:</b> {m['Risk Area']}", styles["Normal"]))
            elements.append(Paragraph("<b>Identified factors:</b>", styles["Normal"]))
            for factor in m["Factors"]:
                elements.append(Paragraph(f"- {factor}", styles["Normal"]))
            elements.append(Paragraph("<b>Recommended actions:</b>", styles["Normal"]))
            for action in m["Actions"]:
                elements.append(Paragraph(f"- {action}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("<b>Evidence -> Regulation Traceability Matrix</b>", styles["Heading2"]))
        trace_data = [["Framework / Requirement", "Evidence", "Status"]] + trace_rows
        elements.append(wrapped_table(trace_data, [usable_width * 0.34, usable_width * 0.52, usable_width * 0.14], styles))
        elements.append(Spacer(1, 12))

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df["performance_rank"] = ranking_df["performance_score"].rank(ascending=False, method="min").astype(int)
    ranking_df["governance_rank"] = ranking_df["governance_score"].rank(ascending=True, method="min").astype(int)
    ranking_df["ranking_shift"] = ranking_df["performance_rank"] - ranking_df["governance_rank"]

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Comparative Evaluation - Performance-Centric vs Governance-Centric Ranking</b>", styles["Heading1"]))
    elements.append(Paragraph(
        "<b>Ranking note:</b> Performance score is ranked highest to lowest. "
        "GRSE is a risk score, so governance score is ranked lowest to highest.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 8))

    rank_rows = [[
        "Model",
        "Performance Score",
        "Performance Rank",
        "GRSE Score (lower = stronger)",
        "Governance Rank",
        "Ranking Shift"
    ]]

    for _, row in ranking_df.sort_values("performance_rank").iterrows():
        rank_rows.append([
            row["model"],
            f"{row['performance_score']:.2f}",
            int(row["performance_rank"]),
            f"{row['governance_score']:.2f}",
            int(row["governance_rank"]),
            int(row["ranking_shift"]),
        ])

    elements.append(wrapped_table(
        rank_rows,
        [
            usable_width * 0.26,
            usable_width * 0.16,
            usable_width * 0.12,
            usable_width * 0.16,
            usable_width * 0.12,
            usable_width * 0.18,
        ],
        styles
    ))
    elements.append(Spacer(1, 12))

    elements.append(PageBreak())
    elements.append(Paragraph("<b>Mitigation Report - Why Loans Were Not Approved (Dataset-Driven)</b>", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for model_name in models.keys():
        drivers = end_report["per_model_drivers"].get(model_name, [])
        cases = end_report["per_model_cases"].get(model_name, [])
        steps = end_report["per_model_steps"].get(model_name, [])

        elements.append(Paragraph(f"<b>{model_name}: Common Rejection Drivers</b>", styles["Heading2"]))

        if not drivers:
            elements.append(Paragraph("- Not enough rejected/approved samples to compute stable rejection drivers.", styles["Normal"]))
            elements.append(Spacer(1, 8))
            continue

        rows = [["Feature", "Effect (Rejected vs Approved)", "Direction", "Human-readable reason"]]
        for feat, eff, direction in drivers[:6]:
            rows.append([feat, f"{eff:.3f}", direction, REASON_TEMPLATES.get(feat, (feat, ""))[0]])

        elements.append(wrapped_table(
            rows,
            [
                usable_width * 0.23,
                usable_width * 0.23,
                usable_width * 0.17,
                usable_width * 0.37,
            ],
            styles
        ))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Example applicant-facing reasons (sample rejected cases)</b>", styles["Heading3"]))
        if not cases:
            elements.append(Paragraph("- No rejected predictions were available for sampling.", styles["Normal"]))
        else:
            for c in cases:
                elements.append(Paragraph(f"- Case #{c['case_index']}: {'; '.join(c['reasons'])}", styles["Normal"]))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("<b>Suggested applicant mitigation steps</b>", styles["Heading3"]))
        if not steps:
            elements.append(Paragraph("- General guidance: strengthen credit profile, reduce exposure, and improve supporting evidence.", styles["Normal"]))
        else:
            for s in steps[:6]:
                elements.append(Paragraph(f"- {s}", styles["Normal"]))
        elements.append(Spacer(1, 14))

    pd.DataFrame(model_rows).sort_values("governance_score_lower_is_better").to_csv(SUMMARY_CSV, index=False)
    ranking_df.to_csv(RANKING_CSV, index=False)
    pd.DataFrame(traceability_rows_all).to_csv(TRACEABILITY_CSV, index=False)

    doc.build(elements)

    print("SUCCESS")
    print(f"Report: {REPORT_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Ranking CSV: {RANKING_CSV}")
    print(f"Traceability CSV: {TRACEABILITY_CSV}")


if __name__ == "__main__":
    main()
