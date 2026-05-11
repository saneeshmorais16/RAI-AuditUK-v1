from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
SUMMARY_CSV = OUT_DIR / "model_summary_fixed.csv"
RANKING_CSV = OUT_DIR / "model_rankings_fixed.csv"
TRACEABILITY_CSV = OUT_DIR / "traceability_summary_fixed.csv"
REPORT_PDF = BASE_DIR / "RAI_Audit_UK_v1_Loan_Approval_Report.pdf"

STATUS_MAP = {"PASS": 0, "PARTIAL": 1, "FAIL": 2}
STATUS_LABELS = {0: "PASS", 1: "PARTIAL", 2: "FAIL"}


# ==========================================================
# STREAMLIT PAGE SETUP
# ==========================================================
st.set_page_config(
    page_title="RAI-AuditUK v1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_MODE = st.sidebar.radio("Theme", ["Day", "Night"], horizontal=True, index=0)
IS_DARK = THEME_MODE == "Night"

if IS_DARK:
    COLORS = {
        "app_bg": "#07111f",
        "sidebar_bg": "#0b1220",
        "card_bg": "#101827",
        "card_bg_2": "#111c2f",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "subtle": "#94a3b8",
        "border": "#334155",
        "border_soft": "#243044",
        "shadow": "0 18px 46px rgba(0, 0, 0, 0.36)",
        "plot_bg": "#101827",
        "grid": "#263348",
        "blue": "#60a5fa",
        "blue_deep": "#2563eb",
        "green": "#22c55e",
        "green_bg": "#064e3b",
        "green_soft": "#0f3b2e",
        "amber": "#f59e0b",
        "amber_bg": "#78350f",
        "amber_soft": "#3f2a0c",
        "red": "#ef4444",
        "red_bg": "#7f1d1d",
        "red_soft": "#3b1116",
        "purple": "#c084fc",
        "cyan": "#38bdf8",
        "grey_bg": "#1e293b",
        "grey_text": "#e2e8f0",
        "topbar": "linear-gradient(135deg, #0f172a 0%, #10233f 50%, #24153d 100%)",
    }
    PLOT_TEMPLATE = "plotly_dark"
else:
    COLORS = {
        "app_bg": "#eef2f7",
        "sidebar_bg": "#ffffff",
        "card_bg": "#ffffff",
        "card_bg_2": "#f8fafc",
        "text": "#0f172a",
        "muted": "#334155",
        "subtle": "#64748b",
        "border": "#cbd5e1",
        "border_soft": "#dbe3ee",
        "shadow": "0 18px 42px rgba(15, 23, 42, 0.10)",
        "plot_bg": "#ffffff",
        "grid": "#dbe4ee",
        "blue": "#2563eb",
        "blue_deep": "#1d4ed8",
        "green": "#16a34a",
        "green_bg": "#bbf7d0",
        "green_soft": "#dcfce7",
        "amber": "#d97706",
        "amber_bg": "#fde68a",
        "amber_soft": "#fef3c7",
        "red": "#dc2626",
        "red_bg": "#fecaca",
        "red_soft": "#fee2e2",
        "purple": "#9333ea",
        "cyan": "#0284c7",
        "grey_bg": "#e2e8f0",
        "grey_text": "#334155",
        "topbar": "linear-gradient(135deg, #ffffff 0%, #e0f2fe 54%, #f3e8ff 100%)",
    }
    PLOT_TEMPLATE = "plotly_white"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {COLORS['app_bg']}; color: {COLORS['text']}; }}
        .block-container {{ padding-top: 1.05rem; padding-bottom: 2rem; max-width: 1500px; }}
        header, footer {{ visibility: hidden; }}

        section[data-testid="stSidebar"] {{
            background: {COLORS['sidebar_bg']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] * {{ color: {COLORS['muted']} !important; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label,
        section[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color: {COLORS['text']} !important; }}

        .topbar {{
            background: {COLORS['topbar']};
            border: 1px solid {COLORS['border']};
            border-radius: 24px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: {COLORS['shadow']};
        }}
        .topbar h1 {{
            margin: 0;
            color: {COLORS['text']};
            font-size: 1.65rem;
            font-weight: 850;
            letter-spacing: -0.03em;
        }}
        .topbar p {{ margin: 0.3rem 0 0 0; color: {COLORS['muted']}; font-size: 0.92rem; }}

        .minirow {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.78rem; }}
        .chip {{
            display: inline-flex; align-items: center; gap: 0.35rem;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['blue']} !important;
            font-size: 0.78rem;
            font-weight: 850;
            box-shadow: 0 5px 14px rgba(0,0,0,0.04);
        }}

        .glass-card {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 22px;
            padding: 1rem;
            box-shadow: {COLORS['shadow']};
            margin-bottom: 1rem;
        }}
        .card-title {{
            color: {COLORS['text']};
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.065em;
            font-weight: 900;
            margin-bottom: 0.65rem;
        }}
        .subtle-text {{ color: {COLORS['subtle']}; font-size: 0.86rem; }}

        div[data-testid="stMetric"] {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 20px;
            padding: 0.9rem 1rem;
            box-shadow: {COLORS['shadow']};
        }}
        div[data-testid="stMetric"] label {{
            color: {COLORS['subtle']} !important;
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            font-weight: 900 !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {COLORS['text']} !important;
            font-size: 1.35rem !important;
            font-weight: 900 !important;
        }}
        div[data-testid="stMetricDelta"] {{ color: {COLORS['muted']} !important; }}

        .pill {{
            display: inline-block;
            padding: 0.40rem 0.82rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 900;
            border: 1px solid transparent;
            white-space: nowrap;
            letter-spacing: 0.01em;
        }}
        .green {{ background: {COLORS['green_bg']}; color: {'#dcfce7' if IS_DARK else '#14532d'} !important; border-color: {COLORS['green']}; }}
        .amber {{ background: {COLORS['amber_bg']}; color: {'#fef3c7' if IS_DARK else '#78350f'} !important; border-color: {COLORS['amber']}; }}
        .red {{ background: {COLORS['red_bg']}; color: {'#fee2e2' if IS_DARK else '#7f1d1d'} !important; border-color: {COLORS['red']}; }}
        .blue {{ background: {'#12315d' if IS_DARK else '#dbeafe'}; color: {'#bfdbfe' if IS_DARK else '#1e40af'} !important; border-color: {COLORS['blue']}; }}
        .grey {{ background: {COLORS['grey_bg']}; color: {COLORS['grey_text']} !important; border-color: {COLORS['border']}; }}

        .decision {{
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: {COLORS['card_bg_2']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['muted']};
            font-size: 0.95rem;
            font-weight: 800;
            box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        }}
        .note-box {{
            padding: 0.85rem 1rem;
            border-radius: 18px;
            background: {COLORS['card_bg_2']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['muted']};
            font-size: 0.90rem;
            line-height: 1.42;
        }}
        .legend-box {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.2rem 0 0.8rem 0;
        }}
        .stDataFrame {{ border-radius: 16px; overflow: hidden; border: 1px solid {COLORS['border_soft']}; }}
        div[data-testid="stDataFrame"] * {{ color: {COLORS['text']}; }}
        button[kind="primary"], .stDownloadButton button {{
            border-radius: 14px !important;
            border: 1px solid {COLORS['blue']} !important;
            background: {COLORS['card_bg']} !important;
            color: {COLORS['blue']} !important;
            font-weight: 900 !important;
        }}
        .stAlert {{ border-radius: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# FORMATTERS AND HELPERS
# ==========================================================
def fmt(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text.lower().replace(" ", "_"))


def asset_path(model: str, kind: str) -> Path:
    return OUT_DIR / f"{safe_name(model)}_{kind}.png"


def band_class(text: str) -> str:
    value = str(text).strip().upper()
    if value in {"GREEN", "PASS", "LOW"}:
        return "green"
    if value in {"AMBER", "PARTIAL", "MEDIUM", "WATCH"}:
        return "amber"
    if value in {"RED", "FAIL", "HIGH", "CRITICAL"}:
        return "red"
    return "grey"


def risk_band(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    if v < 0.25:
        return "Low"
    if v < 0.40:
        return "Watch"
    if v < 0.70:
        return "Medium"
    return "Critical"


def chip(label: str, style: str = "blue") -> str:
    return f"<span class='pill {style}'>{label}</span>"


def status_chip(label: str) -> str:
    return chip(str(label), band_class(label))


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def add_alias_columns(summary: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    ranking = ranking.copy()

    # Backwards and forwards compatibility with the corrected audit-engine output.
    if "governance_score" not in summary.columns:
        col = first_existing(summary, ["governance_score_lower_is_better", "grse_score", "total_grse_score"])
        if col is not None:
            summary["governance_score"] = summary[col]

    if "governance_score" not in ranking.columns:
        col = first_existing(ranking, ["governance_score_lower_is_better", "grse_score", "total_grse_score"])
        if col is not None:
            ranking["governance_score"] = ranking[col]

    alias_map = {
        "spd": ["spd", "spd_unpriv_minus_priv", "statistical_parity_difference"],
        "di": ["di", "di_unpriv_div_priv", "disparate_impact", "demographic_parity_ratio"],
        "eod": ["eod", "eod_unpriv_minus_priv", "equal_opportunity_difference"],
        "approval_rate_unprivileged": ["approval_rate_unprivileged", "approval_rate_age_under_30", "approval_rate_group0"],
        "approval_rate_privileged": ["approval_rate_privileged", "approval_rate_age_30_plus", "approval_rate_group1"],
    }

    for canonical, options in alias_map.items():
        if canonical not in summary.columns:
            existing = first_existing(summary, options)
            if existing is not None:
                summary[canonical] = summary[existing]

    required_defaults = {
        "performance_score": 0.0,
        "governance_score": 0.0,
        "fairness_risk": 0.0,
        "robustness_risk": 0.0,
        "explainability_risk": 0.0,
        "test_accuracy": pd.NA,
        "test_precision": pd.NA,
        "test_recall": pd.NA,
        "test_f1": pd.NA,
        "test_roc_auc": pd.NA,
        "deployment_decision": "No decision available",
        "gar": "N/A",
    }
    for col, default in required_defaults.items():
        if col not in summary.columns:
            summary[col] = default

    return summary, ranking


# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_CSV)
    ranking = pd.read_csv(RANKING_CSV)
    trace = pd.read_csv(TRACEABILITY_CSV)

    summary.columns = [c.strip() for c in summary.columns]
    ranking.columns = [c.strip() for c in ranking.columns]
    trace.columns = [c.strip() for c in trace.columns]

    summary, ranking = add_alias_columns(summary, ranking)

    if "gar" in summary.columns:
        summary["gar"] = summary["gar"].astype(str).str.title().str.strip()
    trace["status"] = trace["status"].astype(str).str.upper().str.strip()

    if "performance_rank" not in ranking.columns and {"model", "performance_score"}.issubset(ranking.columns):
        ranking["performance_rank"] = ranking["performance_score"].rank(ascending=False, method="min").astype(int)
    if "governance_rank" not in ranking.columns and {"model", "governance_score"}.issubset(ranking.columns):
        ranking["governance_rank"] = ranking["governance_score"].rank(ascending=True, method="min").astype(int)
    if "ranking_shift" not in ranking.columns and {"performance_rank", "governance_rank"}.issubset(ranking.columns):
        ranking["ranking_shift"] = ranking["performance_rank"] - ranking["governance_rank"]

    return summary, ranking, trace


def missing_outputs() -> list[Path]:
    return [p for p in [SUMMARY_CSV, RANKING_CSV, TRACEABILITY_CSV] if not p.exists()]


# ==========================================================
# PLOTS
# ==========================================================
def plotly_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=22, r=22, t=40, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["plot_bg"],
        font=dict(family="Inter, Arial", color=COLORS["muted"]),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def performance_vs_governance(summary: pd.DataFrame) -> go.Figure:
    hover_data = {
        "governance_score": ":.2f",
        "performance_score": ":.2f",
        "deployment_decision": True,
        "model": False,
    }
    if "test_roc_auc" in summary.columns:
        hover_data["test_roc_auc"] = ":.3f"

    fig = px.scatter(
        summary,
        x="governance_score",
        y="performance_score",
        text="model",
        size="test_roc_auc" if "test_roc_auc" in summary.columns else None,
        color="gar",
        color_discrete_map={"Green": COLORS["green"], "Amber": COLORS["amber"], "Red": COLORS["red"]},
        hover_name="model",
        hover_data=hover_data,
        labels={
            "governance_score": "GRSE score (lower = stronger)",
            "performance_score": "Performance score",
            "gar": "GAR",
        },
    )
    fig.add_vrect(x0=0, x1=30, fillcolor=COLORS["green"], opacity=0.16, line_width=0)
    fig.add_vrect(x0=30, x1=60, fillcolor=COLORS["amber"], opacity=0.18, line_width=0)
    fig.add_vrect(x0=60, x1=100, fillcolor=COLORS["red"], opacity=0.16, line_width=0)
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1.4, color=COLORS["text"])))
    fig.update_xaxes(range=[max(0, float(summary["governance_score"].min()) - 8), min(100, max(62, float(summary["governance_score"].max()) + 12))])
    return plotly_layout(fig, 390)


def gar_score_bar(summary: pd.DataFrame) -> go.Figure:
    ordered = summary.sort_values("governance_score", ascending=True).copy()
    fig = px.bar(
        ordered,
        x="model",
        y="governance_score",
        color="gar",
        text="governance_score",
        color_discrete_map={"Green": COLORS["green"], "Amber": COLORS["amber"], "Red": COLORS["red"]},
        labels={"governance_score": "GRSE score (lower = stronger)", "model": "Model", "gar": "GAR"},
    )
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"], opacity=0.10, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor=COLORS["amber"], opacity=0.10, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor=COLORS["red"], opacity=0.10, line_width=0)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, max(65, float(ordered["governance_score"].max()) + 10)])
    return plotly_layout(fig, 330)


def gar_distribution(summary: pd.DataFrame) -> go.Figure:
    counts = summary["gar"].value_counts().reindex(["Green", "Amber", "Red"]).fillna(0).reset_index()
    counts.columns = ["GAR", "Count"]
    fig = px.bar(
        counts,
        x="GAR",
        y="Count",
        color="GAR",
        text="Count",
        color_discrete_map={"Green": COLORS["green"], "Amber": COLORS["amber"], "Red": COLORS["red"]},
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_yaxes(dtick=1, range=[0, max(2, int(counts["Count"].max()) + 1)])
    return plotly_layout(fig, 260)


def gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={"font": {"size": 30, "color": COLORS["text"]}},
            title={"text": title, "font": {"size": 14, "color": COLORS["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["subtle"]},
                "bar": {"color": COLORS["blue"]},
                "bgcolor": COLORS["plot_bg"],
                "borderwidth": 1,
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, 30], "color": COLORS["green_soft"]},
                    {"range": [30, 60], "color": COLORS["amber_soft"]},
                    {"range": [60, 100], "color": COLORS["red_soft"]},
                ],
                "threshold": {"line": {"color": COLORS["text"], "width": 4}, "thickness": 0.78, "value": float(value)},
            },
        )
    )
    return plotly_layout(fig, 285)


def performance_bar(row: pd.Series) -> go.Figure:
    df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Value": [
                row.get("test_accuracy"),
                row.get("test_precision"),
                row.get("test_recall"),
                row.get("test_f1"),
                row.get("test_roc_auc"),
            ],
        }
    )
    fig = px.bar(df, x="Metric", y="Value", text="Value", color_discrete_sequence=[COLORS["blue"]])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, 1.08])
    return plotly_layout(fig, 320)


def risk_profile(row: pd.Series) -> go.Figure:
    df = pd.DataFrame(
        {
            "Risk": ["Fairness", "Robustness", "Explainability"],
            "Score": [
                float(row.get("fairness_risk", 0)) * 100,
                float(row.get("robustness_risk", 0)) * 100,
                float(row.get("explainability_risk", 0)) * 100,
            ],
        }
    )
    fig = px.bar(
        df,
        x="Score",
        y="Risk",
        orientation="h",
        text="Score",
        color="Risk",
        color_discrete_map={"Fairness": COLORS["green"], "Robustness": COLORS["cyan"], "Explainability": COLORS["purple"]},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
    fig.update_xaxes(range=[0, 100])
    return plotly_layout(fig, 285)


def component_bar(row: pd.Series) -> go.Figure:
    df = pd.DataFrame(
        {
            "Component": ["Fairness", "Robustness", "Explainability", "Governance"],
            "Score": [
                row.get("grse_fairness_component", 0),
                row.get("grse_robustness_component", 0),
                row.get("grse_explainability_component", 0),
                row.get("grse_governance_component", 0),
            ],
        }
    )
    fig = px.bar(
        df,
        x="Component",
        y="Score",
        text="Score",
        color="Component",
        color_discrete_map={"Fairness": COLORS["green"], "Robustness": COLORS["cyan"], "Explainability": COLORS["purple"], "Governance": COLORS["blue"]},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
    return plotly_layout(fig, 285)


def robustness_bar(row: pd.Series) -> go.Figure:
    df = pd.DataFrame({"Stress test": ["Noise", "Missingness"], "Flip-rate": [row.get("flip_noise", 0), row.get("flip_missingness", 0)]})
    fig = px.bar(df, x="Stress test", y="Flip-rate", text="Flip-rate", color_discrete_sequence=[COLORS["cyan"]])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, max(0.12, float(pd.to_numeric(df["Flip-rate"], errors="coerce").max()) + 0.04)])
    return plotly_layout(fig, 285)


def fairness_metrics_bar(row: pd.Series) -> go.Figure:
    df = pd.DataFrame(
        {
            "Metric": ["SPD", "DI", "EOD"],
            "Value": [row.get("spd"), row.get("di"), row.get("eod")],
        }
    )
    fig = px.bar(df, x="Metric", y="Value", text="Value", color="Metric", color_discrete_map={"SPD": COLORS["blue"], "DI": COLORS["green"], "EOD": COLORS["purple"]})
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["subtle"])
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["subtle"])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    return plotly_layout(fig, 285)


def fairness_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "spd", "di", "eod", "fairness_risk"]
    optional = ["approval_rate_unprivileged", "approval_rate_privileged"]
    cols = [c for c in cols + optional if c in summary.columns]
    table = summary[cols].copy()
    rename_map = {
        "model": "Model",
        "spd": "SPD: unprivileged - privileged",
        "di": "DI: unprivileged / privileged",
        "eod": "EOD: unprivileged - privileged",
        "fairness_risk": "Fairness risk",
        "approval_rate_unprivileged": "Approval rate: unprivileged",
        "approval_rate_privileged": "Approval rate: privileged",
    }
    return table.rename(columns=rename_map)


def trace_heatmap(trace: pd.DataFrame) -> go.Figure:
    df = trace.copy()
    df["status_num"] = df["status"].map(STATUS_MAP).fillna(1).astype(int)
    pivot = df.pivot_table(index="framework_requirement", columns="model", values="status_num", aggfunc="max")
    text = pivot.map(lambda x: STATUS_LABELS.get(int(x), "PARTIAL") if pd.notna(x) else "")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            text=text.values,
            texttemplate="%{text}",
            colorscale=[
                [0, COLORS["green_bg"]],
                [0.49, COLORS["green_bg"]],
                [0.50, COLORS["amber_bg"]],
                [0.74, COLORS["amber_bg"]],
                [0.75, COLORS["red_bg"]],
                [1, COLORS["red_bg"]],
            ],
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )
    return plotly_layout(fig, max(390, 34 * len(pivot.index)))


# ==========================================================
# MITIGATION AND STYLING TABLES
# ==========================================================
def severity(value: float) -> str:
    if value >= 0.70:
        return "Critical"
    if value >= 0.50:
        return "High"
    if value >= 0.40:
        return "Medium"
    return "Low"


def mitigation_table(row: pd.Series, selected_trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    fairness = float(row.get("fairness_risk", 0) or 0)
    robustness = float(row.get("robustness_risk", 0) or 0)
    explain = float(row.get("explainability_risk", 0) or 0)

    if fairness >= 0.40:
        rows.append({
            "Risk area": "Fairness",
            "Severity": severity(fairness),
            "Trigger": f"SPD {fmt(row.get('spd'),3)} | DI {fmt(row.get('di'),3)} | EOD {fmt(row.get('eod'),3)}",
            "Mitigation factor": "Possible group-outcome disparity under the selected audit group",
            "Action": "Review thresholds and apply human review for borderline/adverse cases",
            "Owner": "Compliance",
        })
    if robustness >= 0.40:
        rows.append({
            "Risk area": "Robustness",
            "Severity": severity(robustness),
            "Trigger": f"Noise {pct(row.get('flip_noise'))} | Missing {pct(row.get('flip_missingness'))}",
            "Mitigation factor": "Prediction instability under stress",
            "Action": "Add missingness controls, input validation and scheduled stress tests",
            "Owner": "ML Engineering",
        })
    if explain >= 0.40:
        rows.append({
            "Risk area": "Explainability",
            "Severity": severity(explain),
            "Trigger": f"Explainability risk {fmt(explain,3)}",
            "Mitigation factor": "Explanation evidence needs stronger governance review",
            "Action": "Produce model-level and case-level reason-code explanations",
            "Owner": "Model Risk",
        })

    open_trace = selected_trace[selected_trace["status"].isin(["PARTIAL", "FAIL"])]
    for _, item in open_trace.head(4).iterrows():
        rows.append({
            "Risk area": "Traceability",
            "Severity": "Medium" if item["status"] == "PARTIAL" else "High",
            "Trigger": str(item.get("framework_requirement", "")),
            "Mitigation factor": str(item.get("evidence", "")),
            "Action": "Close evidence gap before production sign-off",
            "Owner": "AI Governance",
        })

    if not rows:
        rows.append({
            "Risk area": "Monitoring",
            "Severity": "Low",
            "Trigger": "No material breach",
            "Mitigation factor": "Deployment still requires control monitoring",
            "Action": "Repeat audit after material model, data or threshold changes",
            "Owner": "AI Governance",
        })
    return pd.DataFrame(rows)


def style_severity(df: pd.DataFrame):
    def colour(value: Any) -> str:
        v = str(value).lower()
        if v in {"critical", "high"}:
            return f"background-color:{COLORS['red_bg']};color:{'#fee2e2' if IS_DARK else '#7f1d1d'};font-weight:900;"
        if v == "medium":
            return f"background-color:{COLORS['amber_bg']};color:{'#fef3c7' if IS_DARK else '#78350f'};font-weight:900;"
        if v == "low":
            return f"background-color:{COLORS['green_bg']};color:{'#dcfce7' if IS_DARK else '#14532d'};font-weight:900;"
        return ""
    return df.style.map(colour, subset=["Severity"])


def style_trace(df: pd.DataFrame):
    def colour(value: Any) -> str:
        v = str(value).upper().strip()
        if v == "PASS":
            return f"background-color:{COLORS['green_bg']};color:{'#dcfce7' if IS_DARK else '#14532d'};font-weight:900;"
        if v == "PARTIAL":
            return f"background-color:{COLORS['amber_bg']};color:{'#fef3c7' if IS_DARK else '#78350f'};font-weight:900;"
        if v == "FAIL":
            return f"background-color:{COLORS['red_bg']};color:{'#fee2e2' if IS_DARK else '#7f1d1d'};font-weight:900;"
        return ""
    return df.style.map(colour, subset=["status"])


# ==========================================================
# APP BODY
# ==========================================================
missing = missing_outputs()
if missing:
    st.markdown("<div class='topbar'><h1>RAI-AuditUK v1</h1><p>Run the audit engine first to generate dashboard data.</p></div>", unsafe_allow_html=True)
    st.error("Missing output files")
    for file in missing:
        st.code(str(file), language="text")
    st.code(f'cd "{BASE_DIR}"\npython LG.py\nstreamlit run dashboard_app.py', language="bash")
    st.stop()

summary_df, ranking_df, trace_df = load_outputs()
models = summary_df["model"].astype(str).tolist()
best_governance = summary_df.sort_values("governance_score", ascending=True).iloc[0]
best_performance = summary_df.sort_values("performance_score", ascending=False).iloc[0]

st.sidebar.markdown("### RAI-AuditUK v1")
page = st.sidebar.radio("Navigation", ["Overview", "Model Review", "Fairness", "Mitigations", "Traceability", "Evidence"], index=0)
selected_model = st.sidebar.selectbox("Model", models, index=models.index(str(best_governance["model"])))
row = summary_df[summary_df["model"].astype(str) == selected_model].iloc[0]
selected_trace = trace_df[trace_df["model"].astype(str) == selected_model].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(status_chip(str(row.get("gar", "N/A"))), unsafe_allow_html=True)
st.sidebar.caption(f"GRSE {fmt(row.get('governance_score'))} · Performance {fmt(row.get('performance_score'))}")
st.sidebar.caption("GRSE is lower-is-better: Green ≤ 30 · Amber ≤ 60 · Red > 60")
st.sidebar.caption("Fairness metrics are audit evidence, not legal conclusions.")

st.markdown(
    """
    <div class="topbar">
        <h1>RAI-AuditUK v1</h1>
        <p>Responsible AI governance dashboard for loan approval model audits</p>
        <div class="minirow">
            <span class="chip">Audit Engine</span>
            <span class="chip">GRSE Lower = Stronger</span>
            <span class="chip">Fairness Direction Fixed</span>
            <span class="chip">Robustness</span>
            <span class="chip">Traceability</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='legend-box'>"
    + chip("Green: deployable", "green")
    + chip("Amber: mitigation required", "amber")
    + chip("Red: restricted / not deployable", "red")
    + chip("Fairness: audit evidence only", "blue")
    + "</div>",
    unsafe_allow_html=True,
)

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best governance", str(best_governance["model"]), f"lowest GRSE {fmt(best_governance['governance_score'])}")
    c2.metric("Best performance", str(best_performance["model"]), f"Score {fmt(best_performance['performance_score'])}")
    c3.metric("Green models", f"{int((summary_df['gar'].astype(str).str.title() == 'Green').sum())}/{len(summary_df)}")
    c4.metric("Open failures", str(int((trace_df["status"] == "FAIL").sum())))

    st.markdown(
        "<div class='note-box'><b>Interpretation:</b> Performance score is higher-is-better, while GRSE is lower-is-better because it is a governance risk score. "
        "Fairness metrics use the favourable class correctly as Approved/Good = 0. SPD and EOD are calculated as unprivileged minus privileged; DI is unprivileged divided by privileged.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("<div class='glass-card'><div class='card-title'>Performance vs governance</div>", unsafe_allow_html=True)
        st.plotly_chart(performance_vs_governance(summary_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass-card'><div class='card-title'>Green / Amber / Red</div>", unsafe_allow_html=True)
        st.plotly_chart(gar_distribution(summary_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'><div class='card-title'>Model ranking</div>", unsafe_allow_html=True)
        ranking_cols = [c for c in ["model", "performance_rank", "governance_rank", "ranking_shift"] if c in ranking_df.columns]
        st.dataframe(ranking_df.sort_values("governance_rank")[ranking_cols], hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='card-title'>GAR score by model</div>", unsafe_allow_html=True)
    st.plotly_chart(gar_score_bar(summary_df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='card-title'>Audit summary</div>", unsafe_allow_html=True)
    cols = [
        "model",
        "performance_score",
        "governance_score",
        "gar",
        "deployment_decision",
        "test_accuracy",
        "test_f1",
        "test_roc_auc",
        "spd",
        "di",
        "eod",
        "fairness_risk",
        "robustness_risk",
        "explainability_risk",
    ]
    st.dataframe(summary_df[[c for c in cols if c in summary_df.columns]].sort_values("governance_score"), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Review":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", selected_model)
    c2.metric("GRSE", fmt(row.get("governance_score")), "lower = stronger")
    c3.metric("Performance", fmt(row.get("performance_score")))
    c4.metric("ROC-AUC", fmt(row.get("test_roc_auc"), 3))
    c5.markdown("<div class='card-title'>GAR</div>" + status_chip(str(row.get("gar", "N/A"))), unsafe_allow_html=True)

    st.markdown(f"<div class='decision'>{row.get('deployment_decision', 'No decision available')}</div>", unsafe_allow_html=True)
    st.write("")

    a, b, c = st.columns([1, 1.1, 1])
    with a:
        st.markdown("<div class='glass-card'><div class='card-title'>GRSE risk score</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(float(row.get("governance_score", 0)), "Governance risk"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='glass-card'><div class='card-title'>Risk profile</div>", unsafe_allow_html=True)
        st.plotly_chart(risk_profile(row), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='glass-card'><div class='card-title'>Stress tests</div>", unsafe_allow_html=True)
        st.plotly_chart(robustness_bar(row), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    f1.metric("SPD", fmt(row.get("spd"), 3), "unprivileged - privileged")
    f2.metric("DI", fmt(row.get("di"), 3), "unprivileged / privileged")
    f3.metric("EOD", fmt(row.get("eod"), 3), "approval TPR gap")

    st.markdown("<div class='glass-card'><div class='card-title'>Predictive metrics</div>", unsafe_allow_html=True)
    st.plotly_chart(performance_bar(row), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Fairness risk", fmt(row.get("fairness_risk"), 3), risk_band(row.get("fairness_risk")))
    m2.metric("Robustness risk", fmt(row.get("robustness_risk"), 3), risk_band(row.get("robustness_risk")))
    m3.metric("Explainability risk", fmt(row.get("explainability_risk"), 3), risk_band(row.get("explainability_risk")))

    st.markdown("<div class='glass-card'><div class='card-title'>Visual evidence</div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    for col, title, path in [
        (v1, "Confusion matrix", asset_path(selected_model, "confusion_matrix")),
        (v2, "Fairness", asset_path(selected_model, "fairness")),
        (v3, "Explainability", asset_path(selected_model, "explainability")),
    ]:
        with col:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Not generated")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Fairness":
    st.markdown("<div class='glass-card'><div class='card-title'>Fairness calculation direction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note-box'>"
        "<b>Target:</b> loan_status, where 0 = Approved/Good and 1 = Rejected/Bad. "
        "The favourable outcome is therefore predicted class 0. "
        "For Dataset B, AGE_UNDER_30 is treated as the unprivileged audit group and AGE_30_PLUS as the privileged audit group. "
        "SPD = approval rate unprivileged - approval rate privileged. DI = approval rate unprivileged / approval rate privileged. "
        "EOD = favourable-class TPR unprivileged - favourable-class TPR privileged. These are audit indicators, not legal conclusions."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Selected model", selected_model)
    b.metric("SPD", fmt(row.get("spd"), 3), "negative means lower unprivileged approval")
    c.metric("DI", fmt(row.get("di"), 3), "below 1 means lower unprivileged approval")
    d.metric("EOD", fmt(row.get("eod"), 3), "approval TPR difference")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='glass-card'><div class='card-title'>Selected model fairness metrics</div>", unsafe_allow_html=True)
        st.plotly_chart(fairness_metrics_bar(row), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='glass-card'><div class='card-title'>Fairness evidence image</div>", unsafe_allow_html=True)
        fair_path = asset_path(selected_model, "fairness")
        if fair_path.exists():
            st.image(str(fair_path), use_container_width=True)
        else:
            st.info("Fairness plot was not generated by the audit engine.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='card-title'>Fairness audit table</div>", unsafe_allow_html=True)
    st.dataframe(fairness_table(summary_df), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Mitigations":
    st.markdown("<div class='glass-card'><div class='card-title'>Mitigation plan</div>", unsafe_allow_html=True)
    mit_df = mitigation_table(row, selected_trace)
    st.dataframe(style_severity(mit_df), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    x1, x2, x3 = st.columns(3)
    x1.metric("Fairness", risk_band(row.get("fairness_risk")), fmt(row.get("fairness_risk"), 3))
    x2.metric("Robustness", risk_band(row.get("robustness_risk")), fmt(row.get("robustness_risk"), 3))
    x3.metric("Explainability", risk_band(row.get("explainability_risk")), fmt(row.get("explainability_risk"), 3))

    st.markdown("<div class='glass-card'><div class='card-title'>GRSE component split</div>", unsafe_allow_html=True)
    st.plotly_chart(component_bar(row), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Traceability":
    t1, t2, t3 = st.columns(3)
    t1.metric("PASS", str(int((selected_trace["status"] == "PASS").sum())))
    t2.metric("PARTIAL", str(int((selected_trace["status"] == "PARTIAL").sum())))
    t3.metric("FAIL", str(int((selected_trace["status"] == "FAIL").sum())))

    st.markdown("<div class='glass-card'><div class='card-title'>Traceability heatmap</div>", unsafe_allow_html=True)
    st.plotly_chart(trace_heatmap(trace_df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='glass-card'><div class='card-title'>{selected_model} evidence mapping</div>", unsafe_allow_html=True)
    st.dataframe(style_trace(selected_trace), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='glass-card'><div class='card-title'>Downloads</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        if REPORT_PDF.exists():
            st.download_button("PDF report", REPORT_PDF.read_bytes(), file_name=REPORT_PDF.name, mime="application/pdf", use_container_width=True)
        else:
            st.info("PDF missing")
    with d2:
        st.download_button("Summary CSV", SUMMARY_CSV.read_bytes(), file_name=SUMMARY_CSV.name, mime="text/csv", use_container_width=True)
    with d3:
        st.download_button("Ranking CSV", RANKING_CSV.read_bytes(), file_name=RANKING_CSV.name, mime="text/csv", use_container_width=True)
    with d4:
        st.download_button("Traceability CSV", TRACEABILITY_CSV.read_bytes(), file_name=TRACEABILITY_CSV.name, mime="text/csv", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='card-title'>Generated files</div>", unsafe_allow_html=True)
    files = sorted(p.name for p in OUT_DIR.glob("*"))
    st.dataframe(pd.DataFrame({"File": files}), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
