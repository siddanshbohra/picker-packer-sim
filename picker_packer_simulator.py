"""
Picker & Packer Manpower Simulator — FirstClub Quick Commerce
=============================================================
Run:
    cd /Users/siddansh/Code && python3 -m streamlit run picker-packer-sim/picker_packer_simulator.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
EXPORTS_DIR = REPO_ROOT / "data" / "exports"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

CENTRAL_EXPORTS_DIR = REPO_ROOT.parent / "data" / "exports"
THROUGHPUT_CSV = CENTRAL_EXPORTS_DIR / "picker_packer_hourly_throughput_4w.csv"
LOCAL_THROUGHPUT_CSV = EXPORTS_DIR / "picker_packer_hourly_throughput_4w.csv"
REFERENCE_THROUGHPUT_CSV = REPO_ROOT / "data" / "reference" / "picker_packer_hourly_throughput_4w.csv"
DEFAULT_OPD_PATH = (
    Path.home()
    / "Downloads"
    / "OPD Overview_ Daily Targets & Planning - OPD View_Inv_Rider Plan.csv"
)

# reporting_langgraph only available locally; Superset fetch degrades gracefully without it
_langgraph_path = REPO_ROOT.parent / "reporting_langgraph"
if _langgraph_path.exists():
    sys.path.insert(0, str(_langgraph_path))

st.set_page_config(
    layout="wide",
    page_title="Picker & Packer Simulator",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
WH_NAME_MAP = {
    "FCHBLRAECS01": "AECS Layout",
    "FCHBLRBEN01": "Bannerghatta",
    "FCHBLRBHO01": "Bhoganhalli",
    "FCHBLRDOD01": "Doddakannelli",
    "FCHBLRELC01": "Electronic City",
    "FCHBLRHAR01": "Haralur",
    "FCHBLRHEB01": "Hebbal",
    "FCHBLRHOO01": "Hoodi",
    "FCHBLRHSAF01": "Safal/Virgonagar",
    "FCHBLRHSR01": "HSR Layout",
    "FCHBLRJPN01": "JP Nagar",
    "FCHBLRKNP01": "Kanakapura Road",
    "FCHBLRKOR01": "Koramangala",
    "FCHBLRRAJ01": "Rajajinagar",
    "FCHBLRSJR01": "Sarjapur",
    "FCHBLRTHA01": "Thanisandra",
    "FCHBLRVAR01": "Varthur",
    "FCHBLRWHF01": "Whitefield",
    "FCHHYDNAR01": "Hyderabad Narsingi",
}
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

REFERENCE_SQL = r"""
SELECT
    DATE(fo.created_at)                      AS order_date,
    HOUR(fo.created_at)                      AS order_hour,
    fo.warehouse_id                          AS ch_id,
    COUNT(DISTINCT fo.id)                    AS order_count,
    SUM(COALESCE(foi.requested_quantity, 0)) AS total_units
FROM inventory.fulfilment_orders fo
INNER JOIN inventory.fulfilment_order_item foi ON foi.fulfilment_order_id = fo.id
WHERE fo.created_at >= DATE_SUB(CURDATE(), INTERVAL 28 DAY)
  AND fo.warehouse_id LIKE 'FC%'
  AND fo.id NOT LIKE 'SB%'
  AND fo.id NOT LIKE 'SL%'
  AND fo.processing_status NOT IN (
      'CANCELLED','ALLOCATION_FAILED','PACKING_FAILED','HANDOVER_FAILED'
  )
GROUP BY DATE(fo.created_at), HOUR(fo.created_at), fo.warehouse_id
ORDER BY order_date DESC, order_hour, ch_id
"""

# ─── PAPER DESIGN TOKENS ──────────────────────────────────────────────────────
# Primary: #111111 | Secondary: #8B5CF6 | Success: #16A34A
# Warning: #D97706 | Danger: #DC2626 | Surface: #FFFFFF | Text: #111827
# Bg: #F8F6F1 | Sidebar bg: #FFFFFF | Border: #E2DDD8

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Montserrat:wght@600;700&display=swap');

/* ── Force CSS variables (overrides Streamlit dark theme vars) ── */
:root, [data-theme="dark"], [data-theme="light"] {
    --background-color: #F8F6F1 !important;
    --secondary-background-color: #EDEAE4 !important;
    --text-color: #111827 !important;
    --primary-color: #111111 !important;
    --font: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Global base ── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    background-color: #F8F6F1 !important;
    color: #111827 !important;
    font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1560px !important;
}

/* ── Sidebar — Paper filter pane ── */
/* Apply border-right only to the outermost sidebar element to avoid stacking borders */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2DDD8 !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"] {
    background-color: #FFFFFF !important;
    border-right: none !important;
}

/* Sidebar text — target specific text nodes only, NOT * (breaks Streamlit icon fonts) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] small {
    font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

section[data-testid="stSidebar"] strong {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* Expander label text only — leave SVG/icon spans alone */
section[data-testid="stSidebar"] details > summary p {
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* ── All inputs light ── */
input, textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="input"] > div,
[data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-color: #E2DDD8 !important;
}

/* ── Select / Multiselect ── */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-color: #E2DDD8 !important;
}
/* Tag crop fix:
   - Pad the ValueContainer (data-baseweb="input") directly — that's where tags live
   - Increase tag left margin so first tag never abuts the left clip edge */
[data-testid="stMultiSelect"] [data-baseweb="input"] {
    padding-left: 12px !important;
    overflow: visible !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #F3F0EC !important;
    border: 1px solid #E2DDD8 !important;
    margin: 2px 3px 2px 8px !important;
    overflow: visible !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #111827 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background-color: #E2DDD8 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"],
[data-testid="stExpander"] > div,
details {
    background-color: #FDFCFA !important;
    border: 1px solid #EDE9E4 !important;
    border-radius: 6px !important;
}
details summary {
    color: #374151 !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
}

/* ── Divider ── */
hr {
    border-color: #EDE9E4 !important;
    margin: 12px 0 !important;
}

/* ── Tab strip ── */
div[data-baseweb="tab-list"] {
    background-color: #EDE9E4 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 4px 4px 0 4px !important;
    gap: 2px !important;
    border-bottom: none !important;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #6B7280 !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    font-weight: 600 !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: rgba(0,0,0,0.04) !important;
    color: #374151 !important;
}
[data-testid="stTabsContent"],
[role="tabpanel"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DDD8 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 20px !important;
}

/* ── Plotly charts container ── */
div[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DDD8 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
div[data-testid="stPlotlyChart"] > div {
    background-color: #FFFFFF !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] iframe {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DDD8 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Buttons ── */
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DDD8 !important;
    color: #374151 !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    background-color: #F3F0EC !important;
    border-color: #C9C4BE !important;
}

/* ── Caption / footer ── */
[data-testid="stCaptionContainer"] p {
    color: #9CA3AF !important;
    font-size: 0.74rem !important;
}

/* ── Warning / info banners ── */
[data-testid="stAlert"] {
    background-color: #FFFBEB !important;
    border: 1px solid #FCD34D !important;
    color: #92400E !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D5D0CA; border-radius: 3px; }

/* ── Formula box ── */
.formula {
    background: #F8F6F1;
    border: 1px solid #E2DDD8;
    border-left: 3px solid #111111;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-family: 'PT Mono', 'Courier New', monospace !important;
    font-size: 11.5px;
    line-height: 1.9;
    color: #374151;
    margin: 10px 0;
}

/* ── Section label ── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 24px 0 10px 0;
    font-family: 'Roboto', sans-serif;
}


</style>
"""


# ─── PAPER PLOTLY LAYOUT ──────────────────────────────────────────────────────
PAPER_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Roboto, -apple-system, sans-serif", color="#374151", size=12),
    xaxis=dict(
        gridcolor="#F0EDE8",
        linecolor="#E2DDD8",
        tickfont=dict(color="#6B7280", size=11),
        title_font=dict(color="#374151", size=12),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#F0EDE8",
        linecolor="#E2DDD8",
        tickfont=dict(color="#6B7280", size=11),
        title_font=dict(color="#374151", size=12),
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#374151"),
        orientation="h",
        y=1.13,
    ),
    margin=dict(l=16, r=16, t=8, b=16),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#E2DDD8",
        font=dict(size=12, color="#111827", family="Roboto, sans-serif"),
    ),
)


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def formula_box(html: str) -> None:
    st.markdown(f'<div class="formula">{html}</div>', unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    accent: str = "",
) -> str:
    """Paper-themed KPI card. accent sets left border color when provided."""
    border_left = f"border-left: 3px solid {accent};" if accent else "border-left: 3px solid transparent;"
    return (
        f'<div style="background:#FFFFFF;border:1px solid #E2DDD8;border-radius:6px;'
        f'padding:16px 18px 14px;{border_left}height:100%;">'
        f'<p style="margin:0 0 8px 0;font-size:10px;font-weight:700;letter-spacing:0.10em;'
        f'text-transform:uppercase;color:#9CA3AF;font-family:Roboto,sans-serif;">{label}</p>'
        f'<p style="margin:0;font-size:26px;font-weight:700;color:#111827;letter-spacing:-0.02em;'
        f'line-height:1.1;font-family:Roboto,sans-serif;">{value}</p>'
        + (
            f'<p style="margin:6px 0 0 0;font-size:11px;color:#9CA3AF;'
            f'font-family:Roboto,sans-serif;line-height:1.4;">{subtext}</p>'
            if subtext
            else ""
        )
        + "</div>"
    )


# ─── DATA FETCH ───────────────────────────────────────────────────────────────
def _from_throughput_csv() -> Optional[pd.DataFrame]:
    """
    Source of truth: Superset throughput export.
    Order/unit volume is at fo.created_at hour grain so it aligns with the
    Rolling Hourly Orders CSV.  Manpower (active_pickers/packers) is at the
    PICKED/PACKED audit-event hour grain.
    """
    source_path = next(
        (p for p in [THROUGHPUT_CSV, LOCAL_THROUGHPUT_CSV, REFERENCE_THROUGHPUT_CSV] if p.exists()),
        None,
    )
    if source_path is None:
        return None
    df = pd.read_csv(source_path)

    # Prefer created-at columns; fall back to picked-event columns for old exports.
    if "orders_created" in df.columns:
        order_col = "orders_created"
    else:
        order_col = "orders_picked"
    if "units_requested" in df.columns:
        unit_col = "units_requested"
    else:
        unit_col = "units_picked"

    out = pd.DataFrame(
        {
            "order_date": pd.to_datetime(df["event_date"]).dt.date,
            "order_hour": df["event_hour"].astype(int),
            "ch_id": df["ch_id"],
            "order_count": pd.to_numeric(df[order_col], errors="coerce").fillna(0).astype(int),
            "total_units": pd.to_numeric(df[unit_col], errors="coerce").fillna(0).astype(int),
        }
    )

    optional_cols = {
        "active_pickers": "actual_pickers",
        "active_packers": "actual_packers",
        "orders_picked": "orders_picked",
        "orders_packed": "orders_packed",
        "units_picked": "units_picked",
        "units_packed": "units_packed",
    }
    for source_col, target_col in optional_cols.items():
        if source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0).astype(int)
    return out[out["order_count"] > 0].reset_index(drop=True)


def _fetch_from_superset() -> Optional[pd.DataFrame]:
    try:
        from src.clients.superset import SupersetClient
        from src.settings import settings
    except Exception as exc:
        st.session_state["_fetch_error"] = f"Superset client not importable: {exc}"
        return None
    for _ in range(3):
        try:
            client = SupersetClient(
                base_url=settings.superset_base_url,
                username=settings.superset_username,
                password=settings.superset_password,
                provider=settings.superset_provider,
                timeout_seconds=settings.superset_request_timeout_seconds,
            )
            auth = client.authenticate()
            resp = client.execute_sql(
                auth,
                database_id=settings.superset_database_id,
                sql=REFERENCE_SQL,
                limit=200_000,
            )
            cols, rows = SupersetClient.extract_rows(resp)
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            continue
    return None


# Real hourly order distribution per CH (avg orders/hr) — sourced from
# Rolling Hourly Orders actuals (28-day aggregate, all Blr CHs).
# Sum = ~232 orders/CH/day, matching the real network average.
_MOCK_HOURLY_PROFILE: dict[int, float] = {
    6: 8.4, 7: 19.9, 8: 28.7, 9: 27.6, 10: 22.6,
    11: 18.3, 12: 15.7, 13: 12.9, 14: 10.8, 15: 11.1,
    16: 13.4, 17: 15.6, 18: 16.8, 19: 16.1, 20: 13.3,
    21: 9.7, 22: 5.2, 23: 2.3,
}
_MOCK_PROFILE_TOTAL: float = sum(_MOCK_HOURLY_PROFILE.values())  # 231.8

# Per-CH average daily orders from real data (same 28-day window).
_MOCK_CH_DAILY: dict[str, int] = {
    "FCHBLRSJR01": 394, "FCHBLRHOO01": 347, "FCHBLRJPN01": 297,
    "FCHBLRKOR01": 286, "FCHBLRWHF01": 268, "FCHBLRHAR01": 268,
    "FCHBLRAECS01": 260, "FCHBLRHSR01": 254, "FCHBLRBEN01": 248,
    "FCHBLRHEB01": 240, "FCHBLRDOD01": 240, "FCHBLRBHO01": 226,
    "FCHBLRVAR01": 215, "FCHBLRELC01": 199, "FCHBLRTHA01": 194,
    "FCHBLRHSAF01": 185, "FCHBLRKNP01": 94, "FCHBLRRAJ01": 37,
    "FCHHYDNAR01": 16,
}


def _generate_mock_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(1, 61)]
    chs = list(WH_NAME_MAP.keys())
    rows: list[dict] = []
    for d in dates:
        weekend_boost = 1.15 if d.weekday() >= 5 else 1.0
        for h, base_orders in _MOCK_HOURLY_PROFILE.items():
            for ch in chs:
                daily_avg = _MOCK_CH_DAILY.get(ch, 200)
                ch_scale = daily_avg / _MOCK_PROFILE_TOTAL
                orders = max(0, int(base_orders * ch_scale * weekend_boost * rng.normal(1.0, 0.12)))
                upo = max(1.5, rng.normal(6.5, 1.0))
                if orders > 0:
                    rows.append(
                        {
                            "order_date": d,
                            "order_hour": h,
                            "ch_id": ch,
                            "order_count": orders,
                            "total_units": int(orders * upo),
                        }
                    )
    return pd.DataFrame(rows)


def _from_hourly_orders_csv() -> Optional[pd.DataFrame]:
    """
    Parse the wide-format Rolling Hourly Orders CSV.
    Checks repo data/reference/ first (works on cloud), then local Downloads.
    Format: order_date, CH, 06:00-07:00, ..., Total
    Returns long-format: order_date, order_hour, ch_id, order_count, total_units
    UPO assumed at 6.5 (historical average basket size).
    """
    candidates = [
        REPO_ROOT / "data" / "reference" / "hourly_orders.csv",
        Path.home() / "Downloads" / "Rolling Hourly Orders - hourly_orders.csv",
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return None
    raw = pd.read_csv(p)
    raw = raw[raw["CH"] != "all"].copy()
    raw["order_date"] = pd.to_datetime(raw["order_date"], dayfirst=True, errors="coerce").dt.date
    raw = raw.dropna(subset=["order_date"])
    hour_cols = {
        col: int(col.split(":")[0])
        for col in raw.columns
        if ":" in col and "-" in col and col[:2].isdigit()
    }
    if not hour_cols:
        return None
    rows: list[dict] = []
    for _, row in raw.iterrows():
        for col, h in hour_cols.items():
            val = pd.to_numeric(str(row[col]).replace(",", ""), errors="coerce")
            if pd.notna(val) and val > 0:
                rows.append(
                    {
                        "order_date": row["order_date"],
                        "order_hour": h,
                        "ch_id": row["CH"],
                        "order_count": int(val),
                        "total_units": int(val * 6.5),
                    }
                )
    return pd.DataFrame(rows) if rows else None


def _verification_delta(
    superset_df: pd.DataFrame, csv_df: Optional[pd.DataFrame]
) -> Optional[dict]:
    """
    Compare Superset (source of truth) daily order totals to Rolling Hourly
    Orders CSV and return reconciliation stats. Used as a sanity check banner.
    """
    if csv_df is None or csv_df.empty or superset_df.empty:
        return None
    sup_day = (
        superset_df.groupby(["order_date", "ch_id"], as_index=False)["order_count"].sum()
    )
    csv_day = (
        csv_df.groupby(["order_date", "ch_id"], as_index=False)["order_count"].sum()
        .rename(columns={"order_count": "csv_orders"})
    )
    merged = sup_day.merge(csv_day, on=["order_date", "ch_id"], how="inner")
    if merged.empty:
        return None
    merged["delta"] = merged["order_count"] - merged["csv_orders"]
    return {
        "rows": int(len(merged)),
        "days": int(merged["order_date"].nunique()),
        "mean_abs_delta": float(merged["delta"].abs().mean()),
        "max_abs_delta": int(merged["delta"].abs().max()),
        "p95_abs_delta": float(merged["delta"].abs().quantile(0.95)),
        "latest_date": str(merged["order_date"].max()),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(source_key: str) -> tuple[pd.DataFrame, str]:
    df: Optional[pd.DataFrame] = None
    label = ""
    live_df: Optional[pd.DataFrame] = None
    if source_key == "live":
        live_df = _fetch_from_superset()
        if live_df is not None and not live_df.empty:
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            out = SNAPSHOTS_DIR / f"picker_sim_{stamp}.csv"
            live_df.to_csv(out, index=False)
    throughput_df = _from_throughput_csv()
    hourly_orders_df = _from_hourly_orders_csv()

    if df is None and throughput_df is not None:
        df = throughput_df
        label = "Superset throughput (source of truth) · created_at hour grain"
        if source_key == "live" and live_df is not None and not live_df.empty:
            label += " · refreshed from Superset"
        # Persist verification stats for sidebar banner.
        verif = _verification_delta(throughput_df, hourly_orders_df)
        if verif is not None:
            st.session_state["_verification"] = verif
        else:
            st.session_state.pop("_verification", None)
    if df is None and hourly_orders_df is not None:
        df = hourly_orders_df
        label = "Rolling Hourly Orders CSV (Superset throughput unavailable)"
    if df is None and live_df is not None and not live_df.empty:
        df = live_df
        label = "Live Superset (snapshot only)"
    if df is None:
        snaps = sorted(SNAPSHOTS_DIR.glob("picker_sim_*.csv"))
        if snaps:
            df = pd.read_csv(snaps[-1])
            label = f"Snapshot — {snaps[-1].name}"
    if df is None:
        df = _generate_mock_data()
        label = "Synthetic mock data"

    df["order_date"] = pd.to_datetime(df["order_date"]).dt.date
    df["order_hour"] = df["order_hour"].astype(int)
    df["order_count"] = df["order_count"].astype(int)
    df["total_units"] = df["total_units"].astype(int)
    for col in ["actual_pickers", "actual_packers", "orders_packed", "units_packed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = np.nan
    df["ch_name"] = df["ch_id"]
    df["upo"] = np.where(df["order_count"] > 0, df["total_units"] / df["order_count"], 0.0)
    df["dow"] = pd.to_datetime(df["order_date"]).dt.day_name().str[:3]
    return df, label


# ─── FORWARD PLAN HELPERS ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_throughput_baseline() -> Optional[pd.DataFrame]:
    source_path = next(
        (p for p in [THROUGHPUT_CSV, LOCAL_THROUGHPUT_CSV, REFERENCE_THROUGHPUT_CSV] if p.exists()),
        None,
    )
    if source_path is None:
        return None
    df = pd.read_csv(source_path)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    df["event_hour"] = df["event_hour"].astype(int)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_opd_projections(path_str: str) -> Optional[pd.DataFrame]:
    p = Path(path_str)
    if not p.exists():
        return None
    raw = pd.read_csv(p, header=None)
    ch_codes = [c for c in raw.iloc[0, 4:].tolist() if isinstance(c, str) and c.startswith("FC")]
    body = raw.iloc[2:].copy()
    body.columns = ["week", "day", "date", "overall_total"] + ch_codes + [
        f"_unused_{i}" for i in range(len(body.columns) - 4 - len(ch_codes))
    ]
    body = body[["date"] + ch_codes].copy()
    body["date"] = pd.to_datetime(body["date"], format="%d/%m/%Y", errors="coerce").dt.date
    body = body.dropna(subset=["date"])
    long = body.melt(id_vars="date", var_name="ch_id", value_name="projected_orders")
    long["projected_orders"] = (
        long["projected_orders"]
        .astype(str)
        .str.replace(",", "")
        .str.strip()
        .replace({"": None, "nan": None})
        .astype(float)
    )
    long = long.dropna(subset=["projected_orders"])
    long["projected_orders"] = long["projected_orders"].astype(int)
    return long


def derive_per_ch_baseline(
    throughput_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = throughput_df[throughput_df["orders_picked"] > 0].copy()
    by_ch_hour = df.groupby(["ch_id", "event_hour"])["orders_picked"].sum().reset_index()
    ch_totals = by_ch_hour.groupby("ch_id")["orders_picked"].transform("sum")
    by_ch_hour["share"] = np.where(
        ch_totals > 0, by_ch_hour["orders_picked"] / ch_totals, 0.0
    )
    upo_df = (
        df.groupby("ch_id")
        .apply(lambda g: g["units_picked"].sum() / max(g["orders_picked"].sum(), 1))
        .reset_index()
    )
    upo_df.columns = ["ch_id", "upo"]
    return by_ch_hour[["ch_id", "event_hour", "share"]], upo_df


def project_hourly_demand(
    opd_long: pd.DataFrame,
    hourly_share: pd.DataFrame,
    ch_upo: pd.DataFrame,
    selected_dates: list,
    selected_chs: list,
) -> pd.DataFrame:
    f = opd_long[
        opd_long["date"].isin(selected_dates) & opd_long["ch_id"].isin(selected_chs)
    ].copy()
    if f.empty:
        return pd.DataFrame()
    plan = f.merge(hourly_share, on="ch_id", how="left").merge(ch_upo, on="ch_id", how="left")
    plan["projected_orders_in_hour"] = (plan["projected_orders"] * plan["share"]).round(1)
    plan["projected_units_in_hour"] = (plan["projected_orders_in_hour"] * plan["upo"]).round(0)
    return plan[
        [
            "date",
            "ch_id",
            "event_hour",
            "projected_orders",
            "share",
            "upo",
            "projected_orders_in_hour",
            "projected_units_in_hour",
        ]
    ].rename(columns={"event_hour": "hour"})


# ─── FORWARD PLAN HELPERS (v2 — upload-based) ─────────────────────────────────
def derive_hourly_shares(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Derive per-CH hourly share of daily orders and average UPO from loaded data.
    Returns:
        share_df: DataFrame(ch_id, order_hour, share)  — share sums to 1.0 per CH
        ch_upo: Series(ch_id -> avg units_per_order)
    """
    df = df[df["order_count"] > 0].copy()
    agg = df.groupby(["ch_id", "order_hour"])["order_count"].sum().reset_index()
    ch_total = agg.groupby("ch_id")["order_count"].transform("sum")
    agg["share"] = np.where(ch_total > 0, agg["order_count"] / ch_total, 0.0)
    ch_upo = df.groupby("ch_id").apply(
        lambda g: g["total_units"].sum() / max(g["order_count"].sum(), 1)
    )
    return agg[["ch_id", "order_hour", "share"]], ch_upo


def parse_projection_upload(file_obj) -> Optional[pd.DataFrame]:
    """
    Parse user-uploaded order projection CSV.
    Accepts:
      - Long format: date, ch_id, projected_orders
      - Pivot format: date, FCH..., FCH..., ...  (CH codes as column headers)
    Returns long format: date (date), ch_id (str), projected_orders (int)
    """
    raw = pd.read_csv(file_obj)
    raw.columns = [str(c).strip() for c in raw.columns]

    if "ch_id" in raw.columns and "projected_orders" in raw.columns:
        out = raw[["date", "ch_id", "projected_orders"]].copy()
    elif any(c.startswith("FC") for c in raw.columns):
        ch_cols = [c for c in raw.columns if c.startswith("FC")]
        date_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
        out = (
            raw[[date_col] + ch_cols]
            .melt(id_vars=date_col, var_name="ch_id", value_name="projected_orders")
            .rename(columns={date_col: "date"})
        )
    else:
        return None

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out["projected_orders"] = (
        out["projected_orders"].astype(str).str.replace(",", "").str.strip()
        .replace({"": None, "nan": None, "None": None})
    )
    out["projected_orders"] = pd.to_numeric(out["projected_orders"], errors="coerce")
    out = out.dropna(subset=["date", "projected_orders"])
    out["projected_orders"] = out["projected_orders"].astype(int)
    return out if not out.empty else None


def project_from_upload(
    opd_long: pd.DataFrame,
    hourly_share: pd.DataFrame,
    ch_upo: pd.Series,
    selected_dates: list,
    selected_chs: list,
) -> pd.DataFrame:
    """
    Apply per-CH hourly shares to daily projected orders.
    Returns a row per (date, ch_id, hour) with projected demand and units.
    """
    f = opd_long[
        opd_long["date"].isin(selected_dates) & opd_long["ch_id"].isin(selected_chs)
    ].copy()
    if f.empty:
        return pd.DataFrame()
    upo_df = ch_upo.reset_index()
    upo_df.columns = ["ch_id", "upo"]
    share = hourly_share.rename(columns={"order_hour": "hour"})
    plan = (
        f.merge(share, on="ch_id", how="left")
         .merge(upo_df, on="ch_id", how="left")
    )
    plan["upo"] = plan["upo"].fillna(6.5)
    # Fallback share if CH not in historical data: uniform across 18 hours
    plan["share"] = plan["share"].fillna(1.0 / 18)
    plan["projected_orders_in_hour"] = (plan["projected_orders"] * plan["share"]).round(1)
    plan["projected_units_in_hour"] = (plan["projected_orders_in_hour"] * plan["upo"]).round(0)
    return plan[
        ["date", "ch_id", "hour", "projected_orders", "share", "upo",
         "projected_orders_in_hour", "projected_units_in_hour"]
    ]


# ─── CALCULATION ENGINE ───────────────────────────────────────────────────────
def compute_manpower(
    df: pd.DataFrame,
    *,
    picker_throughput: float,
    packer_throughput: float,
    picker_buffer_pct: float,
    packer_buffer_pct: float,
    max_pickers_per_ch: int = 999,
    max_packers_per_ch: int = 999,
) -> pd.DataFrame:
    out = df.copy()
    units = out["total_units"].astype(float)
    upo = np.where(out["order_count"] > 0, units / out["order_count"], 1.0)

    pickers_uncapped = np.ceil(
        units / max(picker_throughput, 1e-6) * (1 + picker_buffer_pct / 100)
    ).astype(int)
    packers_uncapped = np.ceil(
        units / max(packer_throughput, 1e-6) * (1 + packer_buffer_pct / 100)
    ).astype(int)

    out["pickers_uncapped"] = pickers_uncapped
    out["pickers_required"] = np.minimum(pickers_uncapped, max_pickers_per_ch)
    out["packers_uncapped"] = packers_uncapped
    out["packers_required"] = np.minimum(packers_uncapped, max_packers_per_ch)

    pick_gap = np.maximum(0, pickers_uncapped - out["pickers_required"]).astype(float)
    pack_gap = np.maximum(0, packers_uncapped - out["packers_required"]).astype(float)

    out["pick_breach_orders"] = np.where(
        upo > 0, np.round(pick_gap * picker_throughput / upo).astype(int), 0
    ).astype(int)
    out["pack_breach_orders"] = np.where(
        upo > 0, np.round(pack_gap * packer_throughput / upo).astype(int), 0
    ).astype(int)
    out["pick_breach_per_order_s"] = np.where(
        out["pick_breach_orders"] > 0,
        np.round(upo * 3600.0 / max(picker_throughput, 1e-6)).astype(int),
        0,
    ).astype(int)
    out["pack_breach_per_order_s"] = np.where(
        out["pack_breach_orders"] > 0,
        np.round(upo * 3600.0 / max(packer_throughput, 1e-6)).astype(int),
        0,
    ).astype(int)
    out["uph_picker"] = np.where(
        out["pickers_required"] > 0, units / out["pickers_required"], 0.0
    ).round(1)
    out["uph_packer"] = np.where(
        out["packers_required"] > 0, units / out["packers_required"], 0.0
    ).round(1)
    return out


def compute_distinct_two_shift(
    df: pd.DataFrame,
    *,
    role_col: str,
    shift_hours: int,
    hour_range: tuple,
) -> tuple[float, pd.DataFrame]:
    if df.empty:
        return 0.0, pd.DataFrame(columns=["ch_id", "shift_a", "shift_b", "distinct"])
    h0 = hour_range[0]
    work = df.copy()
    work["shift"] = np.where(
        work["order_hour"] < h0 + shift_hours,
        "A",
        np.where(work["order_hour"] < h0 + 2 * shift_hours, "B", "OUT"),
    )
    work = work[work["shift"] != "OUT"]
    peaks = (
        work.groupby(["order_date", "ch_id", "shift"])[role_col]
        .max()
        .unstack("shift", fill_value=0)
    )
    for s in ["A", "B"]:
        if s not in peaks.columns:
            peaks[s] = 0
    peaks["distinct"] = peaks["A"].astype(int) + peaks["B"].astype(int)
    avg = (
        float(peaks.groupby(level="order_date")["distinct"].sum().mean())
        if len(peaks)
        else 0.0
    )
    per_ch = (
        peaks.groupby(level="ch_id")
        .agg(shift_a=("A", "mean"), shift_b=("B", "mean"), distinct=("distinct", "mean"))
        .reset_index()
    )
    return avg, per_ch


# ─── FILTER PANE (SIDEBAR) ────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame, source_label: str) -> dict:
    # Header
    st.sidebar.markdown("**Manpower Simulator**")
    st.sidebar.caption(source_label)

    is_using_mock = "Synthetic" in source_label
    if not is_using_mock:
        if st.sidebar.button("Refresh from Superset", use_container_width=True):
            load_data.clear()
            st.session_state["_data_source"] = "live"
            st.rerun()
        err = st.session_state.get("_fetch_error")
        if err:
            st.sidebar.warning(err)

    verif = st.session_state.get("_verification")
    if verif:
        max_d = verif["max_abs_delta"]
        if max_d <= 5:
            tone = "success"
        elif max_d <= 15:
            tone = "info"
        else:
            tone = "warning"
        msg = (
            f"CSV verification ({verif['days']}d): "
            f"mean Δ {verif['mean_abs_delta']:.1f} / max Δ {max_d} orders·CH⁻¹·day⁻¹. "
            f"Latest match {verif['latest_date']}."
        )
        getattr(st.sidebar, tone)(msg)

    st.sidebar.divider()

    # ── Always-visible primary filters ────────────────────────────────────────
    dates = sorted(df["order_date"].unique())
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min(dates), max(dates)),
        min_value=min(dates),
        max_value=max(dates),
    )

    all_chs = sorted(df["ch_id"].unique())
    sel_chs = st.sidebar.multiselect(
        "Warehouses",
        all_chs,
        default=all_chs,
        help="Exact warehouse_id codes from inventory.fulfilment_orders.",
    )

    st.sidebar.divider()

    # ── Collapsible filter groups ──────────────────────────────────────────────
    with st.sidebar.expander("Time Filters", expanded=False):
        sel_dows = st.multiselect("Day of week", DOW_ORDER, default=DOW_ORDER)
        hour_range = st.slider(
            "Hour of day", 0, 23, (6, 23),
            help="Operational window.",
        )

    with st.sidebar.expander("Picker Parameters", expanded=True):
        time_per_pick = st.slider(
            "Sec / unit (pick)", 10, 90, 30, step=1,
            help="Avg seconds a picker takes per unit.",
        )
        picker_throughput = st.slider(
            "Units / picker / hr",
            30, 300,
            int(round(3600.0 / max(time_per_pick, 1))),
            help="Override to model real-world drag below theoretical.",
        )
        picker_buffer = st.slider(
            "Buffer %", 0, 30, 10, key="pick_buf",
            help="Extra headcount above minimum.",
        )

    with st.sidebar.expander("Packer Parameters", expanded=True):
        time_per_pack = st.slider("Sec / unit (pack)", 5, 60, 15, step=1)
        packer_throughput = st.slider(
            "Units / packer / hr",
            60, 500,
            int(round(3600.0 / max(time_per_pack, 1))),
        )
        packer_buffer = st.slider("Buffer %", 0, 30, 10, key="pack_buf")

    with st.sidebar.expander("Shift & Capacity", expanded=False):
        picker_shift_hours = st.slider("Picker shift (hr)", 6, 12, 9)
        packer_shift_hours = st.slider("Packer shift (hr)", 6, 12, 9)
        max_pickers_per_ch = st.slider(
            "Max pickers / CH", 1, 30, 12,
            help="Hard ceiling per CH per hour. Drives the ideal vs constrained gap.",
        )
        max_packers_per_ch = st.slider("Max packers / CH", 1, 30, 12)

    with st.sidebar.expander("Cost", expanded=False):
        daily_pay_picker = st.slider("Daily pay — picker (₹)", 500, 1500, 750, step=10)
        daily_pay_packer = st.slider("Daily pay — packer (₹)", 500, 1500, 750, step=10)

    return {
        "date_range": date_range,
        "chs": sel_chs,
        "dows": sel_dows,
        "hour_range": hour_range,
        "time_per_pick": time_per_pick,
        "picker_throughput": picker_throughput,
        "picker_buffer": picker_buffer,
        "time_per_pack": time_per_pack,
        "packer_throughput": packer_throughput,
        "packer_buffer": packer_buffer,
        "picker_shift_hours": picker_shift_hours,
        "packer_shift_hours": packer_shift_hours,
        "max_pickers_per_ch": max_pickers_per_ch,
        "max_packers_per_ch": max_packers_per_ch,
        "daily_pay_picker": daily_pay_picker,
        "daily_pay_packer": daily_pay_packer,
    }


def apply_filters(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    if isinstance(p["date_range"], tuple) and len(p["date_range"]) == 2:
        d0, d1 = p["date_range"]
        out = out[(out["order_date"] >= d0) & (out["order_date"] <= d1)]
    out = out[out["ch_id"].isin(p["chs"])]
    out = out[out["dow"].isin(p["dows"])]
    out = out[
        (out["order_hour"] >= p["hour_range"][0])
        & (out["order_hour"] <= p["hour_range"][1])
    ]
    return out


# ─── KPI GRID ─────────────────────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame, params: dict) -> None:
    if df.empty:
        st.warning("No data after filters.")
        return

    n_days = max(df["order_date"].nunique(), 1)
    has_actual_counts = (
        "actual_pickers" in df.columns
        and "actual_packers" in df.columns
        and df["actual_pickers"].notna().any()
        and df["actual_packers"].notna().any()
    )
    picker_count_col = "actual_pickers" if has_actual_counts else "pickers_required"
    packer_count_col = "actual_packers" if has_actual_counts else "packers_required"
    count_source = "Superset active users" if has_actual_counts else "Modeled required"

    avg_pickers = df[picker_count_col].mean()
    avg_packers = df[packer_count_col].mean()

    peak_idx = (df[picker_count_col] + df[packer_count_col]).idxmax()
    peak_row = df.loc[peak_idx]
    peak_total = int(peak_row[picker_count_col] + peak_row[packer_count_col])
    peak_label = f"{peak_row['order_date']}  {int(peak_row['order_hour']):02d}:00  {peak_row['ch_id']}"

    total_orders = df["order_count"].sum()
    total_pick_breach = df["pick_breach_orders"].sum()
    total_pack_breach = df["pack_breach_orders"].sum()
    pick_breach_pct = 100 * total_pick_breach / max(total_orders, 1)
    pack_breach_pct = 100 * total_pack_breach / max(total_orders, 1)

    distinct_pickers_avg, _ = compute_distinct_two_shift(
        df,
        role_col=picker_count_col,
        shift_hours=params["picker_shift_hours"],
        hour_range=params["hour_range"],
    )
    distinct_packers_avg, _ = compute_distinct_two_shift(
        df,
        role_col=packer_count_col,
        shift_hours=params["packer_shift_hours"],
        hour_range=params["hour_range"],
    )
    distinct_pickers = int(np.ceil(distinct_pickers_avg))
    distinct_packers = int(np.ceil(distinct_packers_avg))

    daily_orders = df["order_count"].sum() / n_days
    daily_cost = (
        distinct_pickers * params["daily_pay_picker"]
        + distinct_packers * params["daily_pay_packer"]
    )
    cpo = daily_cost / max(daily_orders, 1)
    uph_picker = df["uph_picker"].replace(0, np.nan).mean()

    section_label("Staffing")
    c = st.columns(5)
    cards_row1 = [
        kpi_card("Avg Pickers / Hr", f"{avg_pickers:.1f}", f"{count_source}, per CH"),
        kpi_card("Avg Packers / Hr", f"{avg_packers:.1f}", f"{count_source}, per CH"),
        kpi_card("Peak Hr Headcount", str(peak_total), peak_label),
        kpi_card(
            f"Distinct Pickers / Day",
            str(distinct_pickers),
            f"{count_source}; 2 shifts × {params['picker_shift_hours']}h",
        ),
        kpi_card(
            f"Distinct Packers / Day",
            str(distinct_packers),
            f"{count_source}; 2 shifts × {params['packer_shift_hours']}h",
        ),
    ]
    for col, card in zip(c, cards_row1):
        with col:
            st.markdown(card, unsafe_allow_html=True)

    section_label("Understaffing Risk & Cost")
    b = st.columns(5)
    cards_row2 = [
        kpi_card(
            "Pick Breach %",
            f"{pick_breach_pct:.1f}%",
            f"{int(total_pick_breach):,} of {int(total_orders):,} orders",
            accent="#DC2626" if pick_breach_pct > 0 else "",
        ),
        kpi_card(
            "Pack Breach %",
            f"{pack_breach_pct:.1f}%",
            f"{int(total_pack_breach):,} of {int(total_orders):,} orders",
            accent="#DC2626" if pack_breach_pct > 0 else "",
        ),
        kpi_card("UPH — Picker (0 Breach)", f"{uph_picker:.0f}", "Units / picker / hr"),
        kpi_card(
            "Daily Manpower Cost",
            f"₹{daily_cost:,.0f}",
            f"{distinct_pickers}P + {distinct_packers}Pk",
        ),
        kpi_card(
            "Cost Per Order",
            f"₹{cpo:.1f}",
            f"Avg {daily_orders:,.0f} orders / day",
        ),
    ]
    for col, card in zip(b, cards_row2):
        with col:
            st.markdown(card, unsafe_allow_html=True)


# ─── TAB 1: HOURLY MANPOWER ───────────────────────────────────────────────────
def render_tab_hourly(df: pd.DataFrame, params: dict) -> None:
    if df.empty:
        st.info("No data.")
        return
    has_actual_counts = (
        "actual_pickers" in df.columns
        and "actual_packers" in df.columns
        and df["actual_pickers"].notna().any()
        and df["actual_packers"].notna().any()
    )

    st.markdown(
        '<p style="font-size:12px;color:#6B7280;margin:0 0 16px 0;">'
        + (
            "<b style='color:#374151;'>Actual</b> (solid) — active users from Superset audit logs. "
            if has_actual_counts
            else ""
        )
        + "<b style='color:#374151;'>Constrained</b> (solid) — headcount deployable under the CH cap. "
        "<b style='color:#374151;'>Ideal</b> (dashed) — headcount needed with no cap. "
        "Shaded area = understaffing exposure."
        "</p>",
        unsafe_allow_html=True,
    )

    hourly_aggs = {
        "pickers_required": ("pickers_required", "mean"),
        "packers_required": ("packers_required", "mean"),
        "pickers_ideal": ("pickers_uncapped", "mean"),
        "packers_ideal": ("packers_uncapped", "mean"),
        "units": ("total_units", "mean"),
    }
    if has_actual_counts:
        hourly_aggs["actual_pickers"] = ("actual_pickers", "mean")
        hourly_aggs["actual_packers"] = ("actual_packers", "mean")

    by_hour = (
        df.groupby("order_hour")
        .agg(**hourly_aggs)
        .reset_index()
    )

    fig = go.Figure()

    # Understaffing gap fills
    for ideal_col, req_col, fill_color in [
        ("pickers_ideal", "pickers_required", "rgba(17,17,17,0.07)"),
        ("packers_ideal", "packers_required", "rgba(139,92,246,0.08)"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=by_hour["order_hour"],
                y=by_hour[ideal_col],
                mode="lines",
                line=dict(width=0),
                fill=None,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=by_hour["order_hour"],
                y=by_hour[req_col],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Ideal lines (dashed, muted)
    fig.add_trace(
        go.Scatter(
            x=by_hour["order_hour"],
            y=by_hour["pickers_ideal"],
            name="Pickers — ideal",
            mode="lines",
            line=dict(color="#9CA3AF", width=1.5, dash="dash"),
            hovertemplate="Pickers ideal: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=by_hour["order_hour"],
            y=by_hour["packers_ideal"],
            name="Packers — ideal",
            mode="lines",
            line=dict(color="#C4B5FD", width=1.5, dash="dash"),
            hovertemplate="Packers ideal: %{y:.1f}<extra></extra>",
        )
    )

    # Constrained lines (solid, prominent)
    fig.add_trace(
        go.Scatter(
            x=by_hour["order_hour"],
            y=by_hour["pickers_required"],
            name="Pickers — modeled constrained",
            mode="lines+markers",
            line=dict(color="#9CA3AF" if has_actual_counts else "#111111", width=2.5),
            marker=dict(size=5, color="#111111"),
            hovertemplate="Pickers modeled constrained: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=by_hour["order_hour"],
            y=by_hour["packers_required"],
            name="Packers — modeled constrained",
            mode="lines+markers",
            line=dict(color="#C4B5FD" if has_actual_counts else "#8B5CF6", width=2.5),
            marker=dict(size=5, color="#8B5CF6"),
            hovertemplate="Packers modeled constrained: %{y:.1f}<extra></extra>",
        )
    )

    if has_actual_counts:
        fig.add_trace(
            go.Scatter(
                x=by_hour["order_hour"],
                y=by_hour["actual_pickers"],
                name="Pickers — actual",
                mode="lines+markers",
                line=dict(color="#111111", width=3),
                marker=dict(size=6, color="#111111"),
                hovertemplate="Pickers actual: %{y:.1f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=by_hour["order_hour"],
                y=by_hour["actual_packers"],
                name="Packers — actual",
                mode="lines+markers",
                line=dict(color="#8B5CF6", width=3),
                marker=dict(size=6, color="#8B5CF6"),
                hovertemplate="Packers actual: %{y:.1f}<extra></extra>",
            )
        )

    # Units on secondary axis
    fig.add_trace(
        go.Scatter(
            x=by_hour["order_hour"],
            y=by_hour["units"],
            name="Avg units / hr",
            mode="lines",
            line=dict(color="#E2DDD8", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="Avg units: %{y:.0f}<extra></extra>",
        )
    )

    layout = dict(**PAPER_LAYOUT)
    layout["height"] = 440
    layout["hovermode"] = "x unified"
    layout["xaxis"] = dict(
        **PAPER_LAYOUT["xaxis"],
        title="Hour of day",
        tickmode="linear",
        dtick=1,
    )
    layout["yaxis"] = dict(
        **PAPER_LAYOUT["yaxis"],
        title="Avg headcount required",
    )
    layout["yaxis2"] = dict(
        title="Avg units / hr",
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#D1C9BE", size=11),
        title_font=dict(color="#D1C9BE", size=11),
    )
    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How is this calculated?", expanded=False):
        actual_formula = (
            "actual_pickers / actual_packers = COUNT(DISTINCT user_id) from "
            "fulfilment_order_audits PICKED/PACKED events<br><br>"
            if has_actual_counts
            else ""
        )
        formula_box(
            actual_formula
            + f"pickers_ideal      = ceil( units / {params['picker_throughput']} "
            f"× {1 + params['picker_buffer']/100:.2f} )  — no cap applied<br>"
            f"pickers_constrained = min( pickers_ideal, {params['max_pickers_per_ch']} )<br><br>"
            f"packers_ideal      = ceil( units / {params['packer_throughput']} "
            f"× {1 + params['packer_buffer']/100:.2f} )  — no cap applied<br>"
            f"packers_constrained = min( packers_ideal, {params['max_packers_per_ch']} )"
        )


# ─── TAB 2: RAW DATA ──────────────────────────────────────────────────────────
def render_tab_raw(df: pd.DataFrame, params: dict) -> None:
    if df.empty:
        st.info("No data.")
        return

    display_cols = [
        "order_date", "order_hour", "ch_id",
        "order_count", "total_units", "upo",
    ]
    has_actual_counts = (
        "actual_pickers" in df.columns
        and "actual_packers" in df.columns
        and df["actual_pickers"].notna().any()
        and df["actual_packers"].notna().any()
    )
    if has_actual_counts:
        display_cols.extend(["actual_pickers", "actual_packers", "orders_packed", "units_packed"])
    display_cols.extend(
        [
            "pickers_uncapped", "pickers_required", "pick_breach_orders",
            "packers_uncapped", "packers_required", "pack_breach_orders",
            "uph_picker", "uph_packer",
        ]
    )

    show = df[display_cols].rename(
        columns={
            "order_date": "Date", "order_hour": "Hour", "ch_id": "CH",
            "order_count": "Orders", "total_units": "Units", "upo": "UPO",
            "actual_pickers": "Actual Pickers", "actual_packers": "Actual Packers",
            "orders_packed": "Orders Packed", "units_packed": "Units Packed",
            "pickers_uncapped": "Pickers Ideal", "pickers_required": "Pickers Constrained",
            "pick_breach_orders": "Pick Breach Orders",
            "packers_uncapped": "Packers Ideal", "packers_required": "Packers Constrained",
            "pack_breach_orders": "Pack Breach Orders",
            "uph_picker": "UPH Picker Required for 0 Breach",
            "uph_packer": "UPH Packer Required for 0 Breach",
        }
    ).sort_values(["Date", "Hour", "CH"], ascending=[False, True, True])

    # Streamlit Cloud can fail while marshalling large pandas Styler payloads.
    # Fall back to plain dataframe rendering for large tables.
    should_use_styler = len(show) <= 10000
    if should_use_styler:
        table_data = show.style.format(
            {
                "UPO": "{:.2f}",
                "UPH Picker Required for 0 Breach": "{:.0f}",
                "UPH Packer Required for 0 Breach": "{:.0f}",
            }
        ).map(
            lambda v: (
                "background-color:#FEF2F2;color:#991B1B;font-weight:600"
                if isinstance(v, (int, float)) and v > 0
                else ""
            ),
            subset=["Pick Breach Orders", "Pack Breach Orders"],
        )
    else:
        st.caption(
            f"Showing {len(show):,} rows without cell styling to keep the app stable on Streamlit Cloud."
        )
        table_data = show

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        height=440,
    )

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download manpower plan (CSV)",
        data=csv,
        file_name=f"manpower_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    with st.expander("Column definitions", expanded=False):
        actual_definition = (
            "Actual Pickers / Packers = distinct audit users in PICKED/PACKED events for that CH-hour<br>"
            if has_actual_counts
            else ""
        )
        formula_box(
            actual_definition
            + f"Pickers Ideal      = ceil( Units / {params['picker_throughput']} "
            f"× {1 + params['picker_buffer']/100:.2f} )  uncapped need<br>"
            f"Pickers Constrained = min( Pickers Ideal, {params['max_pickers_per_ch']} )<br>"
            "Pick Breach Orders  = (Pickers Ideal − Pickers Constrained) × throughput / UPO<br>"
            f"Packers Ideal      = ceil( Units / {params['packer_throughput']} "
            f"× {1 + params['packer_buffer']/100:.2f} )<br>"
            f"Packers Constrained = min( Packers Ideal, {params['max_packers_per_ch']} )<br>"
            "UPH Picker Required for 0 Breach = Units / Pickers Constrained"
        )

    with st.expander("Reference SQL", expanded=False):
        st.code(REFERENCE_SQL, language="sql")


# ─── TAB 3: FORWARD PLAN ──────────────────────────────────────────────────────
def render_tab_forward(params: dict, df_raw: pd.DataFrame) -> None:
    st.markdown(
        '<p style="font-size:12px;color:#6B7280;margin:0 0 16px 0;">'
        "Upload CH-level daily order projections to get an hourly manpower plan. "
        "Hourly distribution is derived from your loaded historical data."
        "</p>",
        unsafe_allow_html=True,
    )

    if df_raw.empty:
        st.error("No historical data loaded. Cannot derive hourly distribution.")
        return

    hourly_share, ch_upo = derive_hourly_shares(df_raw)

    # ── Template download ──────────────────────────────────────────────────────
    today = datetime.now().date()
    template_rows = [
        {
            "date": (today + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            "ch_id": ch_id,
            "projected_orders": _MOCK_CH_DAILY.get(ch_id, 200),
        }
        for i in range(7)
        for ch_id in WH_NAME_MAP
    ]
    template_csv = pd.DataFrame(template_rows).to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns([2, 5])
    with c1:
        st.download_button(
            "Download projection template",
            data=template_csv,
            file_name=f"order_projection_template_{today}.csv",
            mime="text/csv",
            help="Pre-filled with next 7 days × all CHs. Edit and re-upload.",
        )

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload order projections CSV",
        type="csv",
        help=(
            "Required columns: date (YYYY-MM-DD or DD/MM/YYYY), ch_id, projected_orders.  \n"
            "Also accepts pivot format with CH codes (FCH...) as column headers."
        ),
    )
    if uploaded is None:
        st.info("Upload a CSV to generate the manpower plan. Use the template above as a starting point.")
        return

    opd = parse_projection_upload(uploaded)
    if opd is None or opd.empty:
        st.error(
            "Could not parse the uploaded file.  \n"
            "Expected columns: `date`, `ch_id`, `projected_orders`.  \n"
            "Download the template above to see the correct format."
        )
        return

    # ── Projection scope selectors ─────────────────────────────────────────────
    available_dates = sorted(opd["date"].unique())
    available_chs = sorted(
        set(opd["ch_id"].unique()) & set(hourly_share["ch_id"].unique())
    )
    missing_chs = sorted(set(opd["ch_id"].unique()) - set(hourly_share["ch_id"].unique()))
    if missing_chs:
        st.caption(
            f"CHs with no historical data (excluded): {', '.join(missing_chs)}"
        )
    if not available_chs:
        st.warning("None of the CHs in your upload have historical data. Cannot project.")
        return

    col_d, col_c = st.columns([3, 2])
    with col_d:
        sel_dates = st.multiselect(
            "Dates to project",
            available_dates,
            default=available_dates[:min(5, len(available_dates))],
            format_func=lambda d: d.strftime("%a %d %b %Y") if hasattr(d, "strftime") else str(d),
        )
    with col_c:
        sel_chs = st.multiselect("CHs to include", available_chs, default=available_chs)

    if not sel_dates or not sel_chs:
        st.info("Select at least one date and one CH.")
        return

    # ── Build plan ─────────────────────────────────────────────────────────────
    plan = project_from_upload(opd, hourly_share, ch_upo, sel_dates, sel_chs)
    if plan.empty:
        st.warning("No projection rows — check that uploaded CH codes match historical data.")
        return

    raw_pick = plan["projected_units_in_hour"] / max(params["picker_throughput"], 1e-6)
    raw_pack = plan["projected_units_in_hour"] / max(params["packer_throughput"], 1e-6)
    plan["pickers_required"] = np.minimum(
        np.ceil(raw_pick * (1 + params["picker_buffer"] / 100)).astype(int),
        params["max_pickers_per_ch"],
    )
    plan["packers_required"] = np.minimum(
        np.ceil(raw_pack * (1 + params["packer_buffer"] / 100)).astype(int),
        params["max_packers_per_ch"],
    )

    h0 = params["hour_range"][0]
    plan["picker_shift"] = np.where(
        plan["hour"] < h0 + params["picker_shift_hours"],
        "A",
        np.where(plan["hour"] < h0 + 2 * params["picker_shift_hours"], "B", "OUT"),
    )
    plan["packer_shift"] = np.where(
        plan["hour"] < h0 + params["packer_shift_hours"],
        "A",
        np.where(plan["hour"] < h0 + 2 * params["packer_shift_hours"], "B", "OUT"),
    )

    pick_peaks = (
        plan[plan["picker_shift"] != "OUT"]
        .groupby(["date", "ch_id", "picker_shift"])["pickers_required"]
        .max()
        .unstack("picker_shift", fill_value=0)
    )
    pack_peaks = (
        plan[plan["packer_shift"] != "OUT"]
        .groupby(["date", "ch_id", "packer_shift"])["packers_required"]
        .max()
        .unstack("packer_shift", fill_value=0)
    )
    for peaks in [pick_peaks, pack_peaks]:
        for s in ["A", "B"]:
            if s not in peaks.columns:
                peaks[s] = 0

    pick_peaks["distinct"] = pick_peaks["A"].astype(int) + pick_peaks["B"].astype(int)
    pack_peaks["distinct"] = pack_peaks["A"].astype(int) + pack_peaks["B"].astype(int)
    pick_per_day = pick_peaks.groupby(level="date")["distinct"].sum()
    pack_per_day = pack_peaks.groupby(level="date")["distinct"].sum()

    summary = (
        plan.groupby("date")
        .agg(
            total_orders=("projected_orders", lambda s: int(s.drop_duplicates().sum())),
            total_units=("projected_units_in_hour", "sum"),
            peak_pickers=("pickers_required", "max"),
            peak_packers=("packers_required", "max"),
        )
        .reset_index()
    )
    summary["distinct_pickers"] = summary["date"].map(pick_per_day).fillna(0).astype(int)
    summary["distinct_packers"] = summary["date"].map(pack_per_day).fillna(0).astype(int)
    summary["daily_cost"] = (
        summary["distinct_pickers"] * params["daily_pay_picker"]
        + summary["distinct_packers"] * params["daily_pay_packer"]
    )
    summary["cpo"] = (
        summary["daily_cost"] / summary["total_orders"].replace(0, np.nan)
    ).round(1)

    st.markdown(
        '<p style="font-size:11px;font-weight:600;color:#374151;'
        'letter-spacing:0.05em;text-transform:uppercase;margin:0 0 8px 0;">Day-level summary</p>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        summary.assign(date=summary["date"].astype(str))
        .rename(
            columns={
                "date": "Date", "total_orders": "Projected Orders",
                "total_units": "Projected Units", "peak_pickers": "Peak Hr Pickers",
                "peak_packers": "Peak Hr Packers",
                "distinct_pickers": f"Distinct Pickers (2×{params['picker_shift_hours']}h)",
                "distinct_packers": f"Distinct Packers (2×{params['packer_shift_hours']}h)",
                "daily_cost": "Daily Cost (₹)", "cpo": "CPO (₹)",
            }
        )
        .style.format(
            {
                "Projected Orders": "{:,}",
                "Projected Units": "{:,.0f}",
                "Daily Cost (₹)": "₹{:,.0f}",
                "CPO (₹)": "₹{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<p style="font-size:11px;font-weight:600;color:#374151;'
        'letter-spacing:0.05em;text-transform:uppercase;margin:16px 0 8px 0;">Headcount by hour</p>',
        unsafe_allow_html=True,
    )
    by_dh = (
        plan.groupby(["date", "hour"])
        .agg(pickers=("pickers_required", "sum"), packers=("packers_required", "sum"))
        .reset_index()
    )
    by_dh["date_str"] = by_dh["date"].astype(str)

    palette = ["#111111", "#8B5CF6", "#D97706", "#DC2626", "#16A34A"]
    fig = go.Figure()
    for i, d in enumerate(sorted(by_dh["date_str"].unique())):
        color = palette[i % len(palette)]
        sub = by_dh[by_dh["date_str"] == d].sort_values("hour")
        fig.add_trace(
            go.Scatter(
                x=sub["hour"], y=sub["pickers"],
                name=f"Pickers · {d}",
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["hour"], y=sub["packers"],
                name=f"Packers · {d}",
                mode="lines+markers",
                line=dict(color=color, dash="dot", width=2),
                marker=dict(size=5, symbol="diamond"),
            )
        )

    layout = dict(**PAPER_LAYOUT)
    layout["height"] = 400
    layout["hovermode"] = "x unified"
    layout["xaxis"] = dict(
        **PAPER_LAYOUT["xaxis"], title="Hour of day", tickmode="linear", dtick=1
    )
    layout["yaxis"] = dict(**PAPER_LAYOUT["yaxis"], title="Total headcount (across CHs)")
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p style="font-size:11px;font-weight:600;color:#374151;'
        'letter-spacing:0.05em;text-transform:uppercase;margin:16px 0 8px 0;">Per-CH breakdown</p>',
        unsafe_allow_html=True,
    )
    by_ch_date = (
        plan.groupby(["date", "ch_id"])
        .agg(
            projected_orders=("projected_orders", "first"),
            projected_units=("projected_units_in_hour", "sum"),
            peak_pickers=("pickers_required", "max"),
            peak_packers=("packers_required", "max"),
        )
        .reset_index()
        .sort_values(["date", "projected_orders"], ascending=[True, False])
    )
    st.dataframe(
        by_ch_date.assign(date=by_ch_date["date"].astype(str))
        .rename(
            columns={
                "date": "Date", "ch_id": "CH",
                "projected_orders": "Daily Orders", "projected_units": "Daily Units",
                "peak_pickers": "Peak Hr Pickers", "peak_packers": "Peak Hr Packers",
            }
        )
        .style.format({"Daily Orders": "{:,}", "Daily Units": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    csv_out = plan.assign(date=plan["date"].astype(str)).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forward plan (CSV)",
        data=csv_out,
        file_name=f"forward_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    apply_styles()

    # Page header
    st.markdown(
        '<h1 style="font-family:Montserrat,Roboto,sans-serif;font-size:22px;font-weight:700;'
        'color:#111827;letter-spacing:-0.02em;margin:0 0 4px 0;">'
        "Picker & Packer Manpower Simulator"
        "</h1>"
        '<p style="font-size:13px;color:#9CA3AF;margin:0 0 20px 0;font-family:Roboto,sans-serif;">'
        "Hourly required headcount per CH · 4 weeks of actuals · Adjust sliders to model what-if scenarios"
        "</p>",
        unsafe_allow_html=True,
    )

    source_key = st.session_state.get("_data_source", "snapshot")
    df_raw, source_label = load_data(source_key)
    if source_key == "live":
        st.session_state["_data_source"] = "snapshot"

    params = render_sidebar(df_raw, source_label)
    df_filtered = apply_filters(df_raw, params)
    df_computed = compute_manpower(
        df_filtered,
        picker_throughput=params["picker_throughput"],
        packer_throughput=params["packer_throughput"],
        picker_buffer_pct=params["picker_buffer"],
        packer_buffer_pct=params["packer_buffer"],
        max_pickers_per_ch=params["max_pickers_per_ch"],
        max_packers_per_ch=params["max_packers_per_ch"],
    )

    render_kpis(df_computed, params)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Hourly Manpower", "Raw Data", "Forward Plan"])
    with tab1:
        render_tab_hourly(df_computed, params)
    with tab2:
        render_tab_raw(df_computed, params)
    with tab3:
        render_tab_forward(params, df_raw)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.caption(
        f"Data: {source_label} · {len(df_raw):,} rows · "
        f"Refreshed: {datetime.now().strftime('%d %b %Y %H:%M IST')}"
    )


if __name__ == "__main__":
    main()
