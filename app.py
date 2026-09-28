import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Soybean Crush Margin Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local dev: set these in .streamlit/secrets.toml (gitignored).
# Deployed: set these in Settings > Secrets on share.streamlit.io.
try:
    FRED_KEY = st.secrets["FRED_KEY"]
except Exception:
    st.error(
        "Missing FRED_KEY. Add it to .streamlit/secrets.toml locally, "
        "or in Settings > Secrets when deployed on Streamlit Community Cloud."
    )
    st.stop()

# Optional, unlocks daily (not monthly) cash prices and USDA stocks-to-use
# data. The app runs fine without these — it just falls back to FRED's
# monthly global price series for everything.
try:
    USDA_AMS_KEY = st.secrets["USDA_AMS_KEY"]
except Exception:
    USDA_AMS_KEY = None
try:
    USDA_FAS_KEY = st.secrets["USDA_FAS_KEY"]
except Exception:
    USDA_FAS_KEY = None
try:
    USDA_NASS_KEY = st.secrets["USDA_NASS_KEY"]
except Exception:
    USDA_NASS_KEY = None

# ── Colour palette ──────────────────────────────────────────────
C = {
    "soybean":    "#D97706",
    "oil":        "#65A30D",
    "meal":       "#7C3AED",
    "margin_pos": "#10B981",
    "margin_neg": "#EF4444",
    "grid":       "rgba(203,213,225,0.4)",
    "bg":         "#FFFFFF",
    "neutral":    "#94A3B8",
}

PLOT_LAYOUT = dict(
    plot_bgcolor=C["bg"],
    paper_bgcolor=C["bg"],
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#1E293B"),
    margin=dict(l=60, r=40, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8FAFC; }
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 16px 20px;
}
div[data-testid="stMetricLabel"] { font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; }
div[data-testid="stMetricValue"] { font-size: 22px; font-weight: 600; color: #0F172A; }
.signal-open {
    background: #F0FDF4; border: 1.5px solid #10B981;
    border-radius: 8px; padding: 20px 24px; margin: 8px 0;
}
.signal-closed {
    background: #FFF1F2; border: 1.5px solid #EF4444;
    border-radius: 8px; padding: 20px 24px; margin: 8px 0;
}
.signal-title { font-size: 18px; font-weight: 600; color: #0F172A; margin-bottom: 6px; }
.signal-body  { font-size: 13px; color: #475569; line-height: 1.7; }
.section-title {
    font-size: 13px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: #64748B;
    border-bottom: 1px solid #E2E8F0; padding-bottom: 6px; margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] { font-size: 13px; font-weight: 500; color: #64748B; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #D97706; border-bottom: 2px solid #D97706; }
div[data-testid="stSidebar"] { background: #F1F5F9; border-right: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ───────────────────────────────────────────────
def _get_with_retry(url: str, params: dict = None, headers: dict = None, auth=None,
                     timeout: int = 25, retries: int = 3, backoff: float = 1.5):
    """GET with retries on transient network errors (timeouts, connection
    drops, 5xx) so one slow response from a free public API doesn't take
    down the whole app.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, auth=auth, timeout=timeout)
            if r.status_code >= 500 and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


@st.cache_data(ttl=3600)
def fetch_fred(series_id: str, days: int = 3650) -> pd.DataFrame:
    """FRED global commodity price (IMF-sourced), monthly, $/metric ton."""
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": FRED_KEY,
        "observation_start": start, "file_type": "json",
        "sort_order": "asc",
    }
    r = _get_with_retry(url, params)
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = df["value"].replace(".", None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).rename(columns={"date": "period"})


@st.cache_data(ttl=3600)
def fetch_ams_daily_price(report_id: str) -> pd.DataFrame:
    """Daily cash grain bids from USDA AMS MyMarketNews (marsapi.ams.usda.gov).
    Returns an empty frame (not an error) if USDA_AMS_KEY isn't set, so the
    app falls back to FRED's monthly series instead of breaking.
    """
    if not USDA_AMS_KEY:
        return pd.DataFrame(columns=["period", "value"])
    try:
        url = f"https://marsapi.ams.usda.gov/services/v1.2/reports/{report_id}"
        r = _get_with_retry(url, auth=(USDA_AMS_KEY, ""))
        rows = r.json().get("results", [])
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"fetch_ams_daily_price({report_id!r}) failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["period", "value"])


# PSD attribute IDs (confirmed against /api/psd/commodityattributes — these
# are not documented anywhere obvious, had to pull the lookup table directly).
PSD_ATTR_ENDING_STOCKS  = 176
PSD_ATTR_EXPORTS        = 88
PSD_ATTR_DOM_CONSUMPTION = 125

# USDA doesn't populate a generic "Total Use" or "Stocks-to-Use" attribute
# for the soybean oilseed commodity specifically (checked — present for some
# commodities, absent here), so stocks-to-use is computed from components
# that USDA does report. The balance identity holds exactly in the raw data
# (Exports + Domestic Consumption + Ending Stocks = Total Supply), confirming
# this is the correct, standard definition, not an approximation.
PSD_YEARS = list(range(2012, datetime.today().year + 1))


@st.cache_data(ttl=3600)
def fetch_psd_stocks_to_use(commodity_code: str, country: str = "US") -> pd.DataFrame:
    """Annual stocks-to-use ratio from USDA FAS PSD Online (api.fas.usda.gov).
    This is this app's equivalent of the Cushing-inventory proxy in the
    oil-market version of the project: the standard agricultural-economics
    measure of how tight the market is. Returns an empty frame if
    USDA_FAS_KEY isn't set or every year's fetch fails.
    """
    if not USDA_FAS_KEY:
        return pd.DataFrame(columns=["market_year", "ending_stocks", "use", "stocks_to_use"])
    rows = []
    for year in PSD_YEARS:
        try:
            url = f"https://api.fas.usda.gov/api/psd/commodity/{commodity_code}/country/{country}/year/{year}"
            r = _get_with_retry(url, headers={"X-Api-Key": USDA_FAS_KEY}, retries=2)
            data = r.json()
        except Exception as e:
            print(f"fetch_psd_stocks_to_use({commodity_code!r}, year={year}) failed: {type(e).__name__}: {e}")
            continue
        by_attr = {row["attributeId"]: row["value"] for row in data}
        stocks = by_attr.get(PSD_ATTR_ENDING_STOCKS)
        exports = by_attr.get(PSD_ATTR_EXPORTS)
        dom_use = by_attr.get(PSD_ATTR_DOM_CONSUMPTION)
        if stocks is None or exports is None or dom_use is None:
            continue
        use = exports + dom_use
        rows.append({
            "market_year": year, "ending_stocks": stocks, "use": use,
            "stocks_to_use": stocks / use if use > 0 else None,
        })
    return pd.DataFrame(rows)


# FAS PSD commodity codes (confirmed live against /api/psd/commodities)
PSD_CODE_SOYBEANS      = "2222000"  # "Oilseed, Soybean"
PSD_CODE_SOYBEAN_MEAL   = "0813100"  # "Meal, Soybean"
PSD_CODE_SOYBEAN_OIL    = "4232000"  # "Oil, Soybean"

# Standard "board crush" yields per bushel (60 lbs) of soybeans, the CBOT
# industry-standard conversion: ~11 lbs oil + ~44 lbs meal per bushel
# processed. The remaining ~5 lbs is hulls/moisture loss and isn't priced.
LBS_PER_BUSHEL       = 60.0
OIL_LBS_PER_BUSHEL   = 11.0
MEAL_LBS_PER_BUSHEL  = 44.0
LBS_PER_METRIC_TON   = 2204.62
LBS_PER_SHORT_TON    = 2000.0
BUSHELS_PER_METRIC_TON = LBS_PER_METRIC_TON / LBS_PER_BUSHEL  # ~36.74


@st.cache_data(ttl=3600)
def build_master() -> dict:
    # FRED global prices, all $/metric ton, monthly — the baseline series
    # that works with zero USDA registration, since crush margin needs all
    # three legs (soybeans, oil, meal) on a common daily-mergeable timeline.
    soy  = fetch_fred("PSOYBUSDM").rename(columns={"value": "soy_usd_tonne"})
    oil  = fetch_fred("PSOILUSDM").rename(columns={"value": "oil_usd_tonne"})
    meal = fetch_fred("PSMEAUSDM").rename(columns={"value": "meal_usd_tonne"})

    df = soy.merge(oil, on="period").merge(meal, on="period").sort_values("period").reset_index(drop=True)

    # Convert to US industry convention units before applying the board-crush formula.
    df["soy_usd_bu"]   = df["soy_usd_tonne"] / BUSHELS_PER_METRIC_TON
    df["oil_cents_lb"] = df["oil_usd_tonne"] / LBS_PER_METRIC_TON * 100
    df["meal_usd_st"]  = df["meal_usd_tonne"] * (LBS_PER_METRIC_TON / LBS_PER_SHORT_TON)  # metric ton -> short ton price

    # Board crush margin, $/bushel: value of the oil + meal one bushel
    # yields, minus the cost of that bushel of soybeans.
    df["oil_value_per_bu"]  = df["oil_cents_lb"] / 100 * OIL_LBS_PER_BUSHEL
    df["meal_value_per_bu"] = df["meal_usd_st"] * (MEAL_LBS_PER_BUSHEL / LBS_PER_SHORT_TON)
    df["crush_margin"]      = df["oil_value_per_bu"] + df["meal_value_per_bu"] - df["soy_usd_bu"]

    # USDA stocks-to-use ratio (optional, needs USDA_FAS_KEY) — the ag
    # equivalent of the Cushing-inventory proxy: how tight the market is
    # relative to how much gets consumed, the standard driver of ag price
    # volatility and seasonality.
    stu = fetch_psd_stocks_to_use(PSD_CODE_SOYBEANS)

    return {
        "crush": df,
        "stocks_to_use": stu,
    }


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Display")
    lookback = st.selectbox("Lookback", ["1 year", "3 years", "5 years", "10 years", "Max"], index=2)
    lb_days = {"1 year": 365, "3 years": 365*3, "5 years": 365*5, "10 years": 365*10, "Max": 365*40}[lookback]

    st.divider()
    st.markdown(
        "<div style='font-size:12px;color:#64748B'>"
        "Crush margin = value of soybean oil + soybean meal from processing "
        "one bushel of soybeans, minus the cost of that bushel. The standard "
        "margin a soybean processor earns."
        "</div>", unsafe_allow_html=True
    )

    if not USDA_FAS_KEY:
        st.divider()
        st.markdown("<div style='font-size:12px;color:#94A3B8'>Stocks-to-use needs USDA_FAS_KEY in secrets.</div>",
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:12px;color:#94A3B8'>Crush margin uses FRED's monthly global prices. "
                "USDA_AMS_KEY is wired for daily cash prices but not yet used in the margin calc.</div>",
                unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────
with st.spinner("Loading market data..."):
    try:
        data = build_master()
        loaded = True
    except Exception as e:
        st.error(f"Data load failed after retries: {e}")
        if st.button("Retry now"):
            st.cache_data.clear()
            st.rerun()
        loaded = False

if not loaded:
    st.stop()

cutoff = datetime.today() - timedelta(days=lb_days)
df = data["crush"][data["crush"]["period"] >= cutoff].copy()

if len(df) == 0:
    st.warning("No data in the selected lookback window.")
    st.stop()

latest = df.iloc[-1]

st.markdown("""
<div style='padding:4px 0 20px 0'>
  <div style='font-size:24px;font-weight:600;color:#0F172A'>Soybean Crush Margin Monitor</div>
  <div style='font-size:13px;color:#64748B;margin-top:4px'>
    Board crush economics: soybean, soybean oil and soybean meal prices, and the processing margin between them
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Snapshot", "Crush Margin History", "Market Structure"])

with tab1:
    st.markdown(f"<div style='font-size:12px;color:#94A3B8;margin-bottom:16px'>Last data: {latest['period'].strftime('%d %b %Y')} (FRED, monthly, IMF-sourced global price)</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Soybeans", f"${latest['soy_usd_bu']:.2f}/bu")
    c2.metric("Soybean Oil", f"{latest['oil_cents_lb']:.1f}¢/lb")
    c3.metric("Soybean Meal", f"${latest['meal_usd_st']:.0f}/ton")
    c4.metric("Crush Margin", f"${latest['crush_margin']:.2f}/bu")

    st.divider()
    css_class = "signal-open" if latest["crush_margin"] > 0 else "signal-closed"
    label = "POSITIVE MARGIN" if latest["crush_margin"] > 0 else "NEGATIVE MARGIN"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="signal-title">{label}</div>
        <div class="signal-body">Current board crush margin is ${latest['crush_margin']:.2f}/bushel.
        {"Processing is economically favorable at current prices." if latest["crush_margin"] > 0 else "Processing economics are unfavorable at current prices; crush plants may reduce run rates."}
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Soybean / Oil / Meal Prices (normalized $/bu basis)", "Crush Margin ($/bushel)"),
        vertical_spacing=0.1, row_heights=[0.5, 0.5]
    )
    fig.add_trace(go.Scatter(x=df["period"], y=df["soy_usd_bu"], name="Soybeans ($/bu)",
        line=dict(color=C["soybean"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["period"], y=df["oil_value_per_bu"], name="Oil value ($/bu equiv)",
        line=dict(color=C["oil"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["period"], y=df["meal_value_per_bu"], name="Meal value ($/bu equiv)",
        line=dict(color=C["meal"], width=2)), row=1, col=1)

    pos = df["crush_margin"].clip(lower=0)
    neg = df["crush_margin"].clip(upper=0)
    fig.add_trace(go.Scatter(x=df["period"], y=pos, name="Positive margin", fill="tozeroy",
        fillcolor="rgba(16,185,129,0.2)", line=dict(color=C["margin_pos"], width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["period"], y=neg, name="Negative margin", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)", line=dict(color=C["margin_neg"], width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1, row=2, col=1)

    # Override the shared legend position for this figure specifically: the
    # default y=1.02 sits right on top of the row-1 subplot title in a
    # multi-row figure (shared legend is figure-level, subplot_titles are
    # per-row), so push it higher and widen the top margin to give both room.
    hist_layout = {**PLOT_LAYOUT, "legend": {**PLOT_LAYOUT["legend"], "y": 1.08},
                   "margin": dict(l=60, r=40, t=70, b=40)}
    fig.update_layout(**hist_layout, height=650)
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"])
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<div class='section-title'>US Soybean Stocks-to-Use Ratio</div>", unsafe_allow_html=True)
    stu = data["stocks_to_use"]
    if len(stu) == 0:
        st.info(
            "USDA_FAS_KEY isn't set (or every request failed), so this section is empty. "
            "This is the standard agricultural-economics measure of market tightness: US soybean "
            "ending stocks divided by total use (exports + domestic consumption) for each marketing year, "
            "sourced live from USDA FAS PSD Online."
        )
    else:
        st.caption(
            "Ending stocks ÷ total use (exports + domestic consumption), by USDA marketing year. "
            "Low values signal a tight market (bullish for prices); high values signal ample supply."
        )
        stu_fig = go.Figure(go.Bar(
            x=stu["market_year"], y=stu["stocks_to_use"] * 100,
            marker_color=C["soybean"],
            text=[f"{v:.1f}%" for v in stu["stocks_to_use"] * 100], textposition="outside",
        ))
        pad = (stu["stocks_to_use"].max() * 100) * 0.2 or 1
        stu_fig.update_layout(**PLOT_LAYOUT, height=340,
                              yaxis_title="Stocks-to-Use (%)", xaxis_title="Marketing Year",
                              showlegend=False, title="US Soybeans: Ending Stocks / Total Use",
                              yaxis=dict(range=[0, stu["stocks_to_use"].max()*100 + pad], showgrid=True, gridcolor=C["grid"]),
                              xaxis=dict(showgrid=False, type="category"))
        st.plotly_chart(stu_fig, use_container_width=True)

        latest_stu = stu.iloc[-1]
        st.markdown(
            f"<div style='font-size:12px;color:#94A3B8'>Most recent marketing year ({int(latest_stu['market_year'])}): "
            f"{latest_stu['stocks_to_use']*100:.1f}% ({latest_stu['ending_stocks']:,.0f} thousand MT ending stocks "
            f"vs {latest_stu['use']:,.0f} thousand MT total use)</div>",
            unsafe_allow_html=True
        )

st.caption(f"Data: FRED (IMF global prices), USDA FAS PSD Online | Built by Nicholas Becnel | {datetime.today().strftime('%d %b %Y')}")
