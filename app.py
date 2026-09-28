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
def fetch_ams_daily_soybean_price() -> pd.DataFrame:
    """Daily soybean cash price from Illinois Grain Bids (AMS_3192), Mississippi
    River barge-loading elevators -- a standard export-linked benchmark, and the
    same crush-belt region used for the oil/meal legs below. USDA publishes two
    quotes per date during a basis revision (current="Yes"/"No"); we keep only
    the current one. Empty frame if USDA_AMS_KEY isn't set or the fetch fails.
    """
    if not USDA_AMS_KEY:
        return pd.DataFrame(columns=["period", "soy_usd_bu"])
    try:
        url = "https://marsapi.ams.usda.gov/services/v1.2/reports/3192/Report%20Detail"
        r = _get_with_retry(url, auth=(USDA_AMS_KEY, ""))
        rows = r.json().get("results", [])
    except Exception as e:
        print(f"fetch_ams_daily_soybean_price failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["period", "soy_usd_bu"])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["period", "soy_usd_bu"])
    df = df[(df["commodity"] == "Soybeans") & (df["trade_loc"] == "Mississippi River")
            & (df["current"] == "Yes")]
    if df.empty:
        return pd.DataFrame(columns=["period", "soy_usd_bu"])
    df["period"] = pd.to_datetime(df["report_date"], format="%m/%d/%Y")
    df = df.rename(columns={"avg_price": "soy_usd_bu"})[["period", "soy_usd_bu"]]
    return df.sort_values("period").drop_duplicates("period").reset_index(drop=True)


@st.cache_data(ttl=3600)
def fetch_ams_weekly_oil_meal() -> pd.DataFrame:
    """Weekly soybean oil (cents/lb) and soybean meal ($/ton) cash prices for
    Illinois, from the National Grain and Oilseed Processor Feedstuff Report
    (AMS_3511). This is USDA's finest free cadence for the two processed legs
    -- there is no daily public source for them, only for the raw bean.
    Uses report_end_date (when that week's price became final) rather than
    report_begin_date, so a later merge_asof against daily soybean prices
    doesn't look ahead of when the quote was actually published. Empty frame
    if USDA_AMS_KEY isn't set or the fetch fails.
    """
    if not USDA_AMS_KEY:
        return pd.DataFrame(columns=["period", "oil_cents_lb", "meal_usd_st"])
    try:
        url = "https://marsapi.ams.usda.gov/services/v1.2/reports/3511/Report%20Detail"
        r = _get_with_retry(url, auth=(USDA_AMS_KEY, ""))
        rows = r.json().get("results", [])
    except Exception as e:
        print(f"fetch_ams_weekly_oil_meal failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["period", "oil_cents_lb", "meal_usd_st"])
    df = pd.DataFrame(rows)
    if df.empty or "trade Loc" not in df.columns:
        return pd.DataFrame(columns=["period", "oil_cents_lb", "meal_usd_st"])

    oil = df[(df["commodity"] == "Soybean Oil") & (df["trade Loc"] == "Illinois")]
    oil = oil[["report_end_date", "avg_price"]].rename(columns={"avg_price": "oil_cents_lb"})

    meal = df[(df["commodity"] == "Soybean Meal") & (df["trade Loc"] == "Illinois")
              & (df["trans_mode"] == "Truck")]
    meal = meal[["report_end_date", "avg_price"]].rename(columns={"avg_price": "meal_usd_st"})

    merged = oil.merge(meal, on="report_end_date", how="outer")
    merged["period"] = pd.to_datetime(merged["report_end_date"], format="%m/%d/%Y")
    merged = merged.drop(columns=["report_end_date"]).sort_values("period")
    return merged.drop_duplicates("period").reset_index(drop=True)


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


@st.cache_data(ttl=3600)
def fetch_nass_soybean_fundamentals() -> pd.DataFrame:
    """US national soybean planted acreage, yield, and production by year
    (USDA NASS QuickStats, annual survey estimates). Upstream supply-side
    context for stocks-to-use: a bad yield year mechanically tightens next
    season's stocks, and planted acreage is the decision farmers make each
    spring partly in response to crush/relative-price economics. Empty frame
    if USDA_NASS_KEY isn't set or every series fails.
    """
    if not USDA_NASS_KEY:
        return pd.DataFrame(columns=["year", "planted_acres", "yield_bu_acre", "production_bu"])
    base = "https://quickstats.nass.usda.gov/api/api_GET/"
    common = {
        "key": USDA_NASS_KEY, "commodity_desc": "SOYBEANS",
        "agg_level_desc": "NATIONAL", "source_desc": "SURVEY",
        "reference_period_desc": "YEAR", "year__GE": 2012, "format": "JSON",
    }
    series = {
        "planted_acres":  "SOYBEANS - ACRES PLANTED",
        "yield_bu_acre":  "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
        "production_bu":  "SOYBEANS - PRODUCTION, MEASURED IN BU",
    }
    frames = []
    for col, short_desc in series.items():
        try:
            r = _get_with_retry(base, params={**common, "short_desc": short_desc}, retries=2)
            rows = r.json().get("data", [])
        except Exception as e:
            print(f"fetch_nass_soybean_fundamentals({col!r}) failed: {type(e).__name__}: {e}")
            continue
        if not rows:
            continue
        fdf = pd.DataFrame(rows)
        fdf[col] = fdf["Value"].str.replace(",", "", regex=False).astype(float)
        frames.append(fdf[["year", col]])
    if not frames:
        return pd.DataFrame(columns=["year", "planted_acres", "yield_bu_acre", "production_bu"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="year", how="outer")
    return out.sort_values("year").reset_index(drop=True)

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

    fred = soy.merge(oil, on="period").merge(meal, on="period").sort_values("period").reset_index(drop=True)

    # Convert to US industry convention units before applying the board-crush formula.
    fred["soy_usd_bu"]   = fred["soy_usd_tonne"] / BUSHELS_PER_METRIC_TON
    fred["oil_cents_lb"] = fred["oil_usd_tonne"] / LBS_PER_METRIC_TON * 100
    fred["meal_usd_st"]  = fred["meal_usd_tonne"] * (LBS_PER_METRIC_TON / LBS_PER_SHORT_TON)
    fred["source"] = "FRED monthly (global, IMF-sourced)"
    fred = fred[["period", "soy_usd_bu", "oil_cents_lb", "meal_usd_st", "source"]]

    # Daily soybean + weekly oil/meal cash prices (USDA AMS), when a key is
    # set. AMS's oil/meal report only publishes weekly, so it's forward-filled
    # onto the daily soybean grid (direction="backward": each day gets the
    # most recent published weekly price, never a future one). FRED covers
    # everything before AMS's coverage window starts, so the lookback
    # selector still has years of history even though AMS itself only goes
    # back to 2022-2023.
    ams_soy = fetch_ams_daily_soybean_price()
    ams_om  = fetch_ams_weekly_oil_meal()
    if not ams_soy.empty and not ams_om.empty:
        ams = pd.merge_asof(ams_soy, ams_om.dropna(subset=["oil_cents_lb", "meal_usd_st"], how="all"),
                             on="period", direction="backward")
        ams = ams.dropna(subset=["oil_cents_lb", "meal_usd_st"], how="any").reset_index(drop=True)
        ams["source"] = "USDA AMS daily cash (Illinois / Mississippi River)"
        ams_start = ams["period"].min()
        df = pd.concat([fred[fred["period"] < ams_start], ams], ignore_index=True)
    else:
        df = fred
    df = df.sort_values("period").reset_index(drop=True)

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

    # US planted acreage / yield / production (optional, needs USDA_NASS_KEY)
    # — the production-side driver upstream of stocks-to-use above.
    fundamentals = fetch_nass_soybean_fundamentals()

    return {
        "crush": df,
        "stocks_to_use": stu,
        "fundamentals": fundamentals,
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
    if USDA_AMS_KEY:
        st.markdown("<div style='font-size:12px;color:#94A3B8'>Recent data is daily soybean cash prices "
                    "(USDA AMS) with weekly oil/meal cash prices forward-filled between updates. "
                    "Older history falls back to FRED's monthly global prices.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px;color:#94A3B8'>Crush margin uses FRED's monthly global prices. "
                    "Set USDA_AMS_KEY in secrets for daily cash prices instead.</div>",
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

tab1, tab2, tab3 = st.tabs(["Snapshot", "Crush Margin History", "Supply & Demand"])

with tab1:
    st.markdown(f"<div style='font-size:12px;color:#94A3B8;margin-bottom:16px'>Last data: {latest['period'].strftime('%d %b %Y')} ({latest['source']})</div>",
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

    st.divider()
    st.markdown("<div class='section-title'>US Planted Acreage &amp; Yield</div>", unsafe_allow_html=True)
    fnd = data["fundamentals"]
    if len(fnd) == 0:
        st.info(
            "USDA_NASS_KEY isn't set (or every request failed), so this section is empty. "
            "Acreage and yield are the production-side drivers upstream of stocks-to-use above: "
            "a weak yield year mechanically tightens next season's stocks, and planted acreage is "
            "the decision farmers make each spring, partly in response to crush economics like the "
            "margin on the other tabs."
        )
    else:
        st.caption(
            "Yield (bars, left axis) and planted acreage (line, right axis) by crop year, USDA NASS "
            "annual survey. A weak yield year tightens next season's stocks-to-use; acreage reflects "
            "what farmers actually planted that spring."
        )
        fyfig = make_subplots(specs=[[{"secondary_y": True}]])
        fyfig.add_trace(go.Bar(
            x=fnd["year"], y=fnd["yield_bu_acre"], name="Yield (bu/acre)",
            marker_color=C["oil"],
            text=[f"{v:.1f}" for v in fnd["yield_bu_acre"]], textposition="outside",
        ), secondary_y=False)
        fyfig.add_trace(go.Scatter(
            x=fnd["year"], y=fnd["planted_acres"] / 1_000_000, name="Planted acres (millions)",
            mode="lines+markers", line=dict(color=C["soybean"], width=2),
        ), secondary_y=True)
        fy_layout = {**PLOT_LAYOUT, "legend": {**PLOT_LAYOUT["legend"], "y": 1.1},
                     "margin": dict(l=60, r=60, t=40, b=40)}
        fyfig.update_layout(**fy_layout, height=340, showlegend=True)
        fyfig.update_xaxes(showgrid=False, type="category", title_text="Crop Year")
        fyfig.update_yaxes(title_text="Yield (bu/acre)", showgrid=True, gridcolor=C["grid"], secondary_y=False)
        fyfig.update_yaxes(title_text="Planted Acres (millions)", showgrid=False, secondary_y=True)
        st.plotly_chart(fyfig, use_container_width=True)

        latest_fnd = fnd.iloc[-1]
        st.markdown(
            f"<div style='font-size:12px;color:#94A3B8'>Most recent crop year ({int(latest_fnd['year'])}): "
            f"{latest_fnd['yield_bu_acre']:.1f} bu/acre on {latest_fnd['planted_acres']/1_000_000:.1f}M planted acres "
            f"({latest_fnd['production_bu']/1_000_000_000:.2f}B bushels produced)</div>",
            unsafe_allow_html=True
        )

st.caption(f"Data: FRED (IMF global prices), USDA FAS PSD Online, USDA NASS QuickStats | Built by Nicholas Becnel | {datetime.today().strftime('%d %b %Y')}")
