import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EM Equity Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color palette ──────────────────────────────────────────────────────────
NAVY      = "#0A1628"
NAVY_MID  = "#112240"
NAVY_CARD = "#1A2F4E"
ACCENT    = "#4FC3F7"
ACCENT2   = "#00E5FF"
GREEN     = "#00C853"
RED       = "#FF1744"
YELLOW    = "#FFD600"
TEXT      = "#E8F0FE"
MUTED     = "#8899AA"

CUSTOM_CSS = f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {NAVY};
      color: {TEXT};
  }}
  [data-testid="stSidebar"] {{
      background-color: {NAVY_MID};
  }}
  [data-testid="stMetric"] {{
      background-color: {NAVY_CARD};
      border: 1px solid #1E3A5F;
      border-radius: 10px;
      padding: 12px;
  }}
  .stTabs [data-baseweb="tab-list"] {{
      background-color: {NAVY_MID};
      border-radius: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{
      color: {MUTED};
  }}
  .stTabs [aria-selected="true"] {{
      color: {ACCENT} !important;
      border-bottom: 2px solid {ACCENT};
  }}
  .stDataFrame {{
      background-color: {NAVY_CARD};
  }}
  .stTextArea textarea {{
      background-color: {NAVY_CARD};
      color: {TEXT};
      border: 1px solid #1E3A5F;
  }}
  h1, h2, h3 {{
      color: {TEXT};
  }}
  .block-container {{
      padding-top: 1.5rem;
  }}
  [data-testid="stSelectbox"] > div > div {{
      background-color: {NAVY_CARD};
      color: {TEXT};
  }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────
TICKERS = {
    "MA":   "Mastercard",
    "V":    "Visa",
    "PYPL": "PayPal",
    "MELI": "MercadoLibre",
    "SE":   "Sea Limited",
    "GRAB": "Grab Holdings",
    "KO":   "Coca-Cola",
    "NESN.SW": "Nestlé",
    "ULVR.L":  "Unilever",
}

PLOTLY_THEME = dict(
    paper_bgcolor=NAVY,
    plot_bgcolor=NAVY_MID,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    margin=dict(l=40, r=20, t=50, b=40),
)

RISK_FREE_RATE = 0.045  # annualised

# ─── Data pipeline ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    close.columns = [c.upper() if "." not in c else c for c in close.columns]
    return close.dropna(how="all")


@st.cache_data(ttl=3600)
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker":       t,
                "Name":         TICKERS.get(t, t),
                "P/E":          info.get("trailingPE"),
                "Fwd P/E":      info.get("forwardPE"),
                "Rev Growth":   info.get("revenueGrowth"),
                "Net Margin":   info.get("profitMargins"),
                "Mkt Cap ($B)": round(info.get("marketCap", 0) / 1e9, 1),
                "Sector":       info.get("sector", "—"),
            })
        except Exception:
            rows.append({"Ticker": t, "Name": TICKERS.get(t, t)})
    return pd.DataFrame(rows)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns).cumprod() - 1


def rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    excess = returns.mean() * 252 - rf
    vol    = returns.std() * np.sqrt(252)
    return float(excess / vol) if vol != 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def build_risk_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        rows.append({
            "Ticker":       col,
            "Name":         TICKERS.get(col, col),
            "Ann. Return":  f"{s.mean() * 252:.1%}",
            "Ann. Vol":     f"{s.std() * np.sqrt(252):.1%}",
            "Sharpe":       f"{sharpe_ratio(s):.2f}",
            "VaR 95%":      f"{compute_var(s):.2%}",
            "Max Drawdown": f"{max_drawdown(s):.2%}",
        })
    return pd.DataFrame(rows)


# ─── Chart helpers ──────────────────────────────────────────────────────────
PALETTE = [ACCENT, GREEN, YELLOW, "#FF6D00", "#AA00FF", "#F50057",
           "#00BFA5", "#FFD740", "#64DD17"]


def fig_cum_returns(cum: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col] * 100,
            name=col, line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Cumulative Return (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_rolling_vol(vol: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(vol.columns):
        fig.add_trace(go.Scatter(
            x=vol.index, y=vol[col] * 100,
            name=col, line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Rolling 30-Day Annualised Volatility (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_corr_matrix(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, RED], [0.5, NAVY_MID], [1, ACCENT]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_THEME, title="Return Correlation Matrix")
    return fig


def fig_var_bar(returns: pd.DataFrame) -> go.Figure:
    tickers = returns.columns.tolist()
    vars_   = [compute_var(returns[t]) * 100 for t in tickers]
    colors  = [RED if v < -2 else YELLOW if v < -1.5 else GREEN for v in vars_]
    fig = go.Figure(go.Bar(
        x=tickers, y=vars_,
        marker_color=colors,
        text=[f"{v:.2f}%" for v in vars_],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Value at Risk — 95% Confidence (Daily %)",
        yaxis_ticksuffix="%",
    )
    return fig


def fig_fundamentals(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["P/E", "Net Margin", "Rev Growth"])
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["P/E Ratio", "Net Margin (%)", "Revenue Growth (%)"])

    fig.add_trace(go.Bar(x=d["Ticker"], y=d["P/E"],
                         marker_color=ACCENT, name="P/E"), row=1, col=1)
    fig.add_trace(go.Bar(x=d["Ticker"], y=d["Net Margin"] * 100,
                         marker_color=GREEN, name="Net Margin"), row=1, col=2)
    fig.add_trace(go.Bar(x=d["Ticker"], y=d["Rev Growth"] * 100,
                         marker_color=YELLOW, name="Rev Growth"), row=1, col=3)

    fig.update_layout(**PLOTLY_THEME, showlegend=False,
                      title="Fundamental Overlay")
    return fig


def fig_ma_price(prices: pd.DataFrame) -> go.Figure:
    col = "MA"
    s   = prices[col].dropna()
    ma50  = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=s.index, open=s, high=s, low=s, close=s,
        name="MA", increasing_line_color=GREEN,
        decreasing_line_color=RED,
    ))

    # Replace candlestick with a proper OHLC if we only have close
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Price",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=ma50.index, y=ma50, name="50-day MA",
                             line=dict(color=YELLOW, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=ma200.index, y=ma200, name="200-day MA",
                             line=dict(color=RED, width=1.5, dash="dash")))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Mastercard (MA) — Price History",
        yaxis_tickprefix="$",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_ma_returns_dist(returns: pd.DataFrame) -> go.Figure:
    s = returns["MA"].dropna() * 100
    var95 = compute_var(returns["MA"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=60, name="Daily Returns",
        marker_color=ACCENT, opacity=0.75,
    ))
    fig.add_vline(x=var95, line_color=RED, line_dash="dash",
                  annotation_text=f"VaR 95%: {var95:.2f}%",
                  annotation_font_color=RED)
    fig.update_layout(
        **PLOTLY_THEME,
        title="Mastercard — Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
    )
    return fig


# ─── Screener helpers ───────────────────────────────────────────────────────
def build_screener(returns: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        row = {
            "Ticker":    col,
            "Name":      TICKERS.get(col, col),
            "Ann. Ret":  s.mean() * 252,
            "Ann. Vol":  s.std() * np.sqrt(252),
            "Sharpe":    sharpe_ratio(s),
            "Max DD":    max_drawdown(s),
            "VaR 95%":   compute_var(s),
        }
        # merge fundamentals
        frow = fundamentals[fundamentals["Ticker"] == col]
        if not frow.empty:
            row["P/E"]        = frow["P/E"].values[0]
            row["Rev Growth"] = frow["Rev Growth"].values[0]
            row["Net Margin"] = frow["Net Margin"].values[0]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Ticker")
    return df


def style_screener(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    fmt = {
        "Ann. Ret":   "{:.1%}",
        "Ann. Vol":   "{:.1%}",
        "Sharpe":     "{:.2f}",
        "Max DD":     "{:.1%}",
        "VaR 95%":    "{:.2%}",
        "P/E":        "{:.1f}",
        "Rev Growth": "{:.1%}",
        "Net Margin": "{:.1%}",
    }
    existing_fmt = {k: v for k, v in fmt.items() if k in df.columns}

    def color_sharpe(v):
        try:
            return f"color: {GREEN}" if float(v) >= 1 else f"color: {RED}"
        except Exception:
            return ""

    styler = (
        df.style
          .format(existing_fmt, na_rep="—")
          .map(color_sharpe, subset=["Sharpe"])
          .set_properties(**{"background-color": NAVY_CARD, "color": TEXT,
                             "border": f"1px solid #1E3A5F"})
          .set_table_styles([
              {"selector": "thead th",
               "props": [("background-color", NAVY_MID), ("color", ACCENT),
                         ("font-weight", "bold"), ("border", f"1px solid #1E3A5F")]},
          ])
    )
    return styler


# ─── App layout ─────────────────────────────────────────────────────────────
def sidebar() -> tuple[str, list[str]]:
    st.sidebar.markdown(f"## 🌍 EM Equity Dashboard")
    st.sidebar.markdown("---")
    period = st.sidebar.selectbox(
        "Lookback Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )
    st.sidebar.markdown("### Universe")
    selected = []
    for ticker, name in TICKERS.items():
        if st.sidebar.checkbox(f"{ticker} — {name}", value=True):
            selected.append(ticker)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data via yfinance · Refreshed hourly")
    return period, selected


def header():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"<h1 style='color:{ACCENT};margin-bottom:0'>🌍 Emerging Markets Equity Dashboard</h1>",
                    unsafe_allow_html=True)
        st.caption("EM-exposed equities — returns, risk, and fundamentals at a glance")
    with c2:
        st.metric("As of", datetime.today().strftime("%d %b %Y"))


def kpi_row(returns: pd.DataFrame, prices: pd.DataFrame):
    cols = returns.columns.tolist()
    if not cols:
        return
    # Last 1-day return for each ticker
    metrics = st.columns(min(len(cols), 5))
    for i, col in enumerate(cols[:5]):
        last_ret = returns[col].iloc[-1]
        ytd_ret  = (prices[col].iloc[-1] / prices[col].iloc[0] - 1)
        metrics[i].metric(
            col,
            f"${prices[col].iloc[-1]:,.2f}",
            f"{last_ret:.2%}",
            delta_color="normal",
        )


# ─── Tab: Screener ──────────────────────────────────────────────────────────
def tab_screener(screener_df: pd.DataFrame):
    st.subheader("Stock Screener")
    c1, c2, c3 = st.columns(3)
    min_sharpe = c1.slider("Min Sharpe Ratio", -2.0, 3.0, 0.0, 0.1)
    max_dd     = c2.slider("Max Drawdown Floor (%)", -80, 0, -40, 5)
    min_ret    = c3.slider("Min Ann. Return (%)", -30, 100, -10, 5)

    filt = screener_df.copy()
    filt = filt[filt["Sharpe"] >= min_sharpe]
    filt = filt[filt["Max DD"] >= max_dd / 100]
    filt = filt[filt["Ann. Ret"] >= min_ret / 100]

    st.markdown(f"**{len(filt)} / {len(screener_df)} equities pass filters**")
    st.write(style_screener(filt).to_html(), unsafe_allow_html=True)

    # Scatter: Risk vs Return
    scatter_df = filt.reset_index()
    scatter_df["Sharpe_str"] = scatter_df["Sharpe"].round(2).astype(str)
    fig = px.scatter(
        scatter_df,
        x="Ann. Vol", y="Ann. Ret",
        color="Sharpe", size_max=18,
        text="Ticker",
        color_continuous_scale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
        labels={"Ann. Vol": "Annualised Volatility", "Ann. Ret": "Annualised Return"},
        title="Risk-Return Scatter",
    )
    fig.update_traces(textposition="top center", marker=dict(size=12, opacity=0.85))
    fig.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)


# ─── Tab: Returns ───────────────────────────────────────────────────────────
def tab_returns(prices: pd.DataFrame, returns: pd.DataFrame):
    st.subheader("Returns Analysis")
    cum = cumulative_returns(returns)
    st.plotly_chart(fig_cum_returns(cum), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        vol = rolling_volatility(returns)
        st.plotly_chart(fig_rolling_vol(vol), use_container_width=True)
    with col2:
        st.plotly_chart(fig_corr_matrix(returns), use_container_width=True)


# ─── Tab: Risk ──────────────────────────────────────────────────────────────
def tab_risk(returns: pd.DataFrame):
    st.subheader("Risk Metrics")
    risk_df = build_risk_table(returns)

    st.dataframe(
        risk_df.set_index("Ticker"),
        use_container_width=True,
        height=320,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_var_bar(returns), use_container_width=True)

    with col2:
        # Max drawdown bar
        tickers = returns.columns.tolist()
        mdd     = [max_drawdown(returns[t]) * 100 for t in tickers]
        fig = go.Figure(go.Bar(
            x=tickers, y=mdd,
            marker_color=[RED if v < -20 else YELLOW if v < -10 else GREEN for v in mdd],
            text=[f"{v:.1f}%" for v in mdd],
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_THEME, title="Maximum Drawdown (%)",
                          yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)


# ─── Tab: Fundamentals ──────────────────────────────────────────────────────
def tab_fundamentals(fund_df: pd.DataFrame):
    st.subheader("Fundamental Overlay")

    display = fund_df.copy()
    for col in ["Rev Growth", "Net Margin"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
    for col in ["P/E", "Fwd P/E"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—"
            )
    if "Mkt Cap ($B)" in display.columns:
        display["Mkt Cap ($B)"] = display["Mkt Cap ($B)"].apply(
            lambda x: f"${x:,.1f}B" if pd.notna(x) else "—"
        )

    st.dataframe(display.set_index("Ticker"), use_container_width=True, height=380)
    st.plotly_chart(fig_fundamentals(fund_df), use_container_width=True)


# ─── Tab: Mastercard Deep Dive ──────────────────────────────────────────────
def tab_mastercard(prices: pd.DataFrame, returns: pd.DataFrame, fund_df: pd.DataFrame):
    st.markdown(f"<h2 style='color:{ACCENT}'>Mastercard (MA) — Deep Dive</h2>",
                unsafe_allow_html=True)

    # KPIs
    ma_ret = returns["MA"].dropna()
    ma_price = prices["MA"].dropna()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current Price",  f"${ma_price.iloc[-1]:,.2f}")
    k2.metric("Period Return",  f"{(ma_price.iloc[-1]/ma_price.iloc[0]-1):.2%}")
    k3.metric("Sharpe Ratio",   f"{sharpe_ratio(ma_ret):.2f}")
    k4.metric("VaR 95% (day)",  f"{compute_var(ma_ret):.2%}")
    k5.metric("Max Drawdown",   f"{max_drawdown(ma_ret):.2%}")

    # Price chart
    st.plotly_chart(fig_ma_price(prices), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_ma_returns_dist(returns), use_container_width=True)
    with col2:
        # Fundamental table
        st.markdown(f"#### Fundamental Snapshot")
        frow = fund_df[fund_df["Ticker"] == "MA"].T
        if not frow.empty:
            frow.columns = ["Value"]
            frow = frow.drop(["Ticker"], errors="ignore")
            st.dataframe(frow, use_container_width=True)

    # Investment thesis
    st.markdown("#### Investment Thesis")
    st.markdown(
        f"<p style='color:{MUTED};font-size:0.85rem'>Document your thesis — saved in session state</p>",
        unsafe_allow_html=True,
    )
    thesis_key = "ma_thesis"
    if thesis_key not in st.session_state:
        st.session_state[thesis_key] = ""

    thesis = st.text_area(
        label="Your notes",
        value=st.session_state[thesis_key],
        height=200,
        placeholder=(
            "e.g. Mastercard benefits from secular growth in cashless payments across "
            "emerging markets — particularly South Asia and Sub-Saharan Africa where "
            "card penetration remains <20%. Network effect moat, asset-light model, "
            "~45% net margins. Key risks: CBDC displacement, regulatory caps on "
            "interchange fees (cf. EU IFR). Current valuation implies ~14% EPS CAGR..."
        ),
        label_visibility="collapsed",
    )
    st.session_state[thesis_key] = thesis

    if thesis.strip():
        wc = len(thesis.split())
        st.caption(f"{wc} words")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    period, selected = sidebar()
    header()

    if not selected:
        st.warning("Select at least one ticker in the sidebar.")
        return

    # ── Data loading ──
    with st.spinner("Fetching market data…"):
        prices = fetch_prices(selected, period=period)

    # Keep only selected tickers that are actually in the dataframe
    valid = [t for t in selected if t in prices.columns]
    if not valid:
        st.error("No valid price data returned. Check your ticker list.")
        return

    prices  = prices[valid]
    returns = compute_returns(prices)

    with st.spinner("Loading fundamentals…"):
        fund_df = fetch_fundamentals(valid)

    screener_df = build_screener(returns, fund_df)

    # ── KPI strip ──
    kpi_row(returns, prices)
    st.markdown("---")

    # ── Tabs ──
    tabs = st.tabs(["📋 Screener", "📈 Returns", "⚠️ Risk", "📊 Fundamentals", "🏦 MA Deep Dive"])

    with tabs[0]:
        tab_screener(screener_df)
    with tabs[1]:
        tab_returns(prices, returns)
    with tabs[2]:
        tab_risk(returns)
    with tabs[3]:
        tab_fundamentals(fund_df)
    with tabs[4]:
        if "MA" in valid:
            tab_mastercard(prices, returns, fund_df)
        else:
            st.info("Add MA (Mastercard) to the screener universe to see the deep dive.")


if __name__ == "__main__":
    main()
