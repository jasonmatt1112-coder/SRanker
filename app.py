"""
Stock Ranker v2: composite factor score dashboard (educational, not financial advice).

Score = wM*M + wT*T + wQ*Q + wG*G + wV*V - wRisk*z(vol60)
Projection = calibrated from this model's own history (score -> forward return), shrunk, with a
volatility-scaled range.  A background tracker logs the Top 15 every trading day and compares
predicted vs actual (open -> close) on the "Past History" tab.

Run:  pip install -r requirements.txt
      streamlit run app.py
Optional: set HISTORY_PATH (e.g. a Google Drive path) so history survives Colab restarts.
"""
import contextlib
import html
import inspect
import io
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Hyperdrift · Stock Ranker", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")
try:                                    # works on old and new Streamlit versions
    _p = inspect.signature(st.dataframe).parameters
    FULL = {"use_container_width": True} if ("use_container_width" in _p and getattr(
        _p.get("width"), "default", None) != "stretch") else {"width": "stretch"}
except Exception:
    FULL = {"use_container_width": True}

DEMO = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA LLY JPM V UNH XOM MA COST HD PG JNJ ABBV "
        "NFLX CRM BAC KO AMD ORCL CVX MRK PEP ADBE WMT TMO CSCO ACN MCD LIN ABT QCOM GE "
        "DIS TXN CAT INTU AMGN IBM PFE NOW ISRG GS").split()

ET = ZoneInfo("America/New_York")
WK = ["M", "T", "Q", "G", "V", "risk"]
FACT = ["M", "T", "Q", "G", "V"]
HORIZONS = {"Today (1 day)": 1, "1 week": 5, "1 month": 21, "1 year": 252}
HIST_PATH = os.environ.get("HISTORY_PATH", "stock_ranker_history.csv")
HIST_COLS = ["date", "ticker", "rank", "score", "pred_low", "pred_mid", "pred_high",
             "open", "close", "final", "logged_at"]
LOCK = threading.Lock()
# Expert benchmark: TipRanks' "Top 10 Wall Street research firms of the past decade" (CNBC, Apr 2023).
# Credit Suisse (bought by UBS) is replaced by Needham, which TipRanks ranks near the top of its
# S&P-benchmarked firm list. Value = (regex matched against yfinance's firm name, TipRanks success rate %).
EXPERTS = {
    "RBC Capital": (r"\brbc\b", 56), "Jefferies": (r"jefferies", 56), "Truist": (r"truist", 60),
    "UBS": (r"\bubs\b", 58), "KBW": (r"keefe|\bkbw\b", 65), "Raymond James": (r"raymond james", 55),
    "Stifel": (r"stifel", 54), "Wolfe Research": (r"wolfe", 61), "Deutsche Bank": (r"deutsche", 56),
    "Needham": (r"needham", None),
}
EXPERT_START = os.environ.get("EXPERT_START", "2026-10-20")
EXP_PATH = os.path.splitext(HIST_PATH)[0] + "_experts.csv"
EXP_COLS = ["date", "expert", "ticker", "rank", "rating_date", "open", "close", "final", "logged_at"]
# Live Trading model (day-trading paper account): $100 per stock per day, 10 stocks.
LIVE_N, COST_BPS, MAX_TRADES, CASH0 = 10, 5, 3, 100.0
LIVE_START = os.environ.get("LIVE_START", "2026-10-21")
LIVE_PATH = os.path.splitext(HIST_PATH)[0] + "_live.csv"
LIVE_TRADES_PATH = os.path.splitext(HIST_PATH)[0] + "_live_trades.csv"
BUY_RE = r"buy|outperform|overweight|positive|accumulate|top pick|\badd\b"
NOT_BUY_RE = r"sell|underperform|underweight|reduce|negative"
# Fixed model used by the daily tracker (price-based, liquid names) so history stays comparable.
TRACK_W = {"M": .5, "T": .5, "Q": 0., "G": 0., "V": 0., "risk": .20}
TRACK_MIN_PRICE, TRACK_MIN_DV, TRACK_N = 10.0, 100e6, 15

PRESETS = {
    "⚖️ Balanced": dict(w_M=.30, w_T=.15, w_Q=.20, w_G=.20, w_V=.15, w_risk=.10,
                        min_price=5.0, min_dv=20.0, min_cap=2.0, top_n=20, cap=8,
                        horizon="1 month"),
    "⚡ Quick gains (today)": dict(w_M=.35, w_T=.40, w_Q=.25, w_G=.0, w_V=.0, w_risk=.25,
                                  min_price=10.0, min_dv=100.0, min_cap=10.0, top_n=15, cap=8,
                                  horizon="Today (1 day)"),
    "🌱 Long-term (1 year)": dict(w_M=.15, w_T=.0, w_Q=.30, w_G=.25, w_V=.30, w_risk=.15,
                                  min_price=10.0, min_dv=20.0, min_cap=10.0, top_n=20, cap=8,
                                  horizon="1 year"),
}

def _dots(seed, n, tile, colors, rmin, rmax):
    """Tiny soft dots at random spots inside one tile (pure CSS, no JS)."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        x, y, r, c = rnd.randint(0, tile), rnd.randint(0, tile), rnd.uniform(rmin, rmax), rnd.choice(colors)
        out.append(f"radial-gradient(circle at {x}px {y}px,{c} 0,{c} {r:.2f}px,transparent {r + 1.1:.2f}px)")
    return ",".join(out)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Manrope:wght@400;500;600;700&display=swap');
:root{--bg:#050409;--ink:#e2dff0;--mute:#9591ae;--dim:#625e7b;--line:rgba(167,139,250,.11);
--v:#8b5cf6;--v2:#a78bfa;--v3:#c4b5fd;--g:#5fd3a8;--r:#f08aa0;--panel:rgba(16,12,28,.72);
--fd:'Sora','Manrope',system-ui,sans-serif;--fb:'Manrope',system-ui,-apple-system,sans-serif}
/* ---------- canvas: black, one faint violet glow, two layers of tiny drifting particles ---------- */
.stApp{isolation:isolate;color:var(--ink);font-family:var(--fb);background-color:var(--bg);background-attachment:fixed;
background-image:radial-gradient(900px 600px at 6% -8%,rgba(109,40,217,.17),transparent 62%),
radial-gradient(900px 640px at 100% 112%,rgba(88,28,135,.14),transparent 62%)}
.stApp::before,.stApp::after{content:"";position:fixed;left:0;top:0;width:100%;height:3000px;z-index:-1;
pointer-events:none;will-change:transform;contain:strict}
.stApp::before{background:__DOTS_A__;background-size:500px 500px;opacity:.55;animation:driftA 140s linear infinite}
.stApp::after{background:__DOTS_B__;background-size:750px 750px;animation:driftB 210s linear infinite,twinkle 9s ease-in-out infinite alternate}
@keyframes driftA{to{transform:translate3d(0,-1500px,0)}}
@keyframes driftB{to{transform:translate3d(0,-1500px,0)}}
@keyframes twinkle{from{opacity:.3}to{opacity:.8}}
@keyframes pulse{50%{opacity:.35;transform:scale(.75)}}
@media (prefers-reduced-motion:reduce){.stApp::before,.stApp::after{animation:none}}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stHeader"],.main{background:transparent!important}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"]{display:none!important}
footer,#MainMenu{visibility:hidden}
.block-container{padding-top:2.2rem!important;max-width:1240px}
[data-stale="true"]{opacity:1!important}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:var(--fd)!important;letter-spacing:-.02em;color:var(--ink)}
.stApp p,.stApp label,.stApp li,.stApp button,.stApp input,.stApp textarea,.stApp [data-baseweb]{font-family:var(--fb)!important}
.stApp,.stApp p,.stApp li,.stApp label{color:var(--ink)}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--mute)!important}
/* ---------- hero ---------- */
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin:0 0 16px}
.brand{font-family:var(--fd);font-weight:800;font-size:40px;line-height:1;letter-spacing:-.04em;
background:linear-gradient(100deg,#ede9fe 0%,#a78bfa 60%,#8b5cf6 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.brand em{font-style:normal;font-weight:600;font-size:14px;letter-spacing:.02em;margin-left:10px;color:var(--mute);-webkit-text-fill-color:var(--mute)}
.tag{color:var(--mute);font-size:13px;margin-top:8px;max-width:620px;line-height:1.5}
.pill{display:inline-flex;align-items:center;gap:8px;padding:5px 13px;border-radius:999px;font-size:12px;font-weight:700;
background:rgba(167,139,250,.06);border:1px solid var(--line);color:var(--mute)}
.pill i{width:7px;height:7px;border-radius:50%;background:var(--dim)}
.pill.on{color:#c4f1de;border-color:rgba(95,211,168,.3);background:rgba(95,211,168,.07)}
.pill.on i{background:var(--g);box-shadow:0 0 8px var(--g);animation:pulse 1.8s ease-in-out infinite}
/* ---------- nav: one clean scrolling row of pills (selectors are deliberately broad) ---------- */
[data-testid="stButtonGroup"]{display:flex!important;flex-wrap:nowrap!important;gap:6px!important;overflow-x:auto;
scrollbar-width:none;margin:2px 0 12px;padding-bottom:2px}
[data-testid="stButtonGroup"]::-webkit-scrollbar{display:none}
[data-testid="stButtonGroup"] button{flex:none;white-space:nowrap;border-radius:999px!important;border:1px solid var(--line)!important;
background:rgba(167,139,250,.05)!important;color:var(--mute)!important;padding:.4rem 1rem!important;font-weight:700!important;
min-height:0!important;box-shadow:none!important;transition:color .2s,border-color .2s,background .2s}
[data-testid="stButtonGroup"] button *{color:inherit!important}
[data-testid="stButtonGroup"] button:hover{color:var(--ink)!important;border-color:rgba(167,139,250,.45)!important}
[data-testid="stButtonGroup"] button:is([data-testid$="Active"],[aria-checked="true"],[aria-pressed="true"],[aria-selected="true"]){
color:#fff!important;border-color:transparent!important;background:linear-gradient(135deg,#7c3aed,#9333ea)!important;
box-shadow:0 6px 22px -10px rgba(124,58,237,.95)!important}
/* ---------- settings popover ---------- */
[data-testid="stPopover"]{margin:0 0 10px}
[data-testid="stPopover"] button{border-radius:999px!important;border:1px solid var(--line)!important;background:rgba(167,139,250,.07)!important;
color:var(--v3)!important;font-weight:700!important;padding:.35rem 1rem!important}
[data-testid="stPopover"] button:hover{border-color:rgba(167,139,250,.5)!important;color:#fff!important}
[data-testid="stPopoverBody"]{background:#0b0815!important;border:1px solid var(--line)!important;border-radius:18px!important;
min-width:min(400px,92vw)!important;max-height:78vh;overflow:auto}
[data-testid="stPopoverBody"] h2{font-size:11px!important;letter-spacing:.16em;text-transform:uppercase;color:var(--mute)!important;margin-top:.8rem}
/* ---------- widgets ---------- */
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 16px}
[data-testid="stMetricLabel"] p{color:var(--mute)!important;font-size:12px!important;font-weight:600}
[data-testid="stMetricValue"]{font-family:var(--fd)!important;font-weight:700;letter-spacing:-.02em}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:14px!important;background:rgba(14,10,24,.55)}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line)}
[data-testid="stAlert"]{border-radius:14px;background:rgba(139,92,246,.09)!important;border:1px solid var(--line)}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="base-input"]{background:rgba(167,139,250,.05)!important;
border-radius:12px!important;border-color:var(--line)!important}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;font-weight:700;
border:1px solid var(--line);background:rgba(167,139,250,.06);color:var(--ink);transition:border-color .2s,color .2s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(167,139,250,.55);color:#fff}
[data-testid="stFormSubmitButton"]>button{border:0;color:#fff;background:linear-gradient(135deg,#7c3aed,#9333ea)}
[data-testid="stForm"]{border:0!important;padding:0!important}
hr{border-color:var(--line)!important}
/* ---------- stock cards ---------- */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:8px}
.card{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:18px;color:var(--ink);
transition:transform .22s cubic-bezier(.2,.8,.2,1),border-color .22s}
.card:hover{transform:translateY(-3px);border-color:rgba(167,139,250,.42)}
.card .top{display:flex;gap:12px;align-items:center}
.logo{position:relative;width:44px;height:44px;border-radius:13px;overflow:hidden;background:#efeff5;display:flex;
align-items:center;justify-content:center;font-weight:800;font-size:15px;flex:none}
.logo img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;padding:6px;box-sizing:border-box}
.id{flex:1;min-width:0}
.tk{font-family:var(--fd);font-weight:800;font-size:17px;letter-spacing:-.02em}
.rk{font-size:10.5px;font-weight:800;background:rgba(139,92,246,.22);border-radius:999px;padding:2px 8px;margin-left:8px;color:var(--v3);vertical-align:2px}
.nm{font-size:12px;color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sc{font-size:11px;color:var(--dim)}
.px{text-align:right;font-weight:700;font-size:14px}
.px small{display:block;font-size:11px;font-weight:700}
.proj{font-family:var(--fd);font-size:24px;font-weight:700;margin:16px 0 2px;letter-spacing:-.03em}
.proj small{font-size:12px;color:var(--dim);font-weight:600;font-family:var(--fb)}
.sub{font-size:12px;color:var(--mute);margin-bottom:12px}
.fr{display:flex;align-items:center;gap:10px;margin:5px 0}
.fr em{font-style:normal;font-size:11px;color:var(--mute);width:62px}
.fr b{flex:1;height:5px;background:rgba(167,139,250,.09);border-radius:6px;position:relative;display:block;overflow:hidden}
.fr b::after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,.18)}
.fr i{position:absolute;top:0;bottom:0;border-radius:6px;display:block}
.foot{display:flex;justify-content:space-between;align-items:center;margin-top:14px;padding-top:12px;
border-top:1px solid var(--line);font-size:12px;color:var(--mute)}
.cf{padding:2px 10px;border-radius:999px;font-weight:700;font-size:11.5px}
.lst{width:100%;border-collapse:collapse;background:var(--panel);border-radius:16px;overflow:hidden;color:var(--ink);border:1px solid var(--line)}
.lst th{text-align:left;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);padding:12px;background:rgba(167,139,250,.05)}
.lst td{padding:10px 12px;border-top:1px solid var(--line);font-size:14px}.lst small{color:var(--mute)}
.lst tr:hover td{background:rgba(139,92,246,.07)}
.sig{display:inline-block;padding:4px 13px;border-radius:999px;font-weight:800;font-size:12.5px;letter-spacing:.06em}
.sig.BUY{background:rgba(95,211,168,.13);color:#8fe3c3}
.sig.HOLD{background:rgba(139,92,246,.18);color:var(--v3)}
.sig.SELL{background:rgba(240,138,160,.14);color:#f6aab9}
.sig.WAIT{background:rgba(167,139,250,.07);color:var(--mute)}
.cf.High{background:rgba(95,211,168,.13);color:#8fe3c3}.cf.Medium{background:rgba(167,139,250,.16);color:var(--v3)}
.cf.Low{background:rgba(240,138,160,.14);color:#f6aab9}
</style>
"""
CSS = _CSS.replace("__DOTS_A__", _dots(7, 24, 500, ["rgba(255,255,255,.7)", "rgba(196,181,253,.8)"], .45, .9)) \
          .replace("__DOTS_B__", _dots(21, 12, 750, ["rgba(167,139,250,.95)", "rgba(196,181,253,.85)"], .8, 1.4))


# ----------------------------------------------------------------- stats helpers
def zs(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score: winsorized at the 2nd/98th percentile, then clipped to +/-3."""
    s = s.astype(float)
    lo, hi = s.quantile(.02), s.quantile(.98)
    s = s.clip(lo, hi)
    sd = s.std()
    if not sd or np.isnan(sd):
        return s * np.nan
    return ((s - s.mean()) / sd).clip(-3, 3)


def zrows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional z-score (winsorized, clipped +/-3)."""
    df = df.clip(df.quantile(.02, axis=1), df.quantile(.98, axis=1), axis=0)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(df.mean(axis=1), axis=0).div(sd, axis=0).clip(-3, 3)


def zf(df, c, inv=False, sec=False):
    """z-score of column c; optionally blended 50/50 with a sector-relative z-score."""
    if c not in df or df[c].notna().sum() < 5:
        return None
    g = zs(df[c])
    if sec and df["sector"].notna().sum() >= 10:
        s = df.groupby("sector")[c].transform(lambda x: zs(x) if x.notna().sum() >= 6 else x * np.nan)
        g = 0.5 * s.fillna(g) + 0.5 * g
    return -g if inv else g


def wavg(parts):
    """Weighted average that skips missing metrics (weights renormalized)."""
    parts = [(s, w) for s, w in parts if s is not None]
    if not parts:
        return None
    num = sum(s.fillna(0) * w for s, w in parts)
    den = sum(s.notna().astype(float) * w for s, w in parts)
    return num / den.replace(0, np.nan)


def put(s):
    return np.nan if s is None else s


def comb_M(Z):
    return wavg([(Z.get("r12_1"), .35), (Z.get("r6"), .25), (Z.get("px_sma200"), .10),
                 (Z.get("px_sma50"), .10), (Z.get("ram"), .20)])


def comb_T(Z):
    return wavg([(Z.get("vol_surge"), .35), (Z.get("pct_high"), .25), (Z.get("range_exp"), .20),
                 (Z.get("news_count"), .20)])


def position_weights(vol: pd.Series, cap: float) -> pd.Series:
    """Inverse-volatility weights with a per-stock cap."""
    cap = max(cap, 1 / len(vol))
    w = 1 / vol
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


# ----------------------------------------------------------------- data (plain functions, thread-safe)
def _sp500():
    try:
        page = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text
        t = pd.read_html(io.StringIO(page))[0]
        t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
        return t.set_index("Symbol")["GICS Sector"].to_dict()
    except Exception:
        return {s: None for s in DEMO}


def _download(tickers, period="5y"):
    d = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
    o, c, v = d["Open"].copy(), d["Close"].copy(), d["Volume"].copy()
    for x in (o, c, v):
        x.index = pd.to_datetime(x.index)
        if x.index.tz is not None:
            x.index = x.index.tz_localize(None)
        x.index = x.index.normalize()
    c = c.dropna(axis=1, how="all")
    return c, v[c.columns], o[c.columns]


def _fundamentals(tickers, with_news):
    keys = ["longName", "sector", "marketCap", "returnOnEquity", "grossMargins", "profitMargins",
            "debtToEquity", "revenueGrowth", "earningsGrowth", "freeCashflow", "enterpriseValue",
            "ebitda", "trailingPE", "targetMeanPrice", "numberOfAnalystOpinions"]

    def one(t):
        out = {"ticker": t}
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            out.update({k: info.get(k) for k in keys})
            if with_news:
                out["news_count"] = len(tk.news or [])
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, tickers))
    df = pd.DataFrame(rows).set_index("ticker")
    for c in df.columns:
        if c not in ("longName", "sector"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fcf_yield"] = df.freeCashflow / df.marketCap
    df["ebitda_ev"] = df.ebitda / df.enterpriseValue
    df["pe"] = df.trailingPE.where(df.trailingPE > 0)
    return df


def _live(tickers):
    """Latest 1-minute prices and today's open for the given tickers (empty if market closed)."""
    try:
        d = yf.download(list(tickers), period="1d", interval="1m", progress=False,
                        threads=True, prepost=False)
        c, o = d["Close"], d["Open"]
        if isinstance(c, pd.Series):
            c, o = c.to_frame(tickers[0]), o.to_frame(tickers[0])
        return pd.DataFrame({"price": c.ffill().iloc[-1], "open": o.bfill().iloc[0]}).dropna()
    except Exception:
        return pd.DataFrame(columns=["price", "open"])


get_sp500 = st.cache_data(ttl=86400, show_spinner=False)(_sp500)
load_prices = st.cache_resource(ttl=900, show_spinner=False)(_download)
load_fundamentals = st.cache_resource(ttl=86400, show_spinner=False)(_fundamentals)
load_live = st.cache_data(ttl=45, show_spinner=False)(_live)


def build_features(close, volume):
    ret = close.pct_change()
    vol60 = ret.rolling(60).std() * np.sqrt(252)
    r12 = close.shift(21) / close.shift(252) - 1            # 12m return skipping last month
    F = {
        "r12_1": r12,
        "r6": close / close.shift(126) - 1,
        "px_sma200": close / close.rolling(200).mean() - 1,
        "px_sma50": close / close.rolling(50).mean() - 1,
        "ram": r12 / vol60.replace(0, np.nan),              # risk-adjusted momentum
        "vol_surge": volume.rolling(10).mean() / volume.rolling(60).mean().replace(0, np.nan),
        "pct_high": close / close.rolling(252).max(),
        "range_exp": ret.abs().rolling(5).mean() / ret.abs().rolling(60).mean().replace(0, np.nan),
        "vol60": vol60,
    }
    return F, (close * volume).rolling(60).mean()


# ----------------------------------------------------------------- scoring
def rank_universe(close, volume, w, min_price, min_dv, min_cap=0.0, fund=None, sectors=None):
    """Score every eligible stock using data up to the last row of `close`."""
    F, dv = build_features(close, volume)
    last = pd.DataFrame({k: v.iloc[-1] for k, v in F.items()})
    last["price"] = close.iloc[-1]
    last["dollar_vol"] = dv.iloc[-1]
    ok = (last.price > min_price) & (last.dollar_vol > min_dv) & \
        last[["r12_1", "r6", "vol60"]].notna().all(axis=1)
    use_f = fund is not None
    if use_f:
        last = last.join(fund)
        ok &= last.marketCap.fillna(0) > min_cap
        last["upside"] = (last.targetMeanPrice / last.price - 1) \
            .where(last.numberOfAnalystOpinions >= 3).clip(-.5, .5)
    if "sector" not in last:
        last["sector"] = None
    if sectors:
        last["sector"] = last.sector.fillna(pd.Series(sectors))
    df = last[ok].copy()
    if len(df) < 10:
        return F, dv, df

    df["M"] = put(comb_M({k: zf(df, k) for k in ["r12_1", "r6", "px_sma200", "px_sma50", "ram"]}))
    df["T"] = put(comb_T({k: zf(df, k) for k in ["vol_surge", "pct_high", "range_exp", "news_count"]}))
    if use_f:
        s = lambda c, inv=False: zf(df, c, inv, True)   # sector-relative fundamentals
        df["Q"] = put(wavg([(s("returnOnEquity"), 1), (s("grossMargins"), 1),
                            (s("profitMargins"), 1), (s("debtToEquity", True), 1)]))
        df["G"] = put(wavg([(s("revenueGrowth"), 1), (s("earningsGrowth"), .7), (s("upside"), 1)]))
        df["V"] = put(wavg([(s("fcf_yield"), 1), (s("ebitda_ev"), 1), (s("pe", True), 1)]))
    else:
        df["Q"] = df["G"] = df["V"] = np.nan
    act = [k for k in FACT if w[k] > 0 and (use_f or k in "MT")] or ["M"]
    df["miss"] = df[act].isna().sum(axis=1)
    df[FACT] = df[FACT].fillna(0)                      # missing factor = neutral
    df["risk_pen"] = zs(df.vol60)
    for k in FACT:
        df[f"c_{k}"] = w[k] * df[k]
    df["c_Risk"] = -w["risk"] * df.risk_pen
    df["Score"] = df[[f"c_{k}" for k in FACT] + ["c_Risk"]].sum(axis=1)
    c = df[[f"c_{k}" for k in act]]
    agree = np.sign(c).eq(np.sign(df.Score), axis=0).mean(axis=1)
    df["conf"] = (0.5 * (1 - df.miss / len(act)) + 0.5 * agree).fillna(.5)
    return F, dv, df


def _rank_cached(_close, _volume, _fund, _sectors, key, w, min_price, min_dv, min_cap):
    return rank_universe(_close, _volume, w, min_price, min_dv, min_cap, _fund, _sectors)


def hist_score(close, F, dv, w, min_price, min_dv):
    """Point-in-time, price-only score for every date (used for calibration and backtest)."""
    elig = (close > min_price) & (dv > min_dv)
    Z = {k: zrows(v.where(elig)) for k, v in F.items()}
    M, T = comb_M(Z), comb_T(Z)
    return w["M"] * M + w["T"] * T - w["risk"] * Z["vol60"]


def _calibrate(_close, _opn, _F, _dv, key, wt, h, min_price, min_dv):
    """Map score -> expected forward return using the model's own history (shrunk)."""
    close, opn, F, dv = _close, _opn, _F, _dv
    w = dict(zip(WK, wt))
    base = 0.08 * h / 252                                # long-run market drift assumption
    out = dict(base=base, beta=0.0, q_lo=-0.67, q_hi=0.67, ic=np.nan, ic_oos=np.nan,
               ic_t=np.nan, n=0)
    hs = hist_score(close, F, dv, w, min_price, min_dv)
    sd = hs.stack().std()
    if not sd or np.isnan(sd):
        return out
    fwd = (close.shift(-1) / opn.shift(-1) - 1) if h == 1 else (close.shift(-h) / close - 1)
    sig = F["vol60"] * np.sqrt(h / 252)
    d = pd.concat({"s": (hs / sd).stack(), "y": fwd.stack(), "sg": sig.stack()}, axis=1).dropna()
    d = d[d.sg > 0]
    if len(d) < 1000:
        return out
    y = d.y.clip(*d.y.quantile([.01, .99]))
    b, a = np.polyfit(d.s, y, 1)
    q_lo, q_hi = ((y - (a + b * d.s)) / d.sg).quantile([.25, .75])
    ic = hs.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1).dropna()
    n = len(ic)
    out.update(
        base=0.5 * y.mean() + 0.5 * base,                # blend sample mean with long-run drift
        beta=max(b, 0) * 0.6 if ic.mean() > 0 else 0.0,  # shrink slope; zero if no edge
        q_lo=q_lo, q_hi=q_hi, ic=ic.mean(), ic_oos=ic.iloc[int(n * .7):].mean(),
        ic_t=ic.mean() / ic.std() * np.sqrt(n / max(h, 1)) if ic.std() else np.nan, n=n)
    return out


calibrate = st.cache_resource(show_spinner=False, max_entries=20)(_calibrate)
rank_cached = st.cache_resource(show_spinner=False, max_entries=6)(_rank_cached)


def project(df, cal, h):
    d = df.copy()
    sd = d.Score.std()
    z = (d.Score / sd).clip(-3, 3) if sd and not np.isnan(sd) else d.Score * 0
    sig = d.vol60 * np.sqrt(h / 252)
    d["p_mid"] = cal["base"] + cal["beta"] * z
    d["p_lo"] = d.p_mid + cal["q_lo"] * sig
    d["p_hi"] = d.p_mid + cal["q_hi"] * sig
    return d


def _hs_cached(_close, _F, _dv, key, wt, min_price, min_dv):
    return hist_score(_close, _F, _dv, dict(zip(WK, wt)), min_price, min_dv)


hs_cached = st.cache_resource(show_spinner=False, max_entries=6)(_hs_cached)


def backtest(close, F, dv, w, top_n, min_price, min_dv, cost_bps, spy, score=None):
    if score is None:
        score = hist_score(close, F, dv, w, min_price, min_dv)
    dates = close.groupby(close.index.to_period("M")).tail(1).index
    dates = [d for d in dates if score.loc[d].notna().sum() >= top_n * 2]
    rows, prev = [], set()
    for d0, d1 in zip(dates[:-1], dates[1:]):
        sel = score.loc[d0].dropna().nlargest(top_n).index
        r = (close.loc[d1, sel] / close.loc[d0, sel] - 1).mean()
        turnover = 1 - len(prev & set(sel)) / top_n
        r -= turnover * 2 * cost_bps / 10000
        rows.append({"date": d1, "Strategy": r, "SPY": spy.loc[d1] / spy.loc[d0] - 1})
        prev = set(sel)
    return pd.DataFrame(rows).set_index("date")


def perf(r: pd.Series):
    eq = (1 + r).cumprod()
    return {"CAGR": (1 + r).prod() ** (12 / len(r)) - 1, "Volatility": r.std() * np.sqrt(12),
            "Sharpe (rf=0)": r.mean() / r.std() * np.sqrt(12),
            "Max drawdown": (eq / eq.cummax() - 1).min()}


# ----------------------------------------------------------------- history tracker
def _cfg(name):
    """Setting from the environment or Streamlit secrets (None if unset)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return str(st.secrets[name])
    except Exception:
        return None


def _csv(src):
    return pd.read_csv(src, dtype={"date": str})


_csv_cached = st.cache_data(ttl=120, show_spinner=False)(_csv)


def _mkt_open(now):
    return now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)


def _note(path, msg=None):
    try:
        d = st.session_state.setdefault("_load_err", {})
        if msg:
            d[os.path.basename(path)] = msg
        else:
            d.pop(os.path.basename(path), None)
    except Exception:
        pass


def _load(path):
    """Read a saved log from GitHub (when DATA_URL_BASE is set) AND from this machine / Drive, and use whichever has
    more rows. A missing or empty copy on one side can no longer hide the real history on the other."""
    base = _cfg("DATA_URL_BASE")        # e.g. https://raw.githubusercontent.com/USER/REPO/main/data/
    found, err = [], None
    for src, remote in (([(base.rstrip("/") + "/" + os.path.basename(path), True)] if base else [])
                        + [(path, False)]):
        try:
            if remote or os.path.exists(src):
                x = (_csv_cached if remote else _csv)(src)
                if "final" in x.columns:
                    x["final"] = x["final"].astype(str).eq("True")
                found.append(x)
        except Exception as e:
            if remote:
                err = f"{type(e).__name__}: {e}"
    if found:
        _note(path, err if err and not any(len(x) for x in found) else None)
        return max(found, key=len)          # ties keep the first (GitHub)
    _note(path, err)
    return None


def _save(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def storage_banner():
    """Shown on the history pages: warns when the saved history lives on Colab's disposable disk."""
    try:
        p = os.path.abspath(HIST_PATH)
        if os.path.exists("/content") and not p.startswith("/content/drive/") and not _cfg("DATA_URL_BASE"):
            st.warning(
                "**Your history is being saved to Colab's temporary disk** (`" + p + "`). Colab erases that disk every "
                "time the runtime restarts or disconnects (and after about 12 hours at most), so the history starts "
                "from zero each time. **Fix:** re-run **Cell 2** in the notebook, click *Allow* on the Google pop-up so "
                "it can connect your Drive, then re-run the app cell.")
    except Exception:
        pass


def read_hist():
    x = _load(HIST_PATH)
    return x if x is not None else pd.DataFrame(columns=HIST_COLS).astype({"final": bool})


def save_hist(h):
    _save(h, HIST_PATH)


def is_closed(d, now):
    return d < now.date() or (d == now.date() and now.time() >= dtime(16, 10))


_last_universe = {}


def read_exp():
    x = _load(EXP_PATH)
    return x if x is not None else pd.DataFrame(columns=EXP_COLS).astype({"final": bool})


def save_exp(e):
    _save(e, EXP_PATH)


def _ratings(tickers):
    """Every analyst rating event (firm, grade, date) yfinance has for these tickers."""
    def one(t):
        try:
            x = yf.Ticker(t).upgrades_downgrades
            if x is None or len(x) == 0:
                return None
            x = x.reset_index()
            x.columns = ["date"] + [str(c).lower() for c in x.columns[1:]]
            x["date"] = pd.to_datetime(x["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            x["ticker"] = t
            return x[["ticker", "date", "firm", "tograde"]]
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = [p for p in ex.map(one, tickers) if p is not None]
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "firm", "tograde"])
    return pd.concat(parts, ignore_index=True)


def expert_picks(r, ok_tickers, today, n=TRACK_N):
    """Each expert's picks: its most recent Buy-type calls (latest rating per stock), known BEFORE today."""
    day = pd.Timestamp(today)
    r = r[(r.date < day) & (r.date >= day - pd.Timedelta(days=365)) & r.ticker.isin(list(ok_tickers))]
    fl = r.firm.fillna("").str.lower()
    out = {}
    for name, (pat, _) in EXPERTS.items():
        m = r[fl.str.contains(pat, regex=True)].sort_values("date").groupby("ticker").tail(1)
        g = m.tograde.fillna("").str.lower()
        m = m[g.str.contains(BUY_RE, regex=True) & ~g.str.contains(NOT_BUY_RE, regex=True)]
        out[name] = m.sort_values(["date", "ticker"], ascending=[False, True]).head(n)
    return out


MAX_BACKFILL = 10


def _day_snapshot(c, v, o, tick, D):
    """Ranking for trading day D using ONLY data from before D (point-in-time)."""
    cols = [t for t in c.columns if t in tick]
    m = c.index < D
    if m.sum() < 300:
        return None
    cc, vv, oo = c.loc[m, cols], v.loc[m, cols], o.loc[m, cols]
    F, dv, df = rank_universe(cc, vv, TRACK_W, TRACK_MIN_PRICE, TRACK_MIN_DV)
    return (F, dv, df, cc, oo) if len(df) >= 30 else None


def tracker_step():
    """Resolve opens/closes of logged picks, then log any trading days not yet logged (ours + experts').
    Missed days (runner was off or late) are back-filled from the data, up to 10 trading days."""
    sec = _sp500()
    if len(sec) > 100 or not _last_universe:
        _last_universe.update(sec)
    tick = list(_last_universe)
    c, v, o = _download(tuple(sorted(set(tick + ["SPY"]))), "3y")
    now = datetime.now(ET)
    today = now.date()
    days = [d for d in c.index[c["SPY"].notna()][-MAX_BACKFILL:] if d.date() <= today] if "SPY" in c else []
    with LOCK:
        h0, e0 = read_hist(), read_exp()
    if len(h0):
        cand_h = [d for d in days if str(d.date()) > h0.date.max() and str(d.date()) not in set(h0.date)]
    else:
        cand_h = [d for d in days if str(d.date()) >= EXPERT_START]
    cand_e = [d for d in days if str(d.date()) >= EXPERT_START and str(d.date()) not in set(e0.date)]

    snaps = {D: _day_snapshot(c, v, o, tick, D) for D in sorted(set(cand_h) | set(cand_e))}
    ratings, ok_e = None, [d for d in cand_e if snaps.get(d)]
    if ok_e:
        ratings = _ratings(list(snaps[ok_e[-1]][2].index))      # slow (minutes): done outside the lock

    with LOCK:
        h, e = read_hist(), read_exp()
        for tbl in (h, e):                                                         # resolve earlier picks
            for i, r in tbl[~tbl.final].iterrows():
                d = pd.Timestamp(r.date)
                if d in c.index and r.ticker in c.columns:
                    tbl.loc[i, ["open", "close"]] = [o.at[d, r.ticker], c.at[d, r.ticker]]
                    tbl.loc[i, "final"] = is_closed(d.date(), now)
        stamp = now.isoformat(timespec="seconds")
        for D, snap in snaps.items():
            if snap is None:
                continue
            F, dv, df, cc, oo = snap
            ds, fin = str(D.date()), is_closed(D.date(), now)
            if D in cand_h and not (h.date == ds).any():
                cal = _calibrate(cc, oo, F, dv, "trk", tuple(TRACK_W[k] for k in WK), 1,
                                 TRACK_MIN_PRICE, TRACK_MIN_DV)
                top = project(df, cal, 1).nlargest(TRACK_N, "Score")
                new = pd.DataFrame({
                    "date": ds, "ticker": top.index, "rank": range(1, len(top) + 1),
                    "score": top.Score.values, "pred_low": top.p_lo.values, "pred_mid": top.p_mid.values,
                    "pred_high": top.p_hi.values, "open": o.loc[D, top.index].values,
                    "close": c.loc[D, top.index].values, "final": fin, "logged_at": stamp})
                h = pd.concat([h, new.dropna(subset=["open"])], ignore_index=True)
            if D in cand_e and ratings is not None and not (e.date == ds).any():
                rows = []
                for name, m in expert_picks(ratings, df.index, D.date()).items():
                    print(f"{ds} expert {name}: {len(m)} picks")
                    for k, (t, rd) in enumerate(zip(m.ticker, m.date), 1):
                        rows.append({"date": ds, "expert": name, "ticker": t, "rank": k,
                                     "rating_date": str(rd.date()), "open": o.at[D, t],
                                     "close": c.at[D, t], "final": fin, "logged_at": stamp})
                if rows:
                    e = pd.concat([e, pd.DataFrame(rows).dropna(subset=["open"])], ignore_index=True)
        save_hist(h)
        save_exp(e)


def compare_data(hist, exp, metric, running):
    """Per-day accuracy for ours, each expert, and the experts' average (only from EXPERT_START on)."""
    h, e = hist[hist.date >= EXPERT_START].copy(), exp[exp.date >= EXPERT_START].copy()
    h["who"] = "Ours"
    e = e.rename(columns={"expert": "who"})
    both = pd.concat([h[["date", "who", "open", "close"]], e[["date", "who", "open", "close"]]])
    both["ret"] = both.close / both.open - 1
    both = both.dropna(subset=["ret"])
    if both.empty:
        return pd.DataFrame()
    f = (lambda s: (s > 0).mean() * 100) if metric == "hit" else (lambda s: s.mean() * 100)
    daily = both.groupby(["date", "who"])["ret"].apply(f).rename("v").reset_index()
    pv = daily.pivot(index="date", columns="who", values="v").sort_index()
    names = [c for c in pv.columns if c != "Ours"]
    pv["Top 10 experts (average)"] = pv[names].mean(axis=1) if names else np.nan
    return pv.expanding().mean().where(pv.notna()) if running else pv


def render_comparison(hist, exp, live_px):
    st.markdown("---")
    st.subheader("Us vs. the Top 10 experts")
    st.caption(
        "Experts are the 10 research firms TipRanks ranked best over the past decade (Credit Suisse, now part of UBS, "
        "is replaced by Needham). A firm's picks each day are its 15 most recent Buy-type ratings (from Wall Street "
        f"rating data in yfinance, known before the open), scored open to close just like ours. Comparison starts {EXPERT_START}.")
    c1, c2 = st.columns(2)
    mlabel = c1.radio("Accuracy measure", ["Hit rate (% of picks that rose)", "Average return (open to close)"])
    metric = "hit" if mlabel.startswith("Hit") else "ret"
    running = c2.radio("Line shows", ["Running average", "Each day"]) == "Running average"
    ytitle = "% of picks up" if metric == "hit" else "Avg move, %"
    exp = exp.copy()
    pend = ~exp.final
    exp.loc[pend, "close"] = exp.loc[pend, "ticker"].map(live_px).fillna(exp.loc[pend, "close"])
    pv = compare_data(hist, exp, metric, running)
    if pv.empty:
        st.info(f"No comparison data yet. The first day is {EXPERT_START}. Once the tracker has logged ours and each "
                "expert's picks and the market has closed, the lines appear here.")
        return
    names = [c for c in pv.columns if c not in ("Ours", "Top 10 experts (average)") and pv[c].notna().any()]
    green, blue = "#c4b5fd", "#7c3aed"

    st.markdown("#### Ours vs. Top 10 experts")
    view = st.radio("Show", ["Both", "Ours only", "Experts only"], horizontal=True, key="v2")
    dom = {"Both": ["Ours", "Top 10 experts (average)"], "Ours only": ["Ours"],
           "Experts only": ["Top 10 experts (average)"]}[view]
    long = pv[[d for d in dom if d in pv]].reset_index().melt("date", var_name="Series", value_name="Value").dropna()
    sc = ["Ours", "Top 10 experts (average)"]
    st.altair_chart(alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("date:T", title="Trading day"), y=alt.Y("Value:Q", title=ytitle),
        color=alt.Color("Series:N", scale=alt.Scale(domain=sc, range=[green, blue]),
                        legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.Opacity("Series:N", scale=alt.Scale(domain=sc, range=[1, .35]), legend=None),
        tooltip=["date:T", "Series", alt.Tooltip("Value:Q", format=".1f")]).properties(height=300),
        **FULL)

    st.markdown("#### Every expert")
    if not names:
        st.info("No expert has logged picks yet (or yfinance had no Buy-rated calls from these firms).")
    else:
        k1, k2 = st.columns([4, 1])
        picked = k1.multiselect("Experts shown (click to toggle)", names, default=names)
        show_ours = k2.checkbox("Include ours", True)
        parts = []
        if picked:
            parts.append(alt.Chart(pv[picked].reset_index().melt("date", var_name="Expert", value_name="Value")
                                   .dropna()).mark_line(point=True, strokeWidth=2, opacity=.5).encode(
                x=alt.X("date:T", title=None), y=alt.Y("Value:Q", title=ytitle),
                color=alt.Color("Expert:N", legend=alt.Legend(orient="right", title=None)),
                tooltip=["date:T", "Expert", alt.Tooltip("Value:Q", format=".1f")]))
        if show_ours and "Ours" in pv:
            parts.append(alt.Chart(pv[["Ours"]].rename(columns={"Ours": "Value"}).reset_index().dropna()).mark_line(
                point=True, strokeWidth=4, color=green).encode(x="date:T", y=alt.Y("Value:Q", title=ytitle)))
        if parts:
            st.altair_chart(alt.layer(*parts).properties(height=230), **FULL)
        hit, ret = compare_data(hist, exp, "hit", False), compare_data(hist, exp, "ret", False)
        tab = pd.DataFrame({
            "Days tracked": hit.notna().sum(), "Avg hit rate %": hit.mean().round(1),
            "Avg return %": ret.mean().round(2)}).drop(index=["Top 10 experts (average)"], errors="ignore")
        tab["TipRanks success rate % (decade)"] = [EXPERTS.get(i, (None, None))[1] for i in tab.index]
        st.dataframe(tab.sort_values("Avg hit rate %", ascending=False), **FULL)
        last = exp[exp.date == exp.date.max()]
        with st.expander("Today's picks by expert"):
            for nm_, g in last.sort_values("rank").groupby("expert"):
                st.write(f"**{nm_}**: " + ", ".join(g.ticker))


@st.cache_resource
def start_tracker():
    if _cfg("DISABLE_APP_TRACKER"):      # a separate headless tracker.py process owns the logging
        return None

    def loop():
        while True:
            all_steps()
            time.sleep(5 * 60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ----------------------------------------------------------------- presentation helpers
def proj_color(x, full):
    """Soft grey at 0 -> mint (positive) / rose (negative) at |x| >= full."""
    t = min(abs(x) / full, 1.0) ** 0.75
    base = (95, 211, 168) if x >= 0 else (240, 138, 160)
    r, g, b = [round(n + (c - n) * t) for n, c in zip((205, 200, 228), base)]
    return f"rgb({r},{g},{b})"


FN = [("M", "Momentum"), ("T", "Trending"), ("Q", "Quality"), ("G", "Growth"), ("V", "Value")]


def card_html(rank, t, r, nm, h, hlabel, full, dec):
    f = lambda x: f"{x * 100:+.{dec}f}%"
    span = lambda x: f'<span style="color:{proj_color(x, full)}">{f(x)}</span>'
    hue = sum(map(ord, t)) * 37 % 360
    logo = (f'<div class="logo" style="color:hsl({hue},50%,35%)">{html.escape(t[:2])}'
            f'<img src="https://financialmodelingprep.com/image-stock/{html.escape(t)}.png" alt="" loading="lazy"></div>')

    def bar(k, n):
        v = float(np.clip(r[k], -3, 3))
        col = "linear-gradient(90deg,#7c3aed,#c4b5fd)" if v >= 0 else "linear-gradient(90deg,#9f4a68,#f08aa0)"
        wd, left = abs(v) / 3 * 50, (50 if v >= 0 else 50 - abs(v) / 3 * 50)
        return (f'<div class="fr"><em>{n}</em><b><i style="left:{left:.0f}%;width:{wd:.0f}%;background:{col}"></i></b></div>')

    facs = "".join(bar(k, n) for k, n in FN if k in "MT" or r["Q"] != 0 or r["G"] != 0 or r["V"] != 0)
    lab = "High" if r.conf >= .75 else "Medium" if r.conf >= .55 else "Low"
    live = ""
    if pd.notna(r.get("live", np.nan)):
        live = f'<small style="color:{proj_color(r.live, full)}">today {r.live * 100:+.2f}%</small>'
    nm = html.escape(nm if isinstance(nm, str) else "")[:34]
    sector = html.escape(str(r.sector)) if isinstance(r.sector, str) else ""
    return (f'<div class="card"><div class="top">{logo}<div class="id"><div class="tk">{html.escape(t)}'
            f'<span class="rk">#{rank}</span></div><div class="nm">{nm}</div><div class="sc">{sector}</div></div>'
            f'<div class="px">${r.price:,.2f}{live}</div></div>'
            f'<div class="proj">{span(r.p_lo)} <small>to</small> {span(r.p_hi)}</div>'
            f'<div class="sub">likely range · midpoint {span(r.p_mid)} · {hlabel}</div>'
            f'{facs}<div class="foot"><span>Score {r.Score:+.2f}</span>'
            f'<span class="cf {lab}">{lab} confidence</span></div></div>')


# ----------------------------------------------------------------- live trading model
def _split(d, tickers):
    out = {}
    if isinstance(d.columns, pd.MultiIndex):
        lv0 = set(d.columns.get_level_values(0))
        for t in tickers:
            try:
                out[t] = (d[t] if t in lv0 else d.xs(t, axis=1, level=1)).dropna(how="all")
            except KeyError:
                pass
    elif len(tickers) == 1:
        out[tickers[0]] = d.dropna(how="all")
    return out


def _download_ohlcv(tickers, period="1y"):
    d = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)
    out = {}
    for k, f in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("v", "Volume")):
        x = d[f].copy()
        x.index = pd.to_datetime(x.index)
        if x.index.tz is not None:
            x.index = x.index.tz_localize(None)
        x.index = x.index.normalize()
        out[k] = x
    cols = out["c"].dropna(axis=1, how="all").columns
    return {k: v[cols] for k, v in out.items()}


def _intraday(tickers, period="1d"):
    """1-minute bars (regular hours) per ticker, index = New York clock time."""
    try:
        d = yf.download(list(tickers), period=period, interval="1m", group_by="ticker",
                        progress=False, threads=True, prepost=False, auto_adjust=True)
    except Exception:
        return {}
    out = {}
    for t, x in _split(d, list(tickers)).items():
        try:
            x = x[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            idx = pd.to_datetime(x.index)
            if idx.tz is not None:
                idx = idx.tz_convert(ET).tz_localize(None)
            x.index = idx
            out[t] = x
        except Exception:
            pass
    return out


def _news_count(tickers):
    out = {}
    for t in tickers:
        try:
            out[t] = len(yf.Ticker(t).news or [])
        except Exception:
            out[t] = None
    return out


load_ohlcv = st.cache_resource(ttl=1800, show_spinner=False)(_download_ohlcv)
load_intraday = st.cache_resource(ttl=30, show_spinner=False)(_intraday)
load_news = st.cache_data(ttl=600, show_spinner=False)(_news_count)


def day_slice(df, D):
    if df is None or len(df) == 0:
        return None
    x = df[df.index.normalize() == pd.Timestamp(D).normalize()]
    return x if len(x) else None


def daily_stats(ox, D):
    """Per-stock day-trading stats using ONLY daily bars before day D."""
    m = ox["c"].index < D
    c, o, h, l, v = (ox[k][m] for k in "cohlv")
    if len(c) < 30:
        return pd.DataFrame()
    return pd.DataFrame({
        "adr": ((h - l) / c).iloc[-20:].mean(),                                    # typical daily range
        "eff": ((c - o).abs() / (h - l).replace(0, np.nan)).iloc[-20:].mean(),     # how directional days are
        "rvol": (v.iloc[-1] / v.iloc[-21:-1].mean().replace(0, np.nan)).clip(upper=5),
        "dvol": (c * v).iloc[-20:].mean(), "price": c.iloc[-1]})


def day_watchlist(ox, D, n=LIVE_N):
    """The 10 most tradeable stocks for day D (liquid, wide-ranging, directional, in the news flow)."""
    s_ = daily_stats(ox, D)
    if s_.empty:
        return s_
    s_ = s_.drop(index=["SPY"], errors="ignore")
    s_ = s_[(s_.price > 10) & (s_.dvol > 300e6) & (s_.adr > 0.01)].dropna()
    if len(s_) <= n:
        return s_.assign(Score=0.0)
    z = lambda x: zs(x).fillna(0)
    s_["Score"] = .45 * z(s_.adr) + .25 * z(s_.rvol) + .20 * z(s_.eff) + .10 * z(np.log(s_.dvol))
    return s_.nlargest(n, "Score")


def run_day(bars, spy, adr, final=False, cost_bps=COST_BPS, cash0=CASH0):
    """Minute-by-minute BUY/HOLD/SELL rules for one stock and one day, with a $100 paper account.
    Every indicator uses only data up to the current minute; a signal at minute t trades at minute t+1's
    open (plus a cost per side). final=True means the day is over: leftovers are closed at the last price."""
    b = bars.dropna(subset=["Open", "High", "Low", "Close"])
    if len(b) < 20:
        return {"ok": False}
    o, h, l, c, v = (b[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    idx, n, k = b.index, len(b), cost_bps / 1e4
    tm = np.array([t.hour * 60 + t.minute for t in idx])
    cv = np.cumsum(v)
    vwap = np.where(cv > 0, np.cumsum((h + l + c) / 3 * v) / np.where(cv > 0, cv, 1), c)
    in_or = (tm >= 570) & (tm < 585)                                    # opening range 9:30-9:44
    or_hi = h[in_or].max() if in_or.sum() >= 10 else np.nan
    or_lo = l[in_or].min() if in_or.sum() >= 10 else np.nan
    cs, vs = pd.Series(c, index=idx), pd.Series(v, index=idx)
    mom5, mom15 = (cs / cs.shift(5) - 1).to_numpy(), (cs / cs.shift(15) - 1).to_numpy()
    vr = (vs.rolling(5).mean() / vs.expanding().mean().replace(0, np.nan)).to_numpy()
    if spy is not None and len(spy):
        sp = spy.reindex(idx.union(spy.index)).ffill().reindex(idx)
        spm = np.nan_to_num((sp / sp.shift(15) - 1).to_numpy(), nan=0.0)
    else:
        spm = np.zeros(n)
    with np.errstate(invalid="ignore"):
        conds = [("Price above the opening-range high", c > or_hi), ("Price above VWAP", c > vwap),
                 ("Momentum up over 5 and 15 min", (mom5 > 0) & (mom15 > 0)),
                 ("Volume picking up", vr > 1.1), ("Market (SPY) not falling", spm > -0.0005)]
        flip = (c < vwap) & (mom5 < 0)
    score = sum(a.astype(int) for _, a in conds)
    ent = (score >= 4) & (tm >= 585) & (tm <= 930)                      # 9:45 - 15:30
    sh1 = lambda a: np.r_[False, a[:-1]]
    ent_conf, flip_conf = ent & sh1(ent), flip & sh1(flip)             # must hold 2 minutes in a row
    stop = float(np.clip(0.25 * adr, 0.003, 0.012)) if adr and not np.isnan(adr) else 0.006

    cash, sh, entry, peak, entry_i, cool, nbuy = cash0, 0.0, 0.0, 0.0, -99, -1, 0
    pending, events, eq, state = None, [], np.zeros(n), "WAIT"
    for i in range(n):
        if pending:
            if pending[0] == "BUY":
                fill = o[i] * (1 + k)
                sh, cash, entry, peak, entry_i, nbuy = cash / fill, 0.0, fill, fill, i, nbuy + 1
            else:
                fill = o[i] * (1 - k)
                cash, sh, cool = sh * fill, 0.0, i + 10
            events.append({"time": idx[i], "action": pending[0], "price": fill, "reason": pending[1]})
            pending = None
        eq[i] = cash + sh * c[i]
        state = "WAIT"
        if sh > 0:
            peak = max(peak, c[i])
            r = None
            if tm[i] >= 955:
                r = "End of day: flat before the close"
            elif c[i] <= entry * (1 - stop):
                r = "Stop-loss hit"
            elif peak >= entry * (1 + stop) and c[i] <= peak * (1 - stop):
                r = "Trailing stop hit (gains locked in)"
            elif flip_conf[i] and i - entry_i >= 5:
                r = "Trend broke: below VWAP with falling momentum"
            state = "SELL" if r else "HOLD"
            if r:
                pending = ("SELL", r)
        elif nbuy < MAX_TRADES and i > cool and ent_conf[i]:
            pending, state = ("BUY", "Breakout confirmed above the opening range, VWAP and momentum agree"), "BUY"
    if final:
        if pending and pending[0] == "SELL" and sh > 0:
            cash, sh = sh * c[-1] * (1 - k), 0.0
            events.append({"time": idx[-1], "action": "SELL", "price": c[-1] * (1 - k), "reason": pending[1]})
        elif sh > 0:
            cash, sh = sh * c[-1] * (1 - k), 0.0
            events.append({"time": idx[-1], "action": "SELL", "price": c[-1] * (1 - k), "reason": "Market closed"})
        pending = None
        eq[-1] = cash + sh * c[-1]
        state = "WAIT" if sh == 0 else state
    return {"ok": True, "bars": b, "vwap": vwap, "or_hi": or_hi, "or_lo": or_lo, "events": events,
            "eq": pd.Series(eq, index=idx), "end": float(eq[-1]), "state": state, "in_pos": sh > 0,
            "entry": entry if sh > 0 else None, "stop": stop, "stop_px": entry * (1 - stop) if sh > 0 else None,
            "trail_px": peak * (1 - stop) if sh > 0 and peak >= entry * (1 + stop) else None,
            "nbuy": nbuy, "conds": [(lab, bool(a[-1])) for lab, a in conds], "score": int(score[-1]),
            "price": float(c[-1]), "last": idx[-1], "day_open": float(o[0])}


def live_replay_step():
    """After the close: replay each finished day's minute bars through the rules and log the $100 results."""
    now = datetime.now(ET)
    sec = _sp500()
    tick = list(sec)
    ox = _download_ohlcv(tuple(sorted(set(tick + ["SPY"]))), "1y")
    c = ox["c"]
    with LOCK:
        d0 = _load(LIVE_PATH)
    done = set(d0.date) if d0 is not None and len(d0) else set()
    days = [d for d in c.index[c["SPY"].notna()][-6:] if str(d.date()) >= LIVE_START
            and str(d.date()) not in done and is_closed(d.date(), now)]
    if not days:
        return
    wl = {D: day_watchlist(ox, D) for D in days}
    need = sorted(set().union(*[set(w.index) for w in wl.values()]) | {"SPY"})
    bars = _intraday(need, "7d")
    rows, trades = [], []
    for D in days:
        ds, spy_b, day_rows = str(D.date()), day_slice(bars.get("SPY"), D), []
        for rank, (t, r) in enumerate(wl[D].iterrows(), 1):
            b = day_slice(bars.get(t), D)
            if b is None or len(b) < 150:
                continue
            res = run_day(b, spy_b["Close"] if spy_b is not None else None, r.adr, final=True)
            if not res["ok"]:
                continue
            day_rows.append({"date": ds, "ticker": t, "rank": rank, "adr": round(float(r.adr), 4),
                             "start": CASH0, "end": round(res["end"], 4), "trades": res["nbuy"],
                             "stock_open_to_close": round(float(b.Close.iloc[-1] / b.Open.iloc[0] - 1), 5)})
            trades += [{"date": ds, "ticker": t, "time": str(e["time"].time()), "action": e["action"],
                        "price": round(float(e["price"]), 4), "reason": e["reason"]} for e in res["events"]]
        if len(day_rows) >= 5:                      # only record days with (nearly) complete data
            rows += day_rows
    if rows:
        with LOCK:
            d1, t1 = _load(LIVE_PATH), _load(LIVE_TRADES_PATH)
            _save(pd.concat([d1, pd.DataFrame(rows)], ignore_index=True) if d1 is not None else pd.DataFrame(rows), LIVE_PATH)
            _save(pd.concat([t1, pd.DataFrame(trades)], ignore_index=True) if t1 is not None else pd.DataFrame(trades),
                  LIVE_TRADES_PATH)


def all_steps():
    """Everything the background job does. Returns the number of steps that failed."""
    errs = 0
    for fn in (tracker_step, live_replay_step):
        try:
            fn()
        except Exception as e:
            errs += 1
            print(f"{fn.__name__} error: {e!r}", flush=True)
    return errs


def live_card(rank, t, r, prev, nn, sector):
    logo = (f'<div class="logo" style="color:hsl({sum(map(ord, t)) * 37 % 360},50%,35%)">{html.escape(t[:2])}'
            f'<img src="https://financialmodelingprep.com/image-stock/{html.escape(t)}.png" alt="" loading="lazy"></div>')
    money = lambda x: "n/a" if x is None else f'<b style="color:{proj_color(x / CASH0 - 1, 0.02)}">${x:,.2f}</b>'
    head = (f'<div class="top">{logo}<div class="id"><div class="tk">{html.escape(t)}<span class="rk">#{rank}</span></div>'
            f'<div class="sc">{html.escape(sector) if isinstance(sector, str) else ""}</div></div>')
    if not r or not r.get("ok"):
        return (f'<div class="card">{head}<div class="px"><span class="sig WAIT">WAIT</span></div></div>'
                f'<div class="sub">Waiting for the market to open (signals start at 9:45 AM ET)</div>'
                f'<div class="foot"><span>Yesterday: {money(prev)}</span></div></div>')
    chg = r["price"] / r["day_open"] - 1
    extra = ""
    if r["in_pos"]:
        extra = f' · stop ${r["stop_px"]:,.2f}' + (f' · trail ${r["trail_px"]:,.2f}' if r["trail_px"] else "")
    news = f' · 📰 {nn}' if nn else ""
    return (f'<div class="card">{head}<div class="px"><span class="sig {r["state"]}">{r["state"]}</span></div></div>'
            f'<div class="proj">{money(r["end"])}</div>'
            f'<div class="sub">paper account today (started at $100){extra}</div>'
            f'<div class="fr"><em>Price</em><span style="font-size:13px">${r["price"]:,.2f} '
            f'<span style="color:{proj_color(chg, 0.02)}">{chg * 100:+.2f}% today</span></span></div>'
            f'<div class="foot"><span>Yesterday finished: {money(prev)}</span>'
            f'<span>{r["nbuy"]} trade{"s" if r["nbuy"] != 1 else ""}{news}</span></div></div>')


def render_live():
    now = datetime.now(ET)
    market_open = _mkt_open(now)
    st.subheader("Live Trading")
    st.caption("Educational paper trading, not advice. Each of today's 10 stocks gets its own $100, resets every day. "
               "Signals trade at the next minute's price with a small cost per trade. Studies find most real day "
               "traders lose money, so judge this by the results below, not by how good it sounds.")
    sec = get_sp500()
    uni = list(sec) if len(sec) > 100 else DEMO
    with (contextlib.nullcontext() if st.session_state.get("_live_ok") else
          st.spinner("Loading live trading data (the first time takes about a minute)...")):
        ox = load_ohlcv(tuple(sorted(set(uni + ["SPY"]))))
        days = ox["c"].index[ox["c"]["SPY"].notna()]
        today = pd.Timestamp(now.date())
        D = today if now.weekday() < 5 else days[-1]
        for _ in range(2):
            wl = day_watchlist(ox, D)
            if wl.empty:
                st.warning("Not enough data to build today's list.")
                return
            tick = list(wl.index)
            bars = load_intraday(tuple(tick + ["SPY"]), "7d")
            spy_b = day_slice(bars.get("SPY"), D)
            if spy_b is None and D == today and now.time() >= dtime(9, 35) and (days < today).any():
                D = days[days < today][-1]            # market holiday: show the last trading day instead
                continue
            break
    st.session_state["_live_ok"] = True
    prevD = days[days < D][-1]
    ps = daily_stats(ox, prevD)
    spy_prev = day_slice(bars.get("SPY"), prevD)
    final_day = D < today or now.time() >= dtime(16, 0)
    res, prev = {}, {}
    pcache = st.session_state.setdefault("_prev_cache", {})   # yesterday never changes
    for t in tick:
        b = day_slice(bars.get(t), D)
        res[t] = run_day(b, spy_b["Close"] if spy_b is not None else None, wl.at[t, "adr"], final=final_day) \
            if b is not None else None
        ck = (t, str(prevD.date()))
        if ck not in pcache:
            pb = day_slice(bars.get(t), prevD)
            pr = run_day(pb, spy_prev["Close"] if spy_prev is not None else None,
                         ps["adr"].get(t) if len(ps) else np.nan, final=True) if pb is not None else None
            pcache[ck] = pr["end"] if pr and pr.get("ok") else None
        prev[t] = pcache[ck]
    live = {t: r for t, r in res.items() if r and r.get("ok")}
    news = load_news(tuple(tick))

    status = "🟢 Market open" if market_open else ("🌅 Pre-market" if now.weekday() < 5 and now.time() < dtime(9, 30)
                                                    else "⚪ Market closed")
    cap = f"{status} · list for {D.date()} · refreshed {now.strftime('%I:%M:%S %p')} ET"
    if live:
        last = max(r["last"] for r in live.values())
        lag = (now.replace(tzinfo=None) - last.to_pydatetime()).total_seconds() / 60 - 1
        cap += f" · latest price bar {last.strftime('%I:%M %p')}"
        if market_open and lag > 5:
            cap += f" ⚠️ data is about {lag:.0f} min behind"
    st.caption(cap)
    if live:
        tot = sum(r["end"] for r in live.values()) / (CASH0 * len(live)) - 1
        spy_ret = spy_b.Close.iloc[-1] / spy_b.Open.iloc[0] - 1 if spy_b is not None else np.nan
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("All 10 stocks today", f"{tot * 100:+.2f}%", help="Average result of the 10 separate $100 paper accounts")
        m2.metric("SPY today (just holding)", "n/a" if np.isnan(spy_ret) else f"{spy_ret * 100:+.2f}%")
        m3.metric("In a position now", sum(r["in_pos"] for r in live.values()))
        m4.metric("Trades today", sum(r["nbuy"] for r in live.values()))
    else:
        st.info("Signals start at 9:45 AM ET. Here is today's list of the 10 most tradeable stocks, chosen from "
                "yesterday's data.")
    st.markdown("<div class='grid'>" + "".join(
        live_card(i, t, res.get(t), prev.get(t), news.get(t), sec.get(t)) for i, t in enumerate(tick, 1))
        + "</div>", unsafe_allow_html=True)

    if live:
        st.markdown("#### Watch one stock")
        pick = st.selectbox("Stock", [t for t in tick if t in live], key="live_pick")
        r = live[pick]
        msg = {"BUY": "BUY: enter at the next minute's price.",
               "HOLD": f"HOLD: in at ${r['entry'] or 0:,.2f}. Sell if it falls to ${r['stop_px'] or 0:,.2f}"
                       + (f" (trailing stop ${r['trail_px']:,.2f})" if r["trail_px"] else "") + ".",
               "SELL": "SELL: get out at the next minute's price.",
               "WAIT": f"WAIT: {r['score']} of 5 entry conditions met. Needs 4, held for 2 minutes in a row."}[r["state"]]
        st.markdown(f"<span class='sig {r['state']}'>{r['state']}</span> &nbsp; {msg}", unsafe_allow_html=True)
        st.write(" · ".join(("✅ " if ok else "❌ ") + lab for lab, ok in r["conds"]))
        b = r["bars"]
        df_ = pd.DataFrame({"Time": b.index, "Price": b.Close.values, "VWAP": r["vwap"]}).melt("Time", var_name="Line", value_name="Value")
        layers = [alt.Chart(df_).mark_line().encode(
            x=alt.X("Time:T", title=None), y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title="Price ($)"),
            color=alt.Color("Line:N", scale=alt.Scale(domain=["Price", "VWAP"], range=["#c4b5fd", "#7c3aed"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("Line:N", legend=None))]
        if not np.isnan(r["or_hi"]):
            layers.append(alt.Chart(pd.DataFrame({"y": [r["or_hi"], r["or_lo"]]})).mark_rule(
                strokeDash=[4, 4], opacity=.5, color="#9fb0d0").encode(y="y:Q"))
        if r["events"]:
            layers.append(alt.Chart(pd.DataFrame(r["events"])).mark_point(size=160, filled=True).encode(
                x="time:T", y="price:Q", color=alt.Color("action:N", scale=alt.Scale(
                    domain=["BUY", "SELL"], range=["#4ade80", "#f87171"]), legend=None),
                tooltip=["time:T", "action", "price", "reason"]))
        st.altair_chart(alt.layer(*layers).properties(height=300), **FULL)
        with st.expander("Every trade today"):
            ev = [dict(Stock=t, Time=e["time"].strftime("%H:%M"), Action=e["action"], Price=round(e["price"], 2),
                       Why=e["reason"]) for t, r_ in live.items() for e in r_["events"]]
            st.dataframe(pd.DataFrame(ev).sort_values("Time") if ev else pd.DataFrame({"Info": ["No trades yet"]}),
                         **FULL)

    lg = _load(LIVE_PATH)
    st.markdown("#### Past days (each day restarts at $100 per stock)")
    if lg is None or lg.empty:
        st.info(f"Daily results appear here after each close, starting {LIVE_START}.")
    else:
        g_ = lg.groupby("date").agg(stocks=("ticker", "count"), end=("end", "sum"), trades=("trades", "sum"),
                                    hold=("stock_open_to_close", "mean"))
        g_["Paper result %"] = (g_.end / (CASH0 * g_.stocks) - 1) * 100
        g_["Just holding the same stocks %"] = g_.hold * 100
        st.line_chart(g_[["Paper result %", "Just holding the same stocks %"]], color=["#c4b5fd", "#7c3aed"])
        st.dataframe(g_[["stocks", "trades", "Paper result %", "Just holding the same stocks %"]].round(2).sort_index(ascending=False),
                     **FULL)


def list_html(top, name, hlabel, full, dec):
    f = lambda x: f"{x * 100:+.{dec}f}%"
    sp = lambda x: f'<b style="color:{proj_color(x, full)}">{f(x)}</b>'
    rows = ""
    for i, (t, r) in enumerate(top.iterrows(), 1):
        nm = name.get(t)
        nm = html.escape(nm if isinstance(nm, str) else "")[:30]
        lab = "High" if r.conf >= .75 else "Medium" if r.conf >= .55 else "Low"
        rows += (f'<tr><td>{i}</td><td><b>{html.escape(t)}</b><br><small>{nm}</small></td>'
                 f'<td>${r.price:,.2f}</td><td>{sp(r.p_lo)} to {sp(r.p_hi)}</td><td>{sp(r.p_mid)}</td>'
                 f'<td><span class="cf {lab}">{lab}</span></td><td>{r.Score:+.2f}</td></tr>')
    return (f'<table class="lst"><tr><th>#</th><th>Stock</th><th>Price</th><th>Likely range ({hlabel})</th>'
            f'<th>Midpoint</th><th>Confidence</th><th>Score</th></tr>{rows}</table>')


# ----------------------------------------------------------------- parlays (DraftKings prices, $10 paper stake each)
import hashlib
import itertools
import json
import zlib
from statistics import NormalDist

try:                                     # player-prop model + slate builder (props.py sits next to app.py)
    import props as PROPS
except Exception as _e:                  # app still works with game-line parlays only
    PROPS = None
    print(f"props.py not loaded: {_e!r}", flush=True)

PLOCK = threading.Lock()
PARLAY_STAKE, PARLAY_MAX_LEGS, PARLAY_SIMS, PARLAY_TILT = 10.0, 3, 60000, 0.5
N_HIT, N_LONG = 5, 5                     # 10 parlays a day: 5 "Hits" (2-4x) and 5 "Longshots" (5-10x)
PROPS_PATH = os.path.splitext(HIST_PATH)[0] + "_props.csv"
PROPS_REFRESH_MIN = int(os.environ.get("PROPS_REFRESH_MIN", "90"))   # minutes between Odds-API prop pulls per game
_PINFO = {"prop_legs": 0, "note": ""}
PARLAY_BOOK = "draftkings"
_DATA_DIR = os.path.dirname(HIST_PATH) or "."
PARLAY_PATH = os.path.splitext(HIST_PATH)[0] + "_parlays.csv"
PARLAY_META = os.path.splitext(HIST_PATH)[0] + "_parlay_meta.csv"
PARLAY_SEED = os.path.join(_DATA_DIR, "parlay_seed_lines.csv")
PARLAY_NOTES = os.path.join(_DATA_DIR, "parlay_notes.csv")
PARLAY_COLS = ["parlay_id", "batch_date", "created_at", "updated_at", "locked", "locked_at", "tier", "n_legs",
               "legs", "stake", "dec_odds", "hit_prob", "ev", "status", "profit", "revisions", "placed_at"]
SPORTS = {"basketball_nba": dict(sd_m=13.0, sd_t=18.0, espn="basketball/nba"),
          "americanfootball_nfl": dict(sd_m=13.3, sd_t=13.0, espn="football/nfl"),
          "americanfootball_ncaaf": dict(sd_m=16.5, sd_t=15.0, espn="football/college-football",
                                         extra=dict(groups="80", limit="400"))}   # groups=80 -> every FBS game
_ND = NormalDist()

# Built-in snapshot (Oct 22 2026 articles) used when no ODDS_API_KEY and no data/parlay_seed_lines.csv exists.
# DraftKings NBA numbers come from an SI article that credits DraftKings; verify on the DK app before betting.
_SEED_CSV = """sport,home,away,commence,book,market,side,line,price
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,spread,home,-7.5,-115
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,spread,away,7.5,-105
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,ml,home,,-298
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,ml,away,,250
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,total,over,235.5,-115
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,draftkings,total,under,235.5,-105
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,spread,home,-8.5,-105
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,spread,away,8.5,-115
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,ml,home,,-350
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,ml,away,,260
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,total,over,235.5,-115
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,hardrock,total,under,235.5,-105
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,spread,home,-8.5,-105
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,spread,away,8.5,-115
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,ml,home,,-300
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,ml,away,,245
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,total,over,235.5,-110
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,bleacherreport,total,under,235.5,-110
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,sportsinteraction,spread,home,-8.5,-110
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,sportsinteraction,spread,away,8.5,-110
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,sportsinteraction,ml,home,,-320
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,sportsinteraction,ml,away,,260
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,betmgm,spread,home,-8.5,-110
basketball_nba,Oklahoma City Thunder,Denver Nuggets,2026-10-23T01:30:00Z,betmgm,spread,away,8.5,-110
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,spread,home,-3,-105
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,spread,away,3,-115
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,ml,home,,-158
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,ml,away,,134
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,total,over,50.5,-115
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,draftkings,total,under,50.5,-105
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,spread,home,-3,-105
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,spread,away,3,-115
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,ml,home,,-156
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,ml,away,,132
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,total,over,50.5,-110
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanduel,total,under,50.5,-110
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanatics,ml,home,,-165
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanatics,ml,away,,135
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanatics,total,over,50.5,-110
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,fanatics,total,under,50.5,-110
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,kalshi,ml,home,,-144
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,kalshi,ml,away,,144
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,polymarket,ml,home,,-150
americanfootball_nfl,Chicago Bears,New England Patriots,2026-10-23T00:15:00Z,polymarket,ml,away,,150
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,spread,home,-5.5,-108
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,spread,away,5.5,-112
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,ml,home,,-218
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,ml,away,,180
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,total,over,227.5,-105
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,draftkings,total,under,227.5,-115
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,spread,home,-5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,spread,away,5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,ml,home,,-205
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,ml,away,,172
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,total,over,225.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanduel,total,under,225.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,spread,home,-5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,spread,away,5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,ml,home,,-210
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,ml,away,,175
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,total,over,226.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betway,total,under,226.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,spread,home,-5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,spread,away,5.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,ml,home,,-215
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,ml,away,,175
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,total,over,227.0,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,fanatics,total,under,227.0,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,spread,home,-4.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,spread,away,4.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,ml,home,,-205
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,ml,away,,163
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,total,over,226.5,-110
basketball_nba,San Antonio Spurs,Houston Rockets,2026-10-24T01:30:00Z,betrivers,total,under,226.5,-110
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,spread,home,-4.5,-110
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,spread,away,4.5,-110
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,ml,home,,-180
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,ml,away,,150
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,total,over,236.5,-112
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,draftkings,total,under,236.5,-108
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,novig,spread,home,-4.5,-106
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,novig,spread,away,4.5,104
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,novig,ml,home,,-163
basketball_nba,Golden State Warriors,Memphis Grizzlies,2026-10-24T02:00:00Z,novig,ml,away,,156
"""

# Expert/analyst read of each game. lean_m: +1 favors the home side, -1 the away side; lean_t: +1 over, -1 under.
# weight = how much the reasoning is trusted (concrete, verifiable, internally consistent = higher).
_NOTES_CSV = """home,away,source,pick,lean_m,lean_t,weight,reasoning,assessment
Oklahoma City Thunder,Denver Nuggets,Sports Illustrated,Thunder -7.5,1,,0.7,Peyton Watson is out so Denver loses wing athleticism and its bench is thin behind DeRozan; OKC averaged +11.7 at home last year,Concrete injury and margin evidence but written against a 7.5 line that later moved to 8.5
Oklahoma City Thunder,Denver Nuggets,Hard Rock Bet,Nuggets +8.5,-1,,0.5,Denver has shown it can compete in Oklahoma City and 8.5 is a lot of points,Reasonable but generic and leans on games played with different rosters
Oklahoma City Thunder,Denver Nuggets,WagerTalk,Nuggets +8.5 and Over 236.5,-1,1,0.3,Says Denver enters at full health and projects a Denver outright win,Low trust: full-health claim conflicts with Watson and Strawther being out and the projection is an outlier
Oklahoma City Thunder,Denver Nuggets,SportsGambler,Nuggets +8.5 and Under 235.5,-1,-1,0.5,Projects a 119-113 OKC win which is inside the spread and under the total,Internally consistent: its own projection supports both picks
Oklahoma City Thunder,Denver Nuggets,BetMGM model,Nuggets cover (70%) while Thunder win (80%),-1,,0.4,Simulation output with no reasoning shown,A 70% cover rate is far from the ~50% market so it is discounted
Oklahoma City Thunder,Denver Nuggets,Betting splits (Hard Rock),Under 235.5,,-1,0.4,Under has 60% of the money on 52% of the tickets,Weak signal that larger bets sit on the Under
Chicago Bears,New England Patriots,Covers (Jason Logan),Bears -2.5 and Under 50.5,1,-1,0.6,Patriots offensive line has multiple starters out and Chicago's blitz-heavy rush had five sacks last week,Concrete and verifiable: Moses Brown and Vera-Tucker all left the Jets game hurt
Chicago Bears,New England Patriots,Sports Illustrated,Bears ML,1,,0.5,Bears scored 90 points in the two games Williams started and finished,Small sample but coherent with the injury picture
Chicago Bears,New England Patriots,Yahoo Sports,Patriots 1H +2.5,,,0.5,Patriots are 6-0 vs the first-quarter spread and Vrabel/Maye are 19-8 vs the first-half spread,Does not contradict the full-game number so it adds no tilt
Chicago Bears,New England Patriots,Bleacher Nation,Bears team total over 26.5,,-1,0.3,Prefers a Chicago team total to the game total and is lukewarm on the over,Thin reasoning
Chicago Bears,New England Patriots,CBS SportsLine model,Under 50.5,,-1,0.4,Under hit in 5 of the last 8 Patriots games and 4 of the last 5 Bears games,Trend-based and shows no injury adjustment
San Antonio Spurs,Houston Rockets,Sports Illustrated,Rockets +6.5,-1,,0.5,Houston's offense ran smoother with VanVleet back and the Rockets were 7-2 against the spread as road underdogs last season,Concrete but small sample; written against 6.5 and the line has since moved to 5.5
San Antonio Spurs,Houston Rockets,Athlon,Rockets ATS lean (Rockets 119-116),-1,,0.3,Projects a narrow Houston win and says San Antonio keeps failing to cover,Low trust: the projected score sits far from the market
San Antonio Spurs,Houston Rockets,BetMGM model,Rockets cover (63%),-1,,0.4,Simulation output with no reasoning shown,A 63% cover rate is far from the ~50% market so it is discounted
San Antonio Spurs,Houston Rockets,iHeart computer model,Spurs 116-113; Rockets ATS; Over,-1,1,0.4,Projects a 3-point Spurs win against a 5.5-6.5 line and 229 points against 226.5,Internally consistent: its own projection supports both picks
Golden State Warriors,Memphis Grizzlies,Dimers model,Grizzlies +4.5 and Under 236.5,-1,-1,0.7,Projects Warriors by 3 against a 4.5 spread and 233 total points against 236.5,Best-stated reasoning of the group: gives its own margin and total next to the market line
Golden State Warriors,Memphis Grizzlies,BetMGM model,Grizzlies cover (57.9%),-1,,0.4,Simulation output with no reasoning shown,Modest edge with no explanation so it is discounted
Golden State Warriors,Memphis Grizzlies,Sports Illustrated,Warriors -3.5,1,,0.3,Expects Curry's supporting cast to bounce back at home,Written against a 3.5 line that has since moved a full point so the edge is likely gone
Golden State Warriors,Memphis Grizzlies,SportsGambler,Warriors ML and Over 236.5,1,1,0.3,Likes the home side on the moneyline and a higher-scoring game,Thin reasoning
"""


# ---- odds math
def am2dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def dec2am(d):
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def fmt_am(a):
    a = int(round(float(a)))
    return f"+{a}" if a > 0 else str(a)


def devig(a, b):
    pa, pb = 1 / am2dec(a), 1 / am2dec(b)
    return pa / (pa + pb), pb / (pa + pb)


_SHORT = {}      # full team name -> ESPN's short name (Bears, Thunder, Penn State...), filled as games are read


def _nick(t):
    return _SHORT.get(str(t)) or str(t).split()[-1]


def _leg_label(home, away, market, side, line):
    if market == "total":
        return f"{'Over' if side == 'over' else 'Under'} {line:g}"
    t = _nick(home if side == "home" else away)
    return f"{t} ML" if market == "ml" else f"{t} {line:+g}"


# ---- lines + scores
def _norm_lines(df):
    cols = ["sport", "home", "away", "commence", "book", "market", "side", "line", "price"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols + ["start", "game_key"])
    df = df[cols].copy()
    df["start"] = pd.to_datetime(df["commence"], utc=True)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["game_key"] = (df.sport + "|" + df.away + "@" + df.home + "|" +
                      df.start.dt.tz_convert(ET).dt.strftime("%Y-%m-%d"))
    return df


def parlay_scores(sport, ymd):
    """{(home_lower, away_lower): {state, home, away, detail}} from ESPN's public scoreboard."""
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{SPORTS[sport]['espn']}/scoreboard",
                         params={"dates": ymd, **SPORTS[sport].get("extra", {})}, timeout=15)
        out = {}
        for ev in r.json().get("events", []):
            comp = ev["competitions"][0]
            c = {x["homeAway"]: x for x in comp["competitors"]}
            t = ev["status"]["type"]
            for _side in ("home", "away"):
                _t = c[_side]["team"]
                if _t.get("shortDisplayName"):
                    _SHORT[_t["displayName"]] = _t["shortDisplayName"]
            out[(c["home"]["team"]["displayName"].lower(), c["away"]["team"]["displayName"].lower())] = dict(
                id=ev.get("id"), state=t.get("state"), home=float(c["home"].get("score") or 0),
                away=float(c["away"].get("score") or 0), detail=t.get("shortDetail", ""))
        return out
    except Exception:
        return {}


def _sports_with_pregame(now):
    """Which sports still have a game that has not started (so we only spend odds-API calls when useful)."""
    need = []
    for sp in SPORTS:
        try:
            states = []
            for d in (now.date(), (now + pd.Timedelta(days=1)).date()):
                states += [v["state"] for v in parlay_scores(sp, d.strftime("%Y%m%d")).values()]
            if "pre" in states:
                need.append(sp)
        except Exception:
            need.append(sp)
    return need


def _num(x):
    """'-110', 'o223.5', 'EVEN', 105.0 -> float (None if it is not a number)."""
    if x is None:
        return None
    t = str(x).strip().lower().lstrip("ou")
    if t in ("even", "ev", "pk", "pick"):
        return 100.0 if t in ("even", "ev") else 0.0
    try:
        return float(t)
    except ValueError:
        return None


def _espn_odds(now):
    """Live lines for games that have not started, read from ESPN's public scoreboard (no key needed).
    ESPN's odds provider is DraftKings, so these rows are stored under the book name 'draftkings'."""
    rows = []
    for sp, cfg in SPORTS.items():
        found = 0
        for d in (now.date(), (now + pd.Timedelta(days=1)).date()):
            try:
                r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{cfg['espn']}/scoreboard",
                                 params={"dates": d.strftime("%Y%m%d"), **cfg.get("extra", {})}, timeout=20)
                events = r.json().get("events", [])
            except Exception as e:
                print(f"espn odds {sp}: {e!r}", flush=True)
                continue
            for ev in events:
                try:
                    comp = ev["competitions"][0]
                    if ev["status"]["type"].get("state") != "pre" or not comp.get("odds"):
                        continue
                    c = {x["homeAway"]: x for x in comp["competitors"]}
                    home, away = c["home"]["team"]["displayName"], c["away"]["team"]["displayName"]
                    for _side in ("home", "away"):
                        _t = c[_side]["team"]
                        if _t.get("shortDisplayName"):
                            _SHORT[_t["displayName"]] = _t["shortDisplayName"]
                    o = next((x for x in comp["odds"] if "draft" in str(x.get("provider", {}).get("name", "")).lower()),
                             comp["odds"][0])
                    book = ("draftkings" if "draft" in str(o.get("provider", {}).get("name", "")).lower()
                            else re.sub(r"[^a-z0-9]", "", str(o.get("provider", {}).get("name", "espn")).lower()))
                    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
                    base = dict(sport=sp, home=home, away=away, commence=ev["date"], book=book)
                    # moneyline
                    mh = _num(ho.get("moneyLine"))
                    ma = _num(ao.get("moneyLine"))
                    if mh is None or ma is None:
                        ml = o.get("moneyline") or {}
                        mh = _num(((ml.get("home") or {}).get("close") or {}).get("odds"))
                        ma = _num(((ml.get("away") or {}).get("close") or {}).get("odds"))
                    if mh is not None and ma is not None:
                        rows += [dict(base, market="ml", side="home", line=None, price=mh),
                                 dict(base, market="ml", side="away", line=None, price=ma)]
                    # point spread (line is the home team's; the favourite carries the minus sign)
                    sh = sa = None
                    ps = o.get("pointSpread") or {}
                    lh = _num(((ps.get("home") or {}).get("close") or {}).get("line"))
                    if lh is None and o.get("spread") is not None:
                        sp_abs = abs(float(o["spread"]))
                        lh = 0.0 if sp_abs == 0 else (-sp_abs if ho.get("favorite") else sp_abs)
                    if lh is not None:
                        sh = _num(ho.get("spreadOdds")) or _num(((ps.get("home") or {}).get("close") or {}).get("odds")) or -110.0
                        sa = _num(ao.get("spreadOdds")) or _num(((ps.get("away") or {}).get("close") or {}).get("odds")) or -110.0
                        rows += [dict(base, market="spread", side="home", line=lh, price=sh),
                                 dict(base, market="spread", side="away", line=-lh, price=sa)]
                    # total
                    tl = _num(o.get("overUnder"))
                    tt = o.get("total") or {}
                    if tl is None:
                        tl = _num(((tt.get("over") or {}).get("close") or {}).get("line"))
                    if tl is not None:
                        ov = _num(o.get("overOdds")) or _num(((tt.get("over") or {}).get("close") or {}).get("odds")) or -110.0
                        un = _num(o.get("underOdds")) or _num(((tt.get("under") or {}).get("close") or {}).get("odds")) or -110.0
                        rows += [dict(base, market="total", side="over", line=tl, price=ov),
                                 dict(base, market="total", side="under", line=tl, price=un)]
                    found += 1
                except Exception as e:
                    print(f"espn odds parse {sp}: {e!r}", flush=True)
        print(f"espn odds {sp}: {found} upcoming games with lines", flush=True)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["sport", "home", "away", "commence", "book", "market", "side"]) if len(df) else df


def parlay_fetch_odds(now=None):
    """-> (long DataFrame of book lines, source label). Live DraftKings+others via The Odds API if
    ODDS_API_KEY is set, otherwise a manual snapshot (file data/parlay_seed_lines.csv or the built-in one)."""
    now = now or datetime.now(ET)
    key = _cfg("ODDS_API_KEY")
    if key:
        need = _sports_with_pregame(now)
        if not need:
            return _norm_lines(None), "The Odds API (no upcoming games)"
        rows = []
        for sp in need:
            try:
                r = requests.get(f"https://api.the-odds-api.com/v4/sports/{sp}/odds", timeout=20, params=dict(
                    apiKey=key, regions="us", markets="h2h,spreads,totals", oddsFormat="american", dateFormat="iso"))
                r.raise_for_status()
                for g in r.json():
                    for bk in g.get("bookmakers", []):
                        for m in bk.get("markets", []):
                            mk = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(m["key"])
                            for o in m.get("outcomes", []):
                                side = (o["name"].lower() if mk == "total" else
                                        "home" if o["name"] == g["home_team"] else "away")
                                rows.append(dict(sport=sp, home=g["home_team"], away=g["away_team"],
                                                 commence=g["commence_time"], book=bk["key"], market=mk,
                                                 side=side, line=o.get("point"), price=o["price"]))
            except Exception as e:
                print(f"odds api {sp}: {e!r}", flush=True)
        if rows:
            return _norm_lines(pd.DataFrame(rows)), "The Odds API (live)"
    try:
        live = _espn_odds(now)
        if len(live):
            return _norm_lines(live), "DraftKings lines via ESPN (live, free)"
    except Exception as e:
        print(f"espn odds: {e!r}", flush=True)
    try:
        df = pd.read_csv(PARLAY_SEED) if os.path.exists(PARLAY_SEED) else pd.read_csv(io.StringIO(_SEED_CSV))
        return _norm_lines(df), "Hand-entered public DraftKings lines from Oct 22-23 (add an ODDS_API_KEY for live prices)"
    except Exception:
        return _norm_lines(None), "no odds available"


def parlay_notes():
    try:
        return pd.read_csv(PARLAY_NOTES) if os.path.exists(PARLAY_NOTES) else pd.read_csv(io.StringIO(_NOTES_CSV))
    except Exception:
        return pd.DataFrame(columns=["home", "away", "source", "pick", "lean_m", "lean_t", "weight",
                                     "reasoning", "assessment"])


# ---- the model: market consensus (de-vigged, every book) + a small, capped expert tilt
def _tilt(n, col):
    x = n[[col, "weight"]].apply(pd.to_numeric, errors="coerce").dropna()
    if x.empty or x.weight.sum() <= 0:
        return 0.0
    return float(np.clip(PARLAY_TILT * (x[col] * x.weight).sum() / x.weight.sum(), -1, 1))


def parlay_model(g, sp, notes):
    s, mus, ts = SPORTS[sp], [], []
    for _, b in g.groupby("book"):
        for mk, m in b.groupby("market"):
            h, a = m[m.side.isin(["home", "over"])], m[m.side.isin(["away", "under"])]
            if h.empty or a.empty:
                continue
            h, a = h.iloc[0], a.iloc[0]
            z = _ND.inv_cdf(min(max(devig(h.price, a.price)[0], .03), .97))
            if mk == "spread" and pd.notna(h.line):
                mus.append((s["sd_m"] * z - h.line, 1.0))          # home covers if margin + line > 0
            elif mk == "ml":
                mus.append((s["sd_m"] * z, 0.5))                   # moneylines count half (longshot bias)
            elif mk == "total" and pd.notna(h.line):
                ts.append(h.line + s["sd_t"] * z)
    if not mus:
        return None
    r0 = g.iloc[0]
    n = notes[(notes.home == r0.home) & (notes.away == r0.away)]
    mu0 = sum(x * w for x, w in mus) / sum(w for _, w in mus)
    t0 = float(np.mean(ts)) if ts else None
    return dict(mu0=mu0, tot0=t0, mu=mu0 + _tilt(n, "lean_m"),
                tot=None if t0 is None else t0 + _tilt(n, "lean_t"), n_books=int(g.book.nunique()))


def _sims(model, sp, key, adj):
    s, n = SPORTS[sp], PARLAY_SIMS
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    mu, tt = (model["mu"], model["tot"]) if adj else (model["mu0"], model["tot0"])
    return mu + s["sd_m"] * z1, (None if tt is None else tt + s["sd_t"] * z2)


def _leg_res(market, side, line, margin, total):
    """1 = win, 0 = push, -1 = loss for every simulated game."""
    if market == "ml":
        return np.where((margin if side == "home" else -margin) > 0, 1, -1).astype(np.int8)
    if market == "spread":
        return np.sign((np.round(margin) if side == "home" else -np.round(margin)) + line).astype(np.int8)
    d = np.round(total) - line
    return (np.sign(d) if side == "over" else -np.sign(d)).astype(np.int8)


def _eval(legs):
    """Hit chance, estimated DraftKings payout and EV for a combination of legs. Legs in the same game are
    simulated jointly; the SGP payout is the product of leg prices scaled down by the correlation (est.)."""
    byg = {}
    for l in legs:
        byg.setdefault(l["game_key"], []).append(l)
    ev, hit, dec = 1.0, 1.0, 1.0
    for ls in byg.values():
        scale = 1.0
        if len(ls) > 1:
            pj = np.all([x["r0"] == 1 for x in ls], axis=0).mean()
            pi = float(np.prod([(x["r0"] == 1).mean() for x in ls]))
            scale = min(1.0, pi / pj) if pj > 0 else 1.0
        pay = np.ones(len(ls[0]["r1"]))
        for x in ls:
            pay = pay * np.where(x["r1"] == 1, x["dec"], 1.0)
        pay = np.where(np.any([x["r1"] == -1 for x in ls], axis=0), 0.0, pay * scale)
        ev *= pay.mean()
        hit *= np.all([x["r1"] == 1 for x in ls], axis=0).mean()
        dec *= float(np.prod([x["dec"] for x in ls])) * scale
    return dict(dec=dec, hit=float(hit), ev=float(ev - 1))


def _tier(hit):
    return "Safer" if hit >= .38 else "Balanced" if hit >= .18 else "Longshot"


def parlay_legs(lines, notes, now, started=False):
    """DraftKings legs for games that have (started=False: not yet / started=True: already) kicked off."""
    nu, out = pd.Timestamp(now).tz_convert("UTC"), []
    for gk, g in lines.groupby("game_key"):
        r0 = g.iloc[0]
        if (r0.start > nu + pd.Timedelta(minutes=2)) == started:
            continue
        model = parlay_model(g, r0.sport, notes)
        if not model:
            continue
        m1, t1 = _sims(model, r0.sport, gk, True)
        m0, t0 = _sims(model, r0.sport, gk, False)
        for _, d in g[g.book == PARLAY_BOOK].iterrows():
            if d.market == "total" and (t1 is None or pd.isna(d.line)):
                continue
            if d.market in ("spread", "total") and pd.isna(d.line):
                continue
            r1 = _leg_res(d.market, d.side, d.line, m1, t1)
            r0_ = _leg_res(d.market, d.side, d.line, m0, t0)
            dec = am2dec(d.price)
            mult = np.where(r1 == 1, dec, np.where(r1 == 0, 1.0, 0.0)).mean()
            out.append(dict(game_key=gk, sport=r0.sport, home=r0.home, away=r0.away, start=r0.start.isoformat(),
                            market=d.market, side=d.side, line=None if pd.isna(d.line) else float(d.line),
                            price=float(d.price), dec=dec, p=float((r1 == 1).mean()), ev1=float(mult - 1),
                            label=_leg_label(r0.home, r0.away, d.market, d.side, d.line),
                            match=f"{_nick(r0.away)} @ {_nick(r0.home)}", n_books=model["n_books"],
                            r1=r1, r0=r0_))
    return out


def parlay_build(legs, top_per_tier=2, max_total=6):
    """Rank every 2-3 leg combination by EV and keep the best few in each risk tier."""
    pool = sorted([l for l in legs if l["p"] >= .25], key=lambda l: -l["ev1"])[:14]
    cands = []
    for k in range(2, PARLAY_MAX_LEGS + 1):
        for combo in itertools.combinations(pool, k):
            if len({(l["game_key"], l["market"]) for l in combo}) < k:
                continue
            if any(x["game_key"] == y["game_key"] and {x["market"], y["market"]} == {"ml", "spread"}
                   for x, y in itertools.combinations(combo, 2)):
                continue                                  # moneyline + spread in one game is the same bet twice
            e = _eval(list(combo))
            cands.append(dict(legs=list(combo), tier=_tier(e["hit"]), **e))
    cands.sort(key=lambda c: -c["ev"])
    chosen, used = [], {}
    for tier in ("Safer", "Balanced", "Longshot"):
        n = 0
        for c in [x for x in cands if x["tier"] == tier]:
            ids = [(l["game_key"], l["market"], l["side"]) for l in c["legs"]]
            if n >= top_per_tier or any(used.get(i, 0) >= 2 for i in ids):
                continue
            chosen.append(c)
            n += 1
            for i in ids:
                used[i] = used.get(i, 0) + 1
    return sorted(chosen, key=lambda c: -c["ev"])[:max_total]


def _pid(batch, legs):
    return hashlib.md5((batch + "|" + "|".join(sorted(
        f"{l['game_key']}:{l['market']}:{l['side']}:{l.get('player', '')}:{l.get('line')}"
        for l in legs))).encode()).hexdigest()[:10]


def _clean(l):
    return {k: v for k, v in l.items() if k not in ("r0", "r1")} | {"status": "pending"}


# ---- grading + storage
def leg_status(leg, sc):
    if not sc or sc["state"] == "pre":
        return "pending"
    h, a, final = sc["home"], sc["away"], sc["state"] == "post"
    if leg["market"] == "total":
        t, line = h + a, leg["line"]
        if leg["side"] == "over":
            if t > line:
                return "won"
        elif t > line:
            return "lost"
        if final:
            return "push" if t == line else ("lost" if leg["side"] == "over" else "won")
        return "live"
    d = (h - a) if leg["side"] == "home" else (a - h)
    if leg["market"] == "spread":
        d += leg["line"]
    if final:
        return "won" if d > 0 else "lost" if d < 0 else "push"
    return "winning" if d > 0 else "losing" if d < 0 else "tied"


def parlay_status(legs, started):
    s = [l["status"] for l in legs]
    if "lost" in s:
        return "lost"
    if all(x in ("won", "push") for x in s):
        return "push" if all(x == "push" for x in s) else "won"
    return "live" if started else "open"


def _leg_result(leg, sc):
    """Plain-English line shown under a leg once its game is underway or over."""
    if not sc or sc["state"] == "pre":
        return ""
    h, a = sc["home"], sc["away"]
    pre = "Final" if sc["state"] == "post" else "Now"
    if leg["market"] == "total":
        return f"{pre}: {_nick(leg['home'])} {h:.0f}, {_nick(leg['away'])} {a:.0f} (total {h + a:.0f})"
    m = (h - a) if leg["side"] == "home" else (a - h)
    who = _nick(leg["home"] if leg["side"] == "home" else leg["away"])
    return f"{pre}: {_nick(leg['home'])} {h:.0f}, {_nick(leg['away'])} {a:.0f} ({who} {m:+.0f})"


def _profit(status, stake, dec, legs):
    if status == "lost":
        return -stake
    if status == "won":
        return round(stake * (dec / float(np.prod([l["dec"] for l in legs if l["status"] == "push"] or [1.0])) - 1), 2)
    return 0.0


def _read_local(path, cols):
    if os.path.exists(path):
        try:
            d = pd.read_csv(path, dtype={"parlay_id": str, "batch_date": str})
            for c in cols:
                if c not in d:
                    d[c] = ""
            for c in ("created_at", "updated_at", "locked_at", "placed_at", "tier", "legs", "status"):
                if c in d:                      # blank columns load as float: keep them text so timestamps fit
                    d[c] = d[c].astype(object).where(d[c].notna(), "")
            if "locked" in d:
                d["locked"] = d["locked"].astype(str).eq("True")
            return d
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def _settle(P, now):
    """Lock parlays whose first game has started, then grade every leg from live/final scores."""
    cache, boxes, nu = {}, {}, pd.Timestamp(now).tz_convert("UTC")
    for i, r in P.iterrows():
        if r.status in ("won", "lost", "push"):
            continue
        legs = json.loads(r.legs)
        started = min(pd.Timestamp(l["start"]) for l in legs) <= nu
        if started and not bool(r.locked):
            P.loc[i, ["locked", "locked_at"]] = [True, now.isoformat(timespec="seconds")]
            if not str(r.placed_at or ""):           # the last price before kickoff is the bet that counts
                P.loc[i, "placed_at"] = r.updated_at
        if not started:
            continue
        for l in legs:
            ck = (l["sport"], pd.Timestamp(l["start"]).tz_convert(ET).strftime("%Y%m%d"))
            if ck not in cache:
                cache[ck] = parlay_scores(*ck)
            _sc = cache[ck].get((l["home"].lower(), l["away"].lower()))
            if l.get("market") == "prop" and PROPS is not None:
                bk = (l["sport"], (_sc or {}).get("id"))
                if _sc and _sc.get("state") != "pre" and bk not in boxes:
                    boxes[bk] = PROPS.fetch_box(l["sport"], bk[1])
                l["status"], _txt = PROPS.grade_prop(l, _sc, boxes.get(bk, {}))
                l["result"] = _txt or l.get("result", "")
            else:
                l["status"] = leg_status(l, _sc)
                l["result"] = _leg_result(l, _sc) or l.get("result", "")
        st_ = parlay_status(legs, True)
        P.loc[i, ["legs", "status", "profit", "updated_at"]] = [
            json.dumps(legs), st_, _profit(st_, r.stake, r.dec_odds, legs), now.isoformat(timespec="seconds")]
    return P


def _prop_legs(now):
    """Player-prop legs priced at DraftKings (empty list when there is no ODDS_API_KEY or no props yet)."""
    _PINFO.update(prop_legs=0, note="")
    if PROPS is None:
        _PINFO["note"] = "props.py missing"
        return []
    try:
        df, note = PROPS.fetch_props(now, _cfg("ODDS_API_KEY"), PROPS_PATH, PROPS_REFRESH_MIN)
        legs = PROPS.build_prop_legs(df, now)
        _PINFO.update(prop_legs=len(legs), note=note)
        return legs
    except Exception as e:
        _PINFO["note"] = f"props error: {e!r}"
        print(_PINFO["note"], flush=True)
        return []


def _refresh(P, lines, notes, now, prop_legs=()):
    """Re-price the not-yet-started parlays (up to kickoff the batch can still change; after it, it is frozen).
    Each day gets N_HIT "Hit" and N_LONG "Longshot" parlays; ones already frozen at kickoff count toward the quota."""
    stamp = now.isoformat(timespec="seconds")
    P["locked"] = P["locked"].astype(str).eq("True")
    in_lines = set(lines.game_key) | {l["game_key"] for l in prop_legs}
    keep, new_ids = [], set()
    legs_all = parlay_legs(lines, notes, now) + list(prop_legs)
    by_date = {}
    for l in legs_all:
        by_date.setdefault(pd.Timestamp(l["start"]).tz_convert(ET).strftime("%Y-%m-%d"), []).append(l)
    rows = []
    for day, ls in by_date.items():
        frozen = P[(P.batch_date == day) & ((P.locked.astype(bool)) | (P.status != "open"))]
        have_h, have_l = int((frozen.tier == "Hit").sum()), int((frozen.tier == "Longshot").sum())
        if PROPS is not None:
            built = PROPS.build_slate(ls, max(N_HIT - have_h, 0), max(N_LONG - have_l, 0))
        else:
            built = parlay_build(ls)
        for c in built:
            pid = _pid(day, c["legs"])
            new_ids.add(pid)
            old = P[P.parlay_id == pid]
            rev = int(old.revisions.iloc[0]) + int(abs(old.dec_odds.iloc[0] - round(c["dec"], 4)) > 1e-6) if len(old) else 0
            rows.append(dict(parlay_id=pid, batch_date=day,
                             created_at=old.created_at.iloc[0] if len(old) else stamp, updated_at=stamp,
                             locked=False, locked_at="", tier=c["tier"], n_legs=len(c["legs"]),
                             legs=json.dumps([_clean(l) for l in c["legs"]]), stake=PARLAY_STAKE,
                             dec_odds=round(c["dec"], 4), hit_prob=round(c["hit"], 4), ev=round(c["ev"], 4),
                             status="open", profit=0.0, revisions=rev, placed_at=""))
    for i, r in P.iterrows():
        if bool(r.locked) or r.status != "open":
            keep.append(r.to_dict())            # frozen (kicked off) or finished: never rewritten
        elif r.parlay_id not in new_ids and not {l["game_key"] for l in json.loads(r.legs)} <= in_lines:
            keep.append(r.to_dict())            # games not in this odds pull: leave as is
    return pd.DataFrame(keep + rows, columns=PARLAY_COLS)


def parlay_step():
    """Every 15 minutes: lock and grade parlays that are underway, re-price the ones that have not started."""
    with PLOCK:
        now = datetime.now(ET)
        P = _with_seed(_read_local(PARLAY_PATH, PARLAY_COLS))
        P = _settle(P, now) if len(P) else P
        lines, src = parlay_fetch_odds(now)
        pl = _prop_legs(now)
        if len(lines) or pl:
            P = _refresh(P, lines, parlay_notes(), now, pl)
        if len(P):
            P = P.drop_duplicates("parlay_id", keep="first")
        _save(P[PARLAY_COLS] if len(P) else pd.DataFrame(columns=PARLAY_COLS), PARLAY_PATH)
        _save(pd.DataFrame([dict(updated_at=now.isoformat(timespec="seconds"), source=src,
                                 n_lines=len(lines), games=lines.game_key.nunique() if len(lines) else 0,
                                 prop_legs=_PINFO["prop_legs"], props_note=_PINFO["note"])]),
              PARLAY_META)
        print(f"parlays: {len(P)} rows ({src})", flush=True)


@st.cache_resource
def start_parlay_tracker():
    if _cfg("DISABLE_APP_TRACKER"):      # GitHub Actions owns the 15-minute refresh
        return None

    def loop():
        while True:
            try:
                parlay_step()
            except Exception as e:
                print(f"parlay_step error: {e!r}", flush=True)
            time.sleep(15 * 60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ---- built-in copy of the first graded bets (Oct 22), so the history survives a missing or reset data folder
PARLAY_SEED_B64 = (
    "H4sIAHfv22oC/+2WyW7bMBCG73kKQacWoQXui09BW/SUNpf0VBQCLdG2YFoyJNpB0PTdS2ppbCVxmyJFCiQ6UNTMaDikPvyjja6t"
    "vk6LHMy0y5Zprp0BWW38LU+1A9tNPkxtla1M3t+CwRWmBmVqzaIB7dA4vTIgN1la5XkDloVLN3U1A2YXXG7bAP84Lxyoza5oiqr0"
    "BquzNtuJUvM5IgIrCjDEfILgBOO96SVGUwSnEE4g9eOth1xCNMVySmTvuay35oj7nba69IsCDOKv3+N4odcmXZnrOJ5GcTzTzcq4"
    "mbY2LWf65oMpd6aOPm8XC+Oas4uV1ctqraP3hbuOLpfbMjf1zW2RcQx8jmZT1e6+dJ3bJzCd9950XZC+0n1FhyX0Kzg9rHC4URJO"
    "6BSGsYtc69oX0IW6ymnbJyjyvoa9NW1RtkZMWMKCYVMXWWvxC7AEBpP/usGAEsUwkfB2bOODCyYMQkr2ruAyO9QmggnEWHFCFYbt"
    "iFq31TNju4K+hIK6GoYteDI7X38I0dnhYZXprKpWTYhh/fF42rpXrqqyC6pNs7X9SXwsSm2nQ5JIcTCcbyRp9KY9qAhJ+DaOf4Do"
    "RUGyvkNIV8o+IOXW2kM+sJIjPghhTEDKlVCSMyH2+BBYSDaGQijI4K+XpBxRMXyqT+fPBcVgP0U4YPEtBsjXDXDCPe0AJoRTCsJW"
    "GFTALwAQTyAF8GENO8nmuZ5znSH8qnmvmveqec8FSbPxfzz573VvIu4wgsaMSK4YZxgJohCBFO8zgtAxRiiR1BctlWQMCiLYAxIY"
    "qvjPRJAknDLmRRAz3okg949BBDFPKD8qgkiwXCOJMv6PRfC8KhfNsnIvTwTv9vRulaM9HTN4gDbp2e9YxgKjcQ+nggqvg8NFxwAP"
    "ID1ND7dV4x7N7zCfdPy+MJ27txlWuz/phY/ROaq8a/8accK5/91jhGOCpCRqjMnF7il74V9hctAMB5Hjid+u/6FLEOEoiBxCGIGQ"
    "H0zagGMq9xNJu/BfaQ8AAA=="
)


def _with_seed(P):
    """Add the built-in graded bets if the data file does not have them yet."""
    import base64, gzip, io
    seed = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(PARLAY_SEED_B64))), dtype={"parlay_id": str, "batch_date": str})
    for c in PARLAY_COLS:
        if c not in seed:
            seed[c] = ""
    seed["locked"] = seed["locked"].astype(str).eq("True")
    if P is None or not len(P):
        return seed[PARLAY_COLS].copy()
    have = set(P["parlay_id"].astype(str))
    add = seed[~seed["parlay_id"].isin(have)]
    return pd.concat([P, add[PARLAY_COLS]], ignore_index=True) if len(add) else P


def _html_table(df):
    """Plain HTML table: no extra browser components to load, so it also works through a tunnel."""
    th = "".join("<th>" + html.escape(str(c)) + "</th>" for c in df.columns)
    body = "".join("<tr>" + "".join("<td>" + html.escape("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)) + "</td>"
                                    for v in r) + "</tr>" for r in df.itertuples(index=False))
    st.markdown("<div class='ptb'><table><thead><tr>" + th + "</tr></thead><tbody>" + body + "</tbody></table></div>",
                unsafe_allow_html=True)


# ---- the tab
PL_CSS = """<style>
.plk{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:4px 0 6px}
.plk div{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 14px}
.plk small{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);margin-bottom:2px}
.plk b{font-family:var(--fd);font-size:22px;letter-spacing:-.02em}
.plk .pos,.pb .pos{color:#8fe3c3}.plk .neg,.pb .neg{color:#f6aab9}
.pg{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px;margin:8px 0 4px}
.pb{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:16px 18px;color:var(--ink)}
.pb-h{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.pb-t{font-family:var(--fd);font-weight:800;font-size:16px;letter-spacing:-.02em}
.pb-s{font-size:11.5px;color:var(--mute);margin-top:2px}
.chip{padding:3px 11px;border-radius:999px;font-weight:800;font-size:11px;letter-spacing:.05em;white-space:nowrap;text-transform:uppercase}
.chip.won{background:rgba(95,211,168,.14);color:#8fe3c3}.chip.lost{background:rgba(240,138,160,.14);color:#f6aab9}
.chip.live{background:rgba(139,92,246,.2);color:var(--v3)}.chip.open,.chip.wait,.chip.push{background:rgba(167,139,250,.08);color:var(--mute)}
.pb-big{font-family:var(--fd);font-size:26px;font-weight:700;letter-spacing:-.03em;margin:12px 0 0}
.pb-cap{font-size:12px;color:var(--mute);margin-bottom:8px}
.pb-leg{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-top:1px solid var(--line)}
.pb-leg .ic{width:20px;text-align:center;flex:none}
.pb-leg .tx{flex:1;font-size:13.5px;min-width:0}.pb-leg .tx small{display:block;color:var(--mute);font-size:11.5px;margin-top:1px}
.pb-leg .px2{font-weight:700;font-size:13px;color:var(--v3);flex:none}
.ptb{overflow-x:auto;margin:6px 0}.ptb table{border-collapse:collapse;width:100%;font-size:12.5px}.ptb th{text-align:left;color:var(--mute);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}.ptb td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.pb-f{margin-top:8px;padding-top:10px;border-top:1px solid var(--line);font-size:11.5px;color:var(--mute);line-height:1.5}
</style>"""

_TIER_NOTE = {"Safer": "Better chance of hitting, smaller payout", "Balanced": "Middle ground",
              "Hit": "Most likely combination paying 2-4x", "Longshot": "Most likely combination paying 5-10x"}
_CHIP = {"open": "Not started", "wait": "Awaiting result", "live": "In play", "won": "Won", "lost": "Lost",
         "push": "Refunded"}
_ICON = {"won": "✅", "lost": "❌", "push": "➖", "winning": "🟢", "losing": "🔴", "tied": "🟡", "live": "🟢",
         "pending": "⏳"}


def _s(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize(ET) if t.tzinfo is None else t


def _tfmt(x, day=True):
    t = _ts(x).tz_convert(ET)
    s = t.strftime("%I:%M %p").lstrip("0")
    return f"{t:%a} {s} ET" if day else f"{s} ET"


def _money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def _day_name(d):
    t = pd.Timestamp(d)
    return f"{t:%A, %b} {t.day}"


def _tip(legs):
    return min(_ts(l["start"]) for l in legs)


def _placed_ok(r, legs):
    """Only bets that were priced before their first kickoff count: no after-the-fact entries."""
    ts = _s(r.get("placed_at")) or _s(r.get("created_at"))
    return (not ts) or _ts(ts) <= _tip(legs)


def _eff_status(r, legs, now):
    s = _s(r.status)
    if s in ("won", "lost", "push", "live"):
        return s
    return "wait" if _tip(legs) <= _ts(now) else "open"


def parlay_card(r, now):
    legs = json.loads(r.legs)
    stt = _eff_status(r, legs, now)
    stake, dec = float(r.stake), float(r.dec_odds)
    profit = float(r.profit or 0)
    tip = _tip(legs)
    if stt in ("won", "lost", "push"):
        cls = "pos" if profit > 0 else "neg" if profit < 0 else ""
        big = f'<div class="pb-big {cls}">{_money(profit)}</div><div class="pb-cap">Result of a ${stake:,.0f} bet</div>'
    else:
        big = (f'<div class="pb-big">${stake * (dec - 1):,.2f}</div><div class="pb-cap">'
               f'to win on a ${stake:,.0f} bet (pays ${stake * dec:,.2f} total)</div>')
    rows = ""
    for l in legs:
        sub = _s(l.get("result")) or f"tip-off {_tfmt(l['start'])}"
        rows += (f'<div class="pb-leg"><span class="ic">{_ICON.get(l.get("status"), "⏳")}</span><div class="tx">'
                 f'<b>{html.escape(l["label"])}</b><small>{html.escape(l["match"])} · {html.escape(sub)}</small></div>'
                 f'<span class="px2">{fmt_am(l["price"])}</span></div>')
    placed = _s(r.get("placed_at")) or (_s(r.updated_at) if bool(r.locked) else "")
    foot = (f"Bet placed at the last pre-game price ({_tfmt(placed)}), tip-off {_tfmt(tip, False)}" if placed
            else f"Last priced {_tfmt(r.updated_at, False)}; re-prices every 15 min until tip-off")
    tier = _s(r.tier)
    return (f'<div class="pb"><div class="pb-h"><div><div class="pb-t">{r.n_legs}-leg parlay · {html.escape(tier)}</div>'
            f'<div class="pb-s">{_TIER_NOTE.get(tier, "")}</div></div><span class="chip {stt}">{_CHIP[stt]}</span></div>'
            f'{big}{rows}<div class="pb-f">Odds {fmt_am(dec2am(dec))} · model hit chance {r.hit_prob:.0%} · '
            f'expected value {r.ev * 100:+.1f}%<br>{foot}</div></div>')


def _cards(df, now):
    st.markdown("<div class='pg'>" + "".join(parlay_card(r, now) for _, r in df.iterrows()) + "</div>",
                unsafe_allow_html=True)


def _day_summary(d):
    """One line for a day: how many bets and, once graded, the record and profit."""
    done = d[d.status.isin(["won", "lost", "push"])]
    w, l_ = int((done.status == "won").sum()), int((done.status == "lost").sum())
    if len(done) == len(d):
        return f"{len(d)} bets · {w}-{l_} · {_money(done.profit.sum())}"
    return f"{len(d)} bets · {len(done)} graded so far" + (f" · {_money(done.profit.sum())}" if len(done) else "")


def _prep(P, now):
    P = P.copy()
    P["locked"] = P["locked"].astype(str).eq("True")
    ok = []
    for i, r in P.iterrows():
        try:
            if _placed_ok(r, json.loads(r.legs)):
                ok.append(i)
        except Exception:
            pass
    P = P.loc[ok].copy()
    P["status"] = P["status"].fillna("").astype(str)
    P["profit"] = pd.to_numeric(P["profit"], errors="coerce").fillna(0.0)
    P["started"] = [_tip(json.loads(x)) <= _ts(now) for x in P.legs]
    return P


def render_parlays():
    now = datetime.now(ET)
    st.markdown(PL_CSS, unsafe_allow_html=True)
    st.subheader("Parlays")
    storage_banner()
    st.caption("Educational paper tracking, not advice and not real bets. Each parlay is a $10 pretend stake priced at "
               "DraftKings. Only bets priced before kick-off are shown. Same-game payouts are estimates, so check the real "
               "price on the DraftKings slip.")
    c1, c2 = st.columns([1, 4])
    if c1.button("Refresh parlays now"):
        with st.spinner("Pricing parlays..."):
            parlay_step()
        st.rerun()
    meta = _load(PARLAY_META)
    if meta is not None and len(meta):
        m = meta.iloc[-1]
        c2.caption(f"Prices from: {m.source}. Last check {_tfmt(m.updated_at)}: "
                   f"{int(m.games)} upcoming game(s) with lines.")
        _pn = _s(m.get("props_note")) if hasattr(m, "get") else ""
        if _pn:
            _pl = pd.to_numeric(m.get("prop_legs"), errors="coerce")
            c2.caption(f"Player props: {_pn} · {0 if pd.isna(_pl) else int(_pl)} priced leg(s).")
        if int(m.games) == 0:
            c2.caption("No upcoming NBA, NFL or college football games with DraftKings lines right now, so there are no "
                       "new bets to place. New ones appear on their own once lines are posted.")
    P = _with_seed(_load(PARLAY_PATH))
    P = _prep(P, now) if P is not None and len(P) else None
    if P is None or P.empty:
        st.info("No parlays yet. They appear after the first refresh (every 15 minutes) once games have lines.")
    else:
        done = P[P.status.isin(["won", "lost", "push"])]
        n_start = int(P.started.sum())
        w, l_ = int((done.status == "won").sum()), int((done.status == "lost").sum())
        prof = float(done.profit.sum())
        roi = f"{prof / (len(done) * PARLAY_STAKE) * 100:+.0f}%" if len(done) else "n/a"
        cls = "pos" if prof > 0 else "neg" if prof < 0 else ""
        st.markdown(
            f"<div class='plk'><div><small>Bets placed</small><b>{n_start}</b></div>"
            f"<div><small>Won - Lost</small><b>{w} - {l_}</b></div>"
            f"<div><small>Total staked</small><b>${n_start * PARLAY_STAKE:,.0f}</b></div>"
            f"<div><small>Profit so far</small><b class='{cls}'>{_money(prof)}</b></div>"
            f"<div><small>Return on stake</small><b class='{cls}'>{roi}</b></div></div>", unsafe_allow_html=True)
        if len(done) < 100:
            st.caption(f"Early days: only {len(done)} graded bets. Parlays swing wildly, so a good or bad day says almost "
                       "nothing about the method. Judge it over hundreds of bets.")

        today = now.strftime("%Y-%m-%d")
        st.markdown("### Today's bets")
        T = P[P.batch_date == today].sort_values("ev", ascending=False)
        if len(T):
            st.caption(_day_summary(T))
            Th, Tl = T[T.tier == "Hit"], T[T.tier == "Longshot"]
            Tother = T[~T.tier.isin(["Hit", "Longshot"])]
            if len(Th):
                st.markdown(f"#### Hits · 2-4x payout ({len(Th)} of {N_HIT})")
                st.caption("The highest-probability combinations that still pay 2-4x. A parlay paying Nx can only hit about "
                           "1 time in N on fair odds, so 'almost certain' and 2x+ cannot both be true: these are the best "
                           "available, not sure things.")
                _cards(Th, now)
            if len(Tl):
                st.markdown(f"#### Longshots · 5-10x payout ({len(Tl)} of {N_LONG})")
                st.caption("The most likely combinations paying 5-10x. Expect most of these to lose.")
                _cards(Tl, now)
            if len(Tother):
                _cards(Tother, now)
            if len(Th) < N_HIT or len(Tl) < N_LONG:
                st.caption("Fewer than 5 + 5 because not enough games or props have lines yet (props need an ODDS_API_KEY).")
        else:
            st.info("No bets for today yet. They appear here once DraftKings posts lines for games that have not started. "
                    "Yesterday's results are in History below.")

        st.markdown("### History")
        past = sorted([d for d in P.batch_date.unique() if d != today], reverse=True)
        if not past:
            st.caption("Each day's bets move here the next day, with every leg and the final result.")
        for k, d in enumerate(past):
            D = P[P.batch_date == d].sort_values("ev", ascending=False)
            with st.expander(f"{_day_name(d)}  ·  {_day_summary(D)}", expanded=(k == 0)):
                _cards(D, now)

        by_day = done.groupby("batch_date").agg(bets=("parlay_id", "count"), profit=("profit", "sum"))
        if len(by_day) >= 2:
            by_day["Total profit $"] = by_day.profit.cumsum()
            st.markdown("#### Profit over time")
            st.line_chart(by_day[["Total profit $"]], color=["#c4b5fd"])
        with st.expander("All bets in one table (download)"):
            tbl = pd.DataFrame({
                "Day": P.batch_date, "Tier": P.tier, "Bet": [" + ".join(l["label"] for l in json.loads(x)) for x in P.legs],
                "Odds": [fmt_am(dec2am(float(x))) for x in P.dec_odds], "Result": P.status.map(_CHIP).fillna("Not started"),
                "Profit $": P.profit.round(2)}).sort_values("Day", ascending=False)
            _html_table(tbl.assign(**{"Profit $": tbl["Profit $"].map(lambda x: f"{x:+.2f}")}))
            st.download_button("Download as CSV", tbl.to_csv(index=False), "parlay_history.csv", "text/csv")

    if PROPS is not None and P is not None and not P.empty:
        with st.expander("Is the prop model calibrated? (graded prop legs: model chance vs what happened)"):
            cal_ = PROPS.calibration(P)
            if cal_.empty:
                st.caption("No graded prop legs yet. This fills in as games finish. A calibrated model's 'actual' "
                           "column sits close to 'model_p'; if actual is consistently lower, the model is too optimistic.")
            else:
                st.dataframe(cal_, **FULL)
    with st.expander("What the experts say and how much weight each gets"):
        n = parlay_notes()
        if len(n):
            _html_table(n.rename(columns={"home": "Home", "away": "Away", "source": "Source", "pick": "Pick",
                                          "lean_m": "Lean (side)", "lean_t": "Lean (total)", "weight": "Weight",
                                          "reasoning": "Reasoning", "assessment": "Our read"}))
    with st.expander("How the parlays are chosen"):
        st.markdown("""
1. **Fair odds first.** Every book's spread, moneyline and total is de-vigged (the sportsbook's margin removed) and
   converted into one consensus expected margin and total per game, using every book we have plus prediction markets.
2. **Small expert tilt.** Expert reads move that consensus by at most half a point, weighted by how concrete and
   verifiable each expert's reasoning is. Streaks like "19-9" are not used: samples that small are mostly luck, and the
   market already prices what the experts know.
3. **Simulation.** Each game is simulated 60,000 times. Legs in the same game are scored together so correlation counts.
4. **Player props.** For each player stat, every book's Over/Under pair is de-vigged and averaged (3+ books required),
   a count or yardage distribution is fitted to that consensus, and every DraftKings line, main and alternate, is priced
   from it. Each probability is cut by a flat 1.5 points plus half the disagreement between books, capped at 90%, and
   legs whose 'edge' is above +15% are dropped as probable stale lines. Voided bets (player did not play) count as pushes.
5. **10 parlays a day.** One leg per game, so legs are independent and the odds multiply exactly. Five **Hits** are the
   most likely combinations paying 2-4x; five **Longshots** are the most likely paying 5-10x. No leg is used in more
   than two parlays of a kind.
6. **Frozen at kick-off.** Until a game starts the bets can change every 15 minutes. At kick-off the last price becomes the
   bet, and it is graded from the final score. Games that had already started when the bets were built get no bet.
7. **Honest caveat.** Parlays multiply the sportsbook's margin, so expected value is usually negative even for the best
   combination. The tab shows that number instead of hiding it; a long record here is the real test of any edge.
""")


# ----------------------------------------------------------------- sidebar
for k, v in PRESETS["⚖️ Balanced"].items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("use_fund", True)
st.session_state.setdefault("use_news", False)
st.session_state.setdefault("cost_bps", 10)
st.session_state.setdefault("auto", True)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


st.markdown(CSS, unsafe_allow_html=True)
_open0 = _mkt_open(datetime.now(ET))
st.markdown(
    '<div class="hero"><div><div class="brand">Hyperdrift<em>stock ranker</em></div>'
    '<div class="tag">Educational tool, not financial advice. Rankings come from a hand-weighted factor model; '
    'projections are statistical estimates, not predictions or promises.</div></div>'
    '<div>' + ('<span class="pill on"><i></i>Market open</span>' if _open0 else
               '<span class="pill"><i></i>Market closed</span>') + '</div></div>', unsafe_allow_html=True)

# Only the selected section runs (tabs would execute ALL of them on every click = the old lag).
PAGES = ["Rankings", "Live Trading", "Parlays", "Why they ranked", "Portfolio", "Backtest", "Past History", "Methodology"]
if hasattr(st, "segmented_control"):
    _sel = st.segmented_control("Section", PAGES, default=PAGES[0], key="nav", label_visibility="collapsed")
else:
    _sel = st.radio("Section", PAGES, horizontal=True, key="nav", label_visibility="collapsed")
if _sel:
    st.session_state["_page"] = _sel
page = st.session_state.get("_page", PAGES[0])

sb = st.popover("⚙️  Settings")      # weights, filters, presets live here now
sb.header("Quick presets")
sb.button("⚡ Quick gains (today)", on_click=apply_preset, args=("⚡ Quick gains (today)",),
          **FULL, help="Liquid, large caps with rising trend and volume; volatility penalized.")
sb.button("🌱 Long-term (1 year)", on_click=apply_preset, args=("🌱 Long-term (1 year)",),
          **FULL, help="Quality, growth and value with a trend filter.")
sb.button("↺ Reset to default", on_click=apply_preset, args=("⚖️ Balanced",),
          **FULL, help="Restores the default weights, filters and horizon.")

fm = sb.form("controls", border=False)
fm.header("Universe")
choice = fm.radio("Stocks to analyze", ["Quick demo (~50 large caps)", "S&P 500 (slow first run)", "Custom tickers"])
custom = fm.text_input("Custom tickers (comma separated)", "AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM")
use_fund = fm.checkbox("Include fundamentals (Quality / Growth / Value)", key="use_fund")
use_news = fm.checkbox("Include news count (slow)", key="use_news", disabled=not use_fund)

fm.header("Projection horizon")
hlabel = fm.selectbox("Show projected move over", list(HORIZONS), key="horizon")
h = HORIZONS[hlabel]

fm.header("Filters")
min_price = fm.number_input("Min price ($)", 0.0, 1000.0, key="min_price")
min_dv = fm.number_input("Min avg daily $ volume (millions)", 0.0, 10000.0, key="min_dv") * 1e6
min_cap = fm.number_input("Min market cap (billions)", 0.0, 5000.0, key="min_cap") * 1e9

fm.header("Factor weights")
w_raw = {
    "M": fm.slider("Momentum", 0.0, 1.0, step=0.05, key="w_M"),
    "T": fm.slider("Trending / attention", 0.0, 1.0, step=0.05, key="w_T"),
    "Q": fm.slider("Quality", 0.0, 1.0, step=0.05, key="w_Q"),
    "G": fm.slider("Growth", 0.0, 1.0, step=0.05, key="w_G"),
    "V": fm.slider("Value", 0.0, 1.0, step=0.05, key="w_V"),
}
if not use_fund:
    w_raw["Q"] = w_raw["G"] = w_raw["V"] = 0.0
tot = sum(w_raw.values()) or 1.0
w = {k: v / tot for k, v in w_raw.items()}
w["risk"] = fm.slider("Volatility penalty", 0.0, 0.5, step=0.05, key="w_risk")
fm.caption("Positive weights are normalized to sum to 1.")

fm.header("Portfolio")
top_n = fm.slider("Top N stocks", 5, 50, key="top_n")
cap = fm.slider("Max weight per stock (%)", 3, 25, key="cap") / 100
cost_bps = fm.slider("Backtest cost per trade (bps)", 0, 50, key="cost_bps")

fm.form_submit_button("Apply changes", type="primary", **FULL)

sb.header("Updates")
auto = sb.checkbox("Live updates (every 60 s)", key="auto", help="Refreshes only the live parts, not the whole app.")

start_tracker()   # background thread: logs Top 15 each trading day, fills in opens/closes
start_parlay_tracker()   # background thread: re-prices and grades parlays every 15 minutes

# ----------------------------------------------------------------- universe + data
sectors = {}
if choice.startswith("Quick"):
    tickers = DEMO
elif choice.startswith("S&P"):
    sectors = get_sp500()
    tickers = list(sectors)
else:
    tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
if len(tickers) < 10:
    st.warning("Use at least 10 tickers so z-scores are meaningful.")
    st.stop()

with st.spinner("Downloading prices..."):
    close_all, vol_all, open_all = load_prices(tuple(sorted(set(tickers + ["SPY"]))))
if "SPY" not in close_all:
    st.error("Could not download data. Check your internet connection and try again.")
    st.stop()

now = datetime.now(ET)
incomplete = close_all.index[-1].date() == now.date() and now.time() < dtime(16, 10)
live_px = close_all.ffill().iloc[-1]
live_chg = (close_all.iloc[-1] / open_all.iloc[-1] - 1) if incomplete else pd.Series(dtype=float)
market_open = now.weekday() < 5 and dtime(9, 30) <= now.time() < dtime(16, 0)
# Features use COMPLETED bars only, so a rank is always "as of the last close".
C, V, O = (close_all.iloc[:-1], vol_all.iloc[:-1], open_all.iloc[:-1]) if incomplete else \
    (close_all, vol_all, open_all)
spy = C["SPY"].dropna()
cols = [c for c in C.columns if c in tickers]
close, volume, opn = C[cols], V[cols], O[cols]

fund = None
if use_fund:
    with st.spinner("Fetching fundamentals (cached for 24h)..."):
        fund = load_fundamentals(tuple(cols), use_news)

dkey = f"{len(cols)}-{hash(tuple(cols))}-{close.index[-1]}-{use_fund}-{use_news}-{bool(sectors)}"
F, dv, df = rank_cached(close, volume, fund, sectors, dkey,
                        {k: round(v, 4) for k, v in w.items()}, min_price, min_dv, min_cap)
if len(df) < 10:
    st.error(f"Only {len(df)} stocks pass the filters. Loosen them in the sidebar.")
    st.stop()

ckey = f"{len(cols)}-{hash(tuple(cols))}-{close.index[-1]}"
cal = calibrate(close, opn, F, dv, ckey, tuple(round(w[k], 4) for k in WK), h, min_price, min_dv)
df = project(df, cal, h)
df["live"] = live_chg.reindex(df.index) if incomplete else np.nan

top = df.nlargest(top_n, "Score").copy()
top["Weight"] = position_weights(top.vol60, cap)
top["sector"] = top.sector.fillna("Unknown")
name = top["longName"] if "longName" in top else pd.Series("", index=top.index)
floor = {1: .012, 5: .025, 21: .05, 252: .15}[h]
full = max(floor, 0.5 * df.vol60.median() * np.sqrt(h / 252))
dec = 2 if h <= 5 else 1
spy_trend = spy.iloc[-1] / spy.rolling(200).mean().iloc[-1] - 1

# ----------------------------------------------------------------- tabs
_n = datetime.now(ET)       # auto-refresh only the live fragments, and only around market hours
every = 60 if auto and _n.weekday() < 5 and dtime(8, 30) <= _n.time() < dtime(16, 30) else None


@st.fragment(run_every=every)
def rankings_page(base):
    now = datetime.now(ET)
    market_open = _mkt_open(now)
    top = base.copy()
    if market_open:                                  # real-time quotes for the displayed stocks
        lq = load_live(tuple(top.index))
        if len(lq):
            top["price"] = lq.price.reindex(top.index).fillna(top.price)
            top["live"] = (lq.price / lq.open - 1).reindex(top.index)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Horizon", hlabel)
    m2.metric("Stocks passing filters", f"{len(df)} / {len(cols)}")
    m3.metric("Model skill (rank IC)", "n/a" if np.isnan(cal["ic"]) else f"{cal['ic']:+.3f}",
              help="Average rank correlation between score and the stock's actual next-period return. "
                   "0 = no edge. Even good models are around +0.02 to +0.06.")
    m4.metric("Market trend (SPY vs 200d)", f"{spy_trend * 100:+.1f}%")
    status = "🟢 Market open" if market_open else "⚪ Market closed"
    st.caption(f"{status} · updated {now.strftime('%I:%M:%S %p')} ET · ranks as of the close on "
               f"{close.index[-1].date()}" + (" · prices and 'today' moves update every minute." if market_open else "."))
    view = st.radio("View", ["Cards", "Compact list"], horizontal=True, label_visibility="collapsed")
    if spy_trend < 0:
        st.warning("The overall market is below its 200-day average. Historically that means higher risk "
                   "and less reliable long positions.")
    if cal["beta"] == 0:
        st.info("At this horizon the model has shown no measurable edge in its history, so the projected "
                "ranges are basically the market's typical drift and volatility. Treat the ranking as "
                "descriptive, not predictive.")
    if view == "Cards":
        st.markdown("<div class='grid'>" + "".join(
            card_html(i, t, top.loc[t], name.get(t), h, hlabel, full, dec)
            for i, t in enumerate(top.index, 1)) + "</div>", unsafe_allow_html=True)
    else:
        st.markdown(list_html(top, name, hlabel, full, dec), unsafe_allow_html=True)
    with st.expander("Full table"):
        show = pd.DataFrame({
            "Name": name, "Sector": top.sector, "Price": top.price.round(2), "Score": top.Score.round(2),
            f"Low ({hlabel}) %": (top.p_lo * 100).round(2), f"High ({hlabel}) %": (top.p_hi * 100).round(2),
            "Confidence": top.conf.round(2), "Momentum": top.M.round(2), "Trending": top["T"].round(2),
            "Quality": top.Q.round(2), "Growth": top.G.round(2), "Value": top.V.round(2),
            "Vol (60d) %": (top.vol60 * 100).round(1)})
        show.index.name = "Ticker"
        st.dataframe(show, **FULL)
        st.download_button("Download CSV", show.to_csv().encode(), "top_stocks.csv", "text/csv")


if page == "Rankings":
    rankings_page(top)

if page == "Why they ranked":
    st.write("Each bar is a factor's contribution (weight x factor score) to the total. "
             "Negative means it pulled the score down.")
    contrib = top[[f"c_{k}" for k in FACT] + ["c_Risk"]]
    contrib.columns = ["Momentum", "Trending", "Quality", "Growth", "Value", "Risk penalty"]
    st.bar_chart(contrib, color=["#8b5cf6", "#c4b5fd", "#6d28d9", "#e879f9", "#a5b4fc", "#f08aa0"])

if page == "Portfolio":
    st.write("Inverse-volatility weights with a per-stock cap.")
    pf = pd.DataFrame({"Name": name, "Sector": top.sector, "Weight %": (top.Weight * 100).round(2)})
    c1, c2 = st.columns([3, 2])
    c1.dataframe(pf, **FULL)
    sec = (top.groupby("sector").Weight.sum() * 100).round(1).sort_values(ascending=False)
    c2.bar_chart(sec, color="#8b5cf6")
    if sec.max() > 30:
        c2.warning(f"{sec.idxmax()} is {sec.max():.0f}% of the portfolio. Consider a 30% sector cap.")

if page == "Backtest":
    st.write("Monthly rebalance, equal-weight top N, using **price-based factors only** "
             "(momentum, trending, volatility), because free fundamentals data isn't point-in-time.")
    bt = backtest(close, F, dv, w, top_n, min_price, min_dv, cost_bps, spy,
                  score=hs_cached(close, F, dv, dkey, tuple(round(w[k], 4) for k in WK), min_price, min_dv))
    if len(bt) < 6:
        st.info("Not enough history for a backtest.")
    else:
        st.line_chart((1 + bt).cumprod(), color=["#a78bfa", "#64607c"])
        st.dataframe(pd.DataFrame({c: perf(bt[c]) for c in bt}).style
                     .format("{:.2%}", subset=pd.IndexSlice[["CAGR", "Volatility", "Max drawdown"], :])
                     .format("{:.2f}", subset=pd.IndexSlice[["Sharpe (rf=0)"], :]))
        st.warning("Survivorship bias: the universe is today's index members, which flatters results. "
                   "Live results are usually worse than backtests.")

if page == "Past History":
    st.subheader("Past History")
    storage_banner()
    st.caption("Every trading day the app logs its Top 15 (fixed 'quick gains' price model, S&P 500, liquid "
               "stocks only) with a 1-day projection made from the prior close. Then it compares the "
               "prediction to what happened from the **open to the close**. Today's entry starts from the "
               "current estimate because the market has already opened.")
    b1, b2 = st.columns([1, 4])
    if b1.button("Update tracker now"):
        with st.spinner("Updating..."):
            all_steps()
        st.rerun()
    b2.caption((f"Reading history from `{_cfg('DATA_URL_BASE')}`." if _cfg("DATA_URL_BASE") else
                f"History files: `{HIST_PATH}` and `{EXP_PATH}`.") + f" Use a Google Drive path (HISTORY_PATH) so it survives Colab restarts.")
    with LOCK:
        hist = read_hist()
    if hist.empty:
        _err = st.session_state.get("_load_err", {}).get(os.path.basename(HIST_PATH))
        if _cfg("DATA_URL_BASE"):
            if _err:
                st.error(f"Couldn't read the saved history from GitHub. {_err}")
            if st.button("Reload history from GitHub"):
                _csv_cached.clear()
                st.rerun()
        st.info("No history yet. The first entry is logged once the market is open (weekdays after 9:30 AM ET) "
                "and the app is running. Click 'Update tracker now' to check.")
    else:
        pend = ~hist.final
        hist.loc[pend, "close"] = hist.loc[pend, "ticker"].map(live_px).fillna(hist.loc[pend, "close"])
        hist["actual"] = hist.close / hist.open - 1
        daily = hist.groupby("date").agg(pred=("pred_mid", "mean"), actual=("actual", "mean"),
                                         n=("ticker", "count"), final=("final", "all"))
        fin = daily[daily.final]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Completed days", len(fin))
        if len(fin):
            k2.metric("Avg predicted / day", f"{fin.pred.mean() * 100:+.2f}%")
            k3.metric("Avg actual / day", f"{fin.actual.mean() * 100:+.2f}%")
            k4.metric("Direction right", f"{(np.sign(fin.pred) == np.sign(fin.actual)).mean() * 100:.0f}%")
        long = (daily[["pred", "actual"]] * 100).rename(
            columns={"pred": "Predicted (avg of Top 15)", "actual": "Actual (avg of Top 15)"}) \
            .reset_index().melt("date", var_name="Series", value_name="Return %")
        st.markdown("#### Top 15 Stocks: predicted vs actual (open → close)")
        st.altair_chart(alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("date:T", title="Trading day"), y=alt.Y("Return %:Q", title="Avg move, %"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Predicted (avg of Top 15)", "Actual (avg of Top 15)"],
                                                         range=["#7c3aed", "#c4b5fd"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["date:T", "Series", alt.Tooltip("Return %:Q", format=".2f")]).properties(height=340),
            **FULL)
        st.markdown("#### Latest picks")
        latest = hist[hist.date == hist.date.max()].sort_values("rank")
        st.dataframe(pd.DataFrame({
            "Rank": latest["rank"], "Ticker": latest.ticker, "Open": latest.open.round(2),
            "Now / Close": latest.close.round(2), "Predicted %": (latest.pred_mid * 100).round(2),
            "Range %": [f"{a * 100:+.2f} to {b * 100:+.2f}" for a, b in zip(latest.pred_low, latest.pred_high)],
            "Actual %": (latest.actual * 100).round(2),
            "Status": np.where(latest.final, "final", "in progress")}).set_index("Rank"),
            **FULL)

    with LOCK:
        exp_hist = read_exp()
    try:
        render_comparison(hist, exp_hist, live_px)
    except Exception as e:
        st.info(f"The experts-vs-ours comparison appears once picks are logged ({type(e).__name__}).")

if page == "Methodology":
    st.markdown("""
**Score = wM·M + wT·T + wQ·Q + wG·G + wV·V − wRisk·z(σ60)**

Every metric is winsorized (2nd to 98th percentile), z-scored across the eligible universe, and clipped to ±3.
Lower-is-better metrics are flipped. A missing metric is dropped from its factor's average (weights renormalized).

| Factor | Components |
|---|---|
| Momentum | 12-1m return, 6m return, price/SMA200, price/SMA50, risk-adjusted momentum (12-1m ÷ volatility) |
| Trending | volume surge (10d/60d), price/52w high, range expansion (5d vs 60d moves), news count (optional) |
| Quality | ROE, gross margin, profit margin, −debt/equity (sector-relative blend) |
| Growth | revenue growth, earnings growth, analyst target upside (3+ analysts) |
| Value | FCF yield, EBITDA/EV, −P/E (sector-relative blend) |
| Risk | −z(60-day annualized volatility) |

**What makes the rating and projection more reliable**
- Outlier-robust z-scores and sector-relative fundamentals, so a bank isn't judged against a software company.
- Risk-adjusted momentum and a volatility penalty; presets also raise liquidity and market-cap floors.
- Projections are **calibrated** on the model's own point-in-time history (score → next-period return), then
  **shrunk**: slope cut by 40%, and the base drift blended 50/50 with an 8%/yr long-run assumption to reduce
  survivorship and bull-market bias. If the model showed no edge (rank IC ≤ 0), the slope is set to zero.
- The range is the middle 50% of historical outcomes, scaled by each stock's own volatility.
- Ranks use completed daily bars only, so there is no peeking at an unfinished day.
- Each stock shows a confidence badge from data coverage and how many factors agree.
- The Past History tab is an out-of-sample scoreboard: predictions are logged before the result is known.

**Expert comparison (Past History tab):** the benchmark is the 10 research firms TipRanks ranked best over the past
decade (RBC Capital, Jefferies, Truist, UBS, KBW, Raymond James, Stifel, Wolfe Research, Deutsche Bank; Credit Suisse was
absorbed by UBS, so Needham takes its place). TipRanks' per-analyst data is paywalled, so each firm's daily picks are its 15 most
recent Buy-type ratings from public rating feeds (via yfinance), using only ratings dated before that day. Accuracy is then
measured by us, the same way as ours: hit rate and average return from open to close. Starts 2026-10-20.

**Live Trading tab (separate model):** each day it picks the 10 most tradeable stocks from data before the open (liquid,
wide daily range, directional days, unusual volume). Then every minute: BUY needs 4 of 5 conditions for 2 minutes in a row
(above the first-15-minute high, above VWAP, 5- and 15-minute momentum up, volume picking up, SPY not falling); it sells on a
stop-loss, a trailing stop, a trend break, or by 3:55 PM. Each stock gets its own $100 per day, trades happen at the next
minute's price with a 5 bps cost per side, and nothing is held overnight. News counts are shown but not used in the rules.

**Limits you should know about:** yfinance is unofficial and fundamentals can be stale or missing (and are not
point-in-time, so they're excluded from calibration and backtests). Calibration is price-only and applied to the full
score. One-day moves are mostly noise, so the "Today" horizon should show small expected moves and wide ranges. That is
the honest answer, not a bug. Earnings dates, news and macro events are not modeled. For anything serious, use a paid
point-in-time data source.
""")

# Live Trading refreshes on its own (fragment) so the rest of the app never re-runs.
@st.fragment(run_every=every)
def live_page():
    try:
        render_live()
    except Exception as e:
        st.error(f"Live Trading could not load yet ({type(e).__name__}: {e}). Try again in a minute.")


if page == "Live Trading":
    live_page()


@st.fragment(run_every=every)
def parlays_page():
    try:
        render_parlays()
    except Exception as e:
        st.error(f"Parlays could not load yet ({type(e).__name__}: {e}). Try again in a minute.")


if page == "Parlays":
    parlays_page()
