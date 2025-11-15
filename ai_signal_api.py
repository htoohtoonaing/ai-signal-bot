# ================================
# AI Signal Backend – v3.0 (All-in-One Edition)
# Naing Kyaw • TwelveData + Finnhub • No Yahoo
# ================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import time
import requests

# ======================================================
# 🔑 API KEYS — (Ready to use)
# ======================================================
TD_KEY = "f5acaa85b3824672b23789adafdb85e9"
FINN_KEY = "d48lfb1r01qnpsnocvngd48lfb1r01qnpsnocvo0"

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# MODELS
# ======================================================
class SignalRequest(BaseModel):
    pair: str
    timeframe: str

class SignalResponse(BaseModel):
    pair: str
    timeframe: str
    side: str
    confidence: int
    trend: str
    price: float
    rsi: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: str


# ======================================================
# HELPERS
# ======================================================
def normalize_pair_for_td(pair):
    return pair.upper().strip()

def normalize_pair_for_finnhub(pair):
    return "OANDA:" + pair.upper().replace("/", "_")

TF_MAP_TD = {"5 sec": "1min", "30 sec": "1min", "1 min": "1min"}
TF_MAP_FH = {"5 sec": "1", "30 sec": "1", "1 min": "1"}

def to_1d(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def ema(s, p):
    s = to_1d(s)
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, period=14):
    s = to_1d(s)
    delta = s.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(s):
    s = to_1d(s)
    fast = ema(s, 12)
    slow = ema(s, 26)
    macd_line = fast - slow
    sig = ema(macd_line, 9)
    return macd_line, sig, macd_line - sig

def atr(df):
    h = to_1d(df["high"])
    l = to_1d(df["low"])
    c = to_1d(df["close"])
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


# ======================================================
# CANDLE FETCH (TwelveData → Finnhub Fallback)
# ======================================================
def fetch_candles(pair, tf, bars=200):

    # ----- TwelveData -----
    try:
        url = (
            "https://api.twelvedata.com/time_series?"
            f"symbol={normalize_pair_for_td(pair)}"
            f"&interval={TF_MAP_TD.get(tf, '1min')}"
            f"&apikey={TD_KEY}&outputsize={bars}"
        )
        r = requests.get(url, timeout=10).json()

        if "values" in r:
            df = pd.DataFrame(r["values"]).rename(columns={"datetime": "time"})
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c])
            df = df.iloc[::-1].reset_index(drop=True)
            if len(df) > 30:
                return df
    except Exception:
        pass

    # ----- Finnhub -----
    try:
        now_ts = int(time.time())
        from_ts = now_ts - (bars * 60)
        url = (
            "https://finnhub.io/api/v1/forex/candle?"
            f"symbol={normalize_pair_for_finnhub(pair)}"
            f"&resolution={TF_MAP_FH.get(tf, '1')}"
            f"&from={from_ts}&to={now_ts}&token={FINN_KEY}"
        )
        r = requests.get(url, timeout=10).json()
        if r.get("s") == "ok":
            df = pd.DataFrame({
                "time": r["t"],
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
            })
            df = df.dropna().sort_values("time")
            return df
    except Exception:
        pass

    raise RuntimeError("No candle data from TD or Finnhub")


# ======================================================
# AI LOGIC (v3.0 ALL-IN-ONE)
# ======================================================
def generate_ai_signal(df):

    df = df.copy()
    close = to_1d(df["close"])

    # Indicators
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = rsi(close)
    macd_line, macd_sig, _ = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig

    last = df.iloc[-1]

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi_val = float(last["rsi"])

    macd_bull = last["macd"] > last["macd_sig"]
    macd_bear = last["macd"] < last["macd_sig"]

    # Trend
    if ema20 > ema50 > ema200:
        trend = "strong_up"
    elif ema20 < ema50 < ema200:
        trend = "strong_down"
    elif ema20 > ema50:
        trend = "up"
    elif ema20 < ema50:
        trend = "down"
    else:
        trend = "flat"

    price = float(last["close"])

    # ---------------------------------
    # ALL-IN-ONE BUY/SELL DECIDER
    # ---------------------------------

    side = "WAIT"
    confidence = 60
    reasons = []

    # 1) Strong Reversal Mode
    if trend in ["strong_down", "down"] and rsi_val < 30 and macd_bull:
        side = "BUY"
        confidence = 90
        reasons.append("Strong oversold reversal (RSI<30 + MACD bull).")

    elif trend in ["strong_up", "up"] and rsi_val > 70 and macd_bear:
        side = "SELL"
        confidence = 90
        reasons.append("Strong overbought reversal (RSI>70 + MACD bear).")

    # 2) Trend-Follow Mode
    elif trend in ["strong_up", "up"] and rsi_val > 50 and macd_bull:
        side = "BUY"
        confidence = 82
        reasons.append("Uptrend trend-follow (RSI>50 + MACD bull).")

    elif trend in ["strong_down", "down"] and rsi_val < 50 and macd_bear:
        side = "SELL"
        confidence = 82
        reasons.append("Downtrend trend-follow (RSI<50 + MACD bear).")

    # 3) Sideway / Flat Mode
    elif trend == "flat":
        if rsi_val > 55 and macd_bull:
            side = "BUY"
            confidence = 75
            reasons.append("Flat but bullish bias.")
        elif rsi_val < 45 and macd_bear:
            side = "SELL"
            confidence = 75
            reasons.append("Flat but bearish bias.")
        else:
            side = "WAIT"
            confidence = 55
            reasons.append("Flat neutral → WAIT.")

    # Safety mid-zone
    if 45 < rsi_val < 55 and side != "WAIT":
        confidence -= 7
        reasons.append("RSI mid-zone, reducing confidence.")

    if side == "WAIT" and not reasons:
        reasons.append("Mixed signals → WAIT.")

    return {
        "side": side,
        "confidence": int(confidence),
        "trend": trend,
        "price": price,
        "rsi": rsi_val,
        "sl": None,
        "tp": None,
        "reason": " ".join(reasons),
    }


# ======================================================
# ENDPOINT
# ======================================================
@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):

    df = fetch_candles(req.pair, req.timeframe, 200)
    sig = generate_ai_signal(df)

    return SignalResponse(
        pair=req.pair,
        timeframe=req.timeframe,
        side=sig["side"],
        confidence=sig["confidence"],
        trend=sig["trend"],
        price=sig["price"],
        rsi=sig["rsi"],
        sl=sig["sl"],
        tp=sig["tp"],
        reason=sig["reason"],
    )
