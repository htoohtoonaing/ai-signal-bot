# =====================================
# AI Signal Backend – v3.0 Balanced Edition
# Naing Kyaw • TwelveData + Finnhub • BUY/SELL Auto-Balance Fix
# =====================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import time
import requests


# 🔑 API KEYS
TD_KEY = "f5acaa85b3824672b23789adafdb85e9"
FINN_KEY = "d48lfb1r01qnpsnocvngd48lfb1r01qnpsnocvo0"


# -------------------------
# FASTAPI INIT
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# MODELS
# -------------------------
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
    sl: Optional[float]
    tp: Optional[float]
    reason: str


# -------------------------
# HELPERS
# -------------------------
def normalize_pair_for_td(pair: str):
    return pair.upper().strip()


def normalize_pair_for_finnhub(pair: str):
    return "OANDA:" + pair.upper().replace("/", "_")


TF_MAP_TD = {"5 sec": "1min", "30 sec": "1min", "1 min": "1min"}
TF_MAP_FH = {"5 sec": "1", "30 sec": "1", "1 min": "1"}


def to_1d(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def ema(series, period):
    s = to_1d(series)
    return s.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    s = to_1d(series)
    delta = s.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series):
    s = to_1d(series)
    fast = ema(s, 12)
    slow = ema(s, 26)
    macd_line = fast - slow
    sig = ema(macd_line, 9)
    hist = macd_line - sig
    return macd_line, sig, hist


def atr(df, period=14):
    h = to_1d(df["high"])
    l = to_1d(df["low"])
    c = to_1d(df["close"])
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# -------------------------
# CANDLE FETCH
# -------------------------
def fetch_candles(pair, timeframe, bars=200):
    # ---- TwelveData ----
    try:
        url = (
            "https://api.twelvedata.com/time_series?"
            f"symbol={normalize_pair_for_td(pair)}"
            f"&interval={TF_MAP_TD.get(timeframe, '1min')}"
            f"&outputsize={bars}"
            f"&apikey={TD_KEY}"
        )
        r = requests.get(url, timeout=10).json()

        if "values" in r:
            df = pd.DataFrame(r["values"]).rename(columns={"datetime": "time"})
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c])
            df = df.iloc[::-1].dropna()
            if len(df) > 40:
                return df
    except Exception as e:
        print("TwelveData error:", e)

    # ---- Finnhub Fallback ----
    try:
        now_ts = int(time.time())
        from_ts = now_ts - (bars * 60)

        url = (
            "https://finnhub.io/api/v1/forex/candle?"
            f"symbol={normalize_pair_for_finnhub(pair)}"
            f"&resolution={TF_MAP_FH.get(timeframe, '1')}"
            f"&from={from_ts}&to={now_ts}"
            f"&token={FINN_KEY}"
        )
        r = requests.get(url, timeout=10).json()

        if r.get("s") == "ok":
            df = pd.DataFrame({
                "time": r["t"],
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"]
            })
            df = df.dropna().sort_values("time")
            return df
    except Exception as e:
        print("Finnhub error:", e)

    raise RuntimeError("No candle data available.")


# -------------------------
# AI ENGINE v3.0 (Balanced)
# -------------------------
def generate_ai_signal(df):
    df = df.copy()
    close = to_1d(df["close"])

    df["ema9"] = ema(close, 9)
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close)
    macd_line, macd_sig, hist = macd(close)

    last = df.iloc[-1]

    # TREND
    if last["ema9"] > last["ema20"] > last["ema50"]:
        trend = "strong_up"
    elif last["ema9"] < last["ema20"] < last["ema50"]:
        trend = "strong_down"
    else:
        trend = "sideways"

    rsi_val = float(last["rsi"])
    macd_bull = macd_line.iloc[-1] > macd_sig.iloc[-1]
    macd_bear = macd_line.iloc[-1] < macd_sig.iloc[-1]

    # -------------------------------
    # Auto-Balancing RSI Mode
    # -------------------------------
    if 45 <= rsi_val <= 55:
        rsi_mode = "neutral"
    elif rsi_val < 45:
        rsi_mode = "oversold"
    else:
        rsi_mode = "overbought"

    side = "WAIT"
    confidence = 60
    reasons = []

    # BUY ZONE
    if (rsi_mode == "neutral" and macd_bull) or \
       (rsi_mode == "oversold" and macd_bull) or \
       (trend == "strong_up" and macd_bull and rsi_val > 48):
        side = "BUY"
        confidence = 82
        reasons.append("BUY: Balanced RSI + MACD + Trend")

    # SELL ZONE
    elif (rsi_mode == "neutral" and macd_bear) or \
         (rsi_mode == "overbought" and macd_bear) or \
         (trend == "strong_down" and macd_bear and rsi_val < 52):
        side = "SELL"
        confidence = 82
        reasons.append("SELL: Balanced RSI + MACD + Trend")

    else:
        side = "WAIT"
        confidence = 55
        reasons.append("WAIT: Weak trend / mixed indicators")

    price = float(last["close"])

    return {
        "side": side,
        "confidence": confidence,
        "trend": trend,
        "price": price,
        "rsi": rsi_val,
        "sl": None,
        "tp": None,
        "reason": " ".join(reasons),
    }


# -------------------------
# ENDPOINT
# -------------------------
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
