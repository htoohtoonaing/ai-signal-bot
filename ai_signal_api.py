# ================================
# AI Signal Backend – v2.8 (Wait-Fix Edition)
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

# 🔑 API KEYS
TD_KEY = "f5acaa85b3824672b23789adafdb85e9"
FINN_KEY = "d48lfb1r01qnpsnocvngd48lfb1r01qnpsnocvo0"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# MODELS
# --------------------------
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


# --------------------------
# HELPERS
# --------------------------
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


# --------------------------
# FETCH CANDLES
# --------------------------
def fetch_candles(pair, tf, bars=200):
    # ---------- TWELVEDATA ----------
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
            df = df.iloc[::-1].reset_index()
            if len(df) > 30:
                return df
    except:
        pass

    # ---------- FINNHUB ----------
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
                "close": r["c"]
            })
            df = df.dropna().sort_values("time")
            return df
    except:
        pass

    raise RuntimeError("No candle data from TD or Finnhub")


# --------------------------
# AI LOGIC (v2.8 NO MORE WAIT)
# --------------------------
def generate_ai_signal(df):

    df = df.copy()
    close = to_1d(df["close"])

    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close)
    macd_line, macd_sig, _ = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig

    last = df.iloc[-1]

    # --- Trend (Soft Mode) ---
    if last["ema20"] > last["ema50"]:
        trend = "up"
    elif last["ema20"] < last["ema50"]:
        trend = "down"
    else:
        trend = "flat"

    rsi_val = float(last["rsi"])
    macd_bull = last["macd"] > last["macd_sig"]
    macd_bear = last["macd"] < last["macd_sig"]

    # --- Signal Logic (Boosted) ---
    side = "WAIT"
    confidence = 60
    reason = []

    if trend == "up" and rsi_val > 48 and macd_bull:
        side = "BUY"
        confidence = 85
        reason.append("Uptrend + RSI>48 + MACD Bullish")
    elif trend == "down" and rsi_val < 52 and macd_bear:
        side = "SELL"
        confidence = 85
        reason.append("Downtrend + RSI<52 + MACD Bearish")
    else:
        side = "WAIT"
        confidence = 55
        reason.append("Indecision / Small candles")

    price = float(last["close"])

    return {
        "side": side,
        "confidence": confidence,
        "trend": trend,
        "price": price,
        "rsi": rsi_val,
        "sl": None,
        "tp": None,
        "reason": " ".join(reason),
    }


# --------------------------
# ENDPOINT
# --------------------------
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
