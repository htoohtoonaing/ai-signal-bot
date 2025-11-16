# =====================================
# AI Signal Backend – v4.0 Balanced + Reverse Pair Fix
# Naing Kyaw • TwelveData + Finnhub • No Yahoo
# BUY/SELL Balanced + EURCHF / CHFEUR Correct
# =====================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import time
import requests

# 🔑 API KEYS (မင်းကိုယ်ပိုင် key တွေ)
TD_KEY = "f5acaa85b3824672b23789adafdb85e9"
FINN_KEY = "d48lfb1r01qnpsnocvngd48lfb1r01qnpsnocvo0"

# ===========================
# FastAPI App
# ===========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Web / Mini App / Telegram WebApp အကုန်ခွင့်ပြု
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Models
# ===========================
class SignalRequest(BaseModel):
    pair: str        # e.g. "EUR/CHF"
    timeframe: str   # "5 sec" / "30 sec" / "1 min"

class SignalResponse(BaseModel):
    pair: str
    timeframe: str
    side: str        # BUY / SELL / WAIT
    confidence: int
    trend: str
    price: float
    rsi: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason: str

# ===========================
# Helper Functions
# ===========================
def normalize_pair_for_td(pair: str) -> str:
    """
    TwelveData → symbol = "EUR/CHF"
    """
    return pair.upper().strip()

def normalize_pair_for_finnhub(pair: str) -> str:
    """
    Finnhub → "OANDA:EUR_CHF"
    """
    return "OANDA:" + pair.upper().replace("/", "_")

TF_MAP_TD = {
    "5 sec": "1min",
    "30 sec": "1min",
    "1 min": "1min",
}
TF_MAP_FH = {
    "5 sec": "1",
    "30 sec": "1",
    "1 min": "1",
}

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
    tr = pd.concat([
        (h - l),
        (h - prev).abs(),
        (l - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ===========================
# Auto Reverse Pair (EUR/CHF vs CHF/EUR)
# ===========================
def auto_reverse_if_needed(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    ဥပမာ:
      - EUR/CHF → normal
      - CHF/EUR → auto invert candles
    """
    try:
        base, quote = pair.upper().split("/")
    except ValueError:
        return df

    major_bases = {"EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"}

    # base ကို major list မှာမတွေ့ / quote ကို major မှာတွေ့ → reverseလိုတယ်လို့ယူ
    if base not in major_bases and quote in major_bases:
        df = df.copy()
        # numeric convert
        o = pd.to_numeric(df["open"], errors="coerce")
        h = pd.to_numeric(df["high"], errors="coerce")
        l = pd.to_numeric(df["low"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")

        df["open"] = 1.0 / o
        df["high"] = 1.0 / l
        df["low"]  = 1.0 / h
        df["close"] = 1.0 / c

        df = df.dropna()
        print(f"[Reverse] Applied auto-reverse for pair {pair}")
    return df

# ===========================
# Fetch Candles (TwelveData + Finnhub)
# ===========================
def fetch_candles(pair: str, timeframe: str, bars: int = 200) -> pd.DataFrame:
    # ---- 1) TwelveData primary ----
    try:
        url = (
            "https://api.twelvedata.com/time_series?"
            f"symbol={normalize_pair_for_td(pair)}"
            f"&interval={TF_MAP_TD.get(timeframe, '1min')}"
            f"&outputsize={bars}"
            f"&apikey={TD_KEY}"
        )
        print("TwelveData URL:", url)
        r = requests.get(url, timeout=10).json()

        if "values" in r:
            df = pd.DataFrame(r["values"]).rename(columns={"datetime": "time"})
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.iloc[::-1].dropna().reset_index(drop=True)
            if len(df) > 40:
                print("TwelveData OK")
                return df
            else:
                print("TwelveData: too few bars")
        else:
            print("TwelveData error:", r)
    except Exception as e:
        print("TwelveData exception:", e)

    # ---- 2) Finnhub fallback ----
    try:
        now_ts = int(time.time())
        from_ts = now_ts - (bars * 60)  # 1min candles

        url = (
            "https://finnhub.io/api/v1/forex/candle?"
            f"symbol={normalize_pair_for_finnhub(pair)}"
            f"&resolution={TF_MAP_FH.get(timeframe, '1')}"
            f"&from={from_ts}&to={now_ts}"
            f"&token={FINN_KEY}"
        )
        print("Finnhub URL:", url)
        r = requests.get(url, timeout=10).json()

        if r.get("s") == "ok":
            df = pd.DataFrame({
                "time": r["t"],
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
            })
            df = df.dropna().sort_values("time").reset_index(drop=True)
            print("Finnhub OK")
            return df
        else:
            print("Finnhub error:", r)
    except Exception as e:
        print("Finnhub exception:", e)

    raise RuntimeError(f"No candle data from TwelveData or Finnhub for {pair}")

# ===========================
# AI LOGIC v4.0 (Balanced BUY/SELL)
# ===========================
def generate_ai_signal(df: pd.DataFrame):
    df = df.copy()
    close = to_1d(df["close"])

    # Indicators
    df["ema10"] = ema(close, 10)
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close, 14)

    macd_line, macd_sig, _ = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig

    last = df.iloc[-1]

    ema10 = float(last["ema10"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    rsi_val = float(last["rsi"])
    macd_bull = last["macd"] > last["macd_sig"]
    macd_bear = last["macd"] < last["macd_sig"]

    # ----- Trend -----
    if ema10 > ema20 > ema50:
        trend = "up"
    elif ema10 < ema20 < ema50:
        trend = "down"
    else:
        trend = "flat"

    # ----- Candle Strength -----
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]
    strong = (body / range_) > 0.4 if range_ != 0 else False

    # ----- Decision -----
    side = "WAIT"
    confidence = 50
    reason = []

    # BUY ZONE
    if trend == "up" and rsi_val >= 48 and macd_bull:
        side = "BUY"
        confidence = 75 + (10 if strong else 0)
        reason.append("Uptrend + RSI>=48 + MACD Bullish")

    # SELL ZONE
    elif trend == "down" and rsi_val <= 52 and macd_bear:
        side = "SELL"
        confidence = 75 + (10 if strong else 0)
        reason.append("Downtrend + RSI<=52 + MACD Bearish")

    # Mixed / Flat → WAIT
    else:
        side = "WAIT"
        confidence = 55
        reason.append("Market mixed/flat → avoid fake signals.")

    return {
        "side": side,
        "confidence": confidence,
        "trend": trend,
        "price": float(last["close"]),
        "rsi": rsi_val,
        "sl": None,
        "tp": None,
        "reason": " | ".join(reason),
    }

# ===========================
# Endpoint
# ===========================
@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):
    # 1) candles fetch
    df = fetch_candles(req.pair, req.timeframe, bars=200)
    # 2) reverse pair fix (CHF/EUR အကြောင်းပြန်)
    df = auto_reverse_if_needed(req.pair, df)
    # 3) AI logic
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
