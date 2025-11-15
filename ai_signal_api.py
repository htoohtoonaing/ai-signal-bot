# ================================
# AI Signal Backend (FastAPI)
# Naing Kyaw – TwelveData + Finnhub Version
# NO YAHOO, NO BLOCK
# ================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import time
import requests

# 🔑 API KEYS (သင့်ကိုယ်ပိုင် key ထည့်ပါ)
TD_KEY = "f5acaa85b3824672b23789adafdb85e9"   # e.g. "sk_..............."
FINN_KEY = "d48lfb1r01qnpsnocvngd48lfb1r01qnpsnocvo0"    # e.g. "d48l..............."


app = FastAPI()

# CORS – Mini App / Browser ကနေ ခေါ်လို့ရအောင်
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------
# Request + Response Models
# -----------------------------------------------------

class SignalRequest(BaseModel):
    pair: str        # "EUR/USD"
    timeframe: str   # "5 sec" / "30 sec" / "1 min"


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


# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def normalize_pair_for_td(pair: str) -> str:
    """
    TwelveData symbol → "EUR/USD"
    """
    return pair.upper().strip()


def normalize_pair_for_finnhub(pair: str) -> str:
    """
    Finnhub forex symbol → "OANDA:EUR_USD"
    """
    p = pair.upper().replace("/", "_").strip()
    return f"OANDA:{p}"


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

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    s = to_1d(series)
    fast_ema = ema(s, fast)
    slow_ema = ema(s, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def atr(df, period=14):
    h = to_1d(df["high"])
    l = to_1d(df["low"])
    c = to_1d(df["close"])
    prev = c.shift(1)

    tr = pd.concat([
        h - l,
        (h - prev).abs(),
        (l - prev).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# -----------------------------------------------------
# Candle Fetch – TwelveData (Primary) + Finnhub (Fallback)
# -----------------------------------------------------

def fetch_candles(pair: str, timeframe_label: str, bars: int = 200) -> pd.DataFrame:
    """
    1) Try TwelveData
    2) If fail → try Finnhub
    3) If both fail → RuntimeError
    """
    # ---------- Try TwelveData ----------
    if TD_KEY and TD_KEY != "YOUR_TWELVEDATA_KEY":
        try:
            symbol_td = normalize_pair_for_td(pair)
            interval_td = TF_MAP_TD.get(timeframe_label, "1min")

            url = (
                "https://api.twelvedata.com/time_series"
                f"?symbol={symbol_td}"
                f"&interval={interval_td}"
                f"&outputsize={bars}"
                f"&apikey={TD_KEY}"
            )
            print("TwelveData URL:", url)

            r = requests.get(url, timeout=10)
            data = r.json()

            if "values" in data:
                values = data["values"]
                df = pd.DataFrame(values)

                # TwelveData returns newest first → reverse
                df = df.rename(columns={
                    "datetime": "time",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })

                # Some responses have no volume
                if "volume" not in df:
                    df["volume"] = 0.0

                df["open"] = pd.to_numeric(df["open"], errors="coerce")
                df["high"] = pd.to_numeric(df["high"], errors="coerce")
                df["low"] = pd.to_numeric(df["low"], errors="coerce")
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

                df = df.dropna().iloc[::-1].reset_index(drop=True)
                if len(df) >= 50:
                    return df.tail(bars)
                else:
                    print("TwelveData returned too few bars → try Finnhub")

            else:
                print("TwelveData error:", data)

        except Exception as e:
            print("TwelveData exception:", e)

    # ---------- Fallback → Finnhub ----------
    if FINN_KEY and FINN_KEY != "YOUR_FINNHUB_KEY":
        try:
            symbol_fh = normalize_pair_for_finnhub(pair)
            resolution = TF_MAP_FH.get(timeframe_label, "1")
            now_ts = int(time.time())
            from_ts = now_ts - bars * 60  # 1min bars

            url = (
                "https://finnhub.io/api/v1/forex/candle"
                f"?symbol={symbol_fh}"
                f"&resolution={resolution}"
                f"&from={from_ts}"
                f"&to={now_ts}"
                f"&token={FINN_KEY}"
            )
            print("Finnhub URL:", url)

            r = requests.get(url, timeout=10)
            data = r.json()

            if data.get("s") == "ok":
                df = pd.DataFrame({
                    "time": data["t"],
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                })
                df["volume"] = 0.0
                df = df.dropna().sort_values("time").reset_index(drop=True)
                if len(df) >= 50:
                    return df.tail(bars)
                else:
                    print("Finnhub returned too few bars.")
            else:
                print("Finnhub error:", data)

        except Exception as e:
            print("Finnhub exception:", e)

    raise RuntimeError(f"No candle data from TwelveData or Finnhub for {pair}")


# -----------------------------------------------------
# AI Signal Logic
# -----------------------------------------------------

def generate_ai_signal(df: pd.DataFrame):
    df = df.copy()
    close = to_1d(df["close"])

    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)
    df["rsi"] = rsi(close)
    df["atr"] = atr(df)

    macd_line, macd_sig, hist = macd(close)
    df["macd"] = macd_line
    df["macd_sig"] = macd_sig

    last = df.iloc[-1]

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])

    # Trend
    if ema20 > ema50 and ema50 > ema200:
        trend = "up"
    elif ema20 < ema50 and ema50 < ema200:
        trend = "down"
    else:
        trend = "flat"

    rsi_val = float(last["rsi"])
    macd_bull = last["macd"] > last["macd_sig"]
    macd_bear = last["macd"] < last["macd_sig"]

    body = abs(last["close"] - last["open"])
    rng = last["high"] - last["low"]
    strong_candle = (body / rng) > 0.6 if rng != 0 else False

    side = "WAIT"
    confidence = 58
    reasons = []

    if trend == "up" and rsi_val > 45 and macd_bull and strong_candle:
        side = "BUY"
        confidence = 82
        reasons.append("Uptrend + RSI>45 + MACD bull + strong bullish candle.")
    elif trend == "down" and rsi_val < 55 and macd_bear and strong_candle:
        side = "SELL"
        confidence = 82
        reasons.append("Downtrend + RSI<55 + MACD bear + strong bearish candle.")
    else:
        reasons.append("Conditions mixed → WAIT zone.")

    reasons.append(f"RSI≈{rsi_val:.1f}, trend={trend}.")

    price = float(last["close"])
    atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else None

    sl = tp = None
    if atr_val and side != "WAIT":
        if side == "BUY":
            sl = price - 1.5 * atr_val
            tp = price + 2.5 * atr_val
        elif side == "SELL":
            sl = price + 1.5 * atr_val
            tp = price - 2.5 * atr_val

    return {
        "side": side,
        "confidence": int(confidence),
        "trend": trend,
        "price": price,
        "rsi": rsi_val,
        "sl": sl,
        "tp": tp,
        "reason": " ".join(reasons),
    }


# -----------------------------------------------------
# /signal Endpoint
# -----------------------------------------------------

@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):
    pair = req.pair
    tf = req.timeframe

    df = fetch_candles(pair, tf, bars=200)
    sig = generate_ai_signal(df)

    return SignalResponse(
        pair=pair,
        timeframe=tf,
        side=sig["side"],
        confidence=sig["confidence"],
        trend=sig["trend"],
        price=sig["price"],
        rsi=sig["rsi"],
        sl=sig["sl"],
        tp=sig["tp"],
        reason=sig["reason"],
    )