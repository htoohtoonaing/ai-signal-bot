# ===========================
# AI LOGIC v4.0 (Balanced BUY/SELL) - FIXED
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

    # ----- Decision - BALANCED FIX -----
    side = "WAIT"
    confidence = 50
    reason = []

    # BUY ZONE - FIXED CONDITIONS
    if trend == "up" and rsi_val >= 45 and rsi_val <= 70 and macd_bull:
        side = "BUY"
        confidence = 70 + (10 if strong else 0)
        reason.append("Uptrend + RSI(45-70) + MACD Bullish")

    # SELL ZONE - FIXED CONDITIONS  
    elif trend == "down" and rsi_val <= 55 and rsi_val >= 30 and macd_bear:
        side = "SELL"
        confidence = 70 + (10 if strong else 0)
        reason.append("Downtrend + RSI(30-55) + MACD Bearish")

    # NEUTRAL ZONE - BALANCED WAIT
    else:
        side = "WAIT"
        confidence = 60
        
        if trend == "flat":
            reason.append("Market flat - wait for clear trend")
        elif rsi_val > 70:
            reason.append("RSI overbought - avoid BUY")
        elif rsi_val < 30:
            reason.append("RSI oversold - avoid SELL") 
        elif not (macd_bull or macd_bear):
            reason.append("MACD neutral - wait for momentum")
        else:
            reason.append("Mixed signals - wait for confirmation")

    # Add SL/TP calculation
    current_price = float(last["close"])
    atr_val = atr(df.tail(14)).iloc[-1] if len(df) >= 14 else current_price * 0.002
    
    if side == "BUY":
        sl = current_price - (atr_val * 1.5)
        tp = current_price + (atr_val * 2.0)
    elif side == "SELL":
        sl = current_price + (atr_val * 1.5) 
        tp = current_price - (atr_val * 2.0)
    else:
        sl = tp = None

    return {
        "side": side,
        "confidence": min(confidence, 95),  # Cap at 95%
        "trend": trend,
        "price": current_price,
        "rsi": rsi_val,
        "sl": round(sl, 5) if sl else None,
        "tp": round(tp, 5) if tp else None,
        "reason": " | ".join(reason),
    }
