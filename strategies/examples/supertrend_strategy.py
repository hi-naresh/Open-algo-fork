#!/usr/bin/env python
"""
Supertrend Strategy — Live Trading Template
Upload via OpenAlgo → Python Strategies (/python)

Configure the parameters below (or pass them as environment variables)
before uploading. Use supertrend_optimization.py to find the optimal
ATR_PERIOD and ATR_MULTIPLIER for your symbol first, then set them here.

Prerequisites
  pip install openalgo pandas numpy

Environment variables accepted (all optional — defaults shown):
  OPENALGO_APIKEY     API key from the OpenAlgo portal
  OPENALGO_HOST       Server URL  (default: http://127.0.0.1:5000)
  SYMBOL              Trading symbol  (default: RELIANCE)
  EXCHANGE            Exchange  (default: NSE)
  PRODUCT             Product type  (default: MIS)
  QUANTITY            Lot size  (default: 1)
  ATR_PERIOD          Supertrend ATR period  (default: 7)
  ATR_MULTIPLIER      Supertrend ATR multiplier  (default: 3.0)
  INTERVAL            Historical interval for signal calculation  (default: 15m)
  LOOKBACK_DAYS       Days of history to fetch per tick  (default: 5)
  SLEEP_SECONDS       Seconds between signal checks  (default: 60)
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openalgo import api

# ─────────────────── CONFIGURATION ────────────────────────
API_KEY    = os.getenv("OPENALGO_APIKEY", "")
HOST       = os.getenv("OPENALGO_HOST",   "http://127.0.0.1:5000")
SYMBOL     = os.getenv("SYMBOL",          "RELIANCE")
EXCHANGE   = os.getenv("EXCHANGE",        "NSE")
PRODUCT    = os.getenv("PRODUCT",         "MIS")
QUANTITY   = int(os.getenv("QUANTITY",    "1"))

# ── Supertrend parameters (tune via supertrend_optimization.py) ──
ATR_PERIOD     = int(float(os.getenv("ATR_PERIOD",     "7")))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "3.0"))

INTERVAL      = os.getenv("INTERVAL",       "15m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "5"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", "60"))

STRATEGY_NAME = f"Supertrend({ATR_PERIOD},{ATR_MULTIPLIER}) {SYMBOL}"

# ─────────────────── VALIDATION ───────────────────────────
if not API_KEY:
    print("Error: OPENALGO_APIKEY environment variable is not set")
    raise SystemExit(1)

# ─────────────────── INIT CLIENT ──────────────────────────
client = api(api_key=API_KEY, host=HOST)


# ──────────────────── SUPERTREND ──────────────────────────
def calculate_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    """
    Returns a boolean Series: True = bullish (price above Supertrend),
    False = bearish (price below Supertrend).
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (close.shift() - low).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()

    hl2   = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = pd.Series(True, index=df.index)

    for i in range(1, len(df)):
        if upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iat[i] = upper.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iat[i] = lower.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iat[i] = True
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iat[i] = False
        else:
            trend.iat[i] = trend.iat[i - 1]

    return trend


# ─────────────────── MAIN STRATEGY LOOP ───────────────────
def run_strategy() -> None:
    print(f"{'='*60}")
    print(f"  Supertrend Strategy Started")
    print(f"  Symbol     : {SYMBOL} ({EXCHANGE})")
    print(f"  Parameters : ATR({ATR_PERIOD}), Multiplier({ATR_MULTIPLIER})")
    print(f"  Interval   : {INTERVAL}")
    print(f"  Quantity   : {QUANTITY}  |  Product : {PRODUCT}")
    print(f"{'='*60}\n")

    position = 0  # 0 = flat, >0 = long

    while True:
        try:
            now        = datetime.now()
            end_date   = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

            df = client.history(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=INTERVAL,
                start_date=start_date,
                end_date=end_date,
            )

            if isinstance(df, dict):
                df = pd.DataFrame(df.get("data", df))
            if df.empty:
                print(f"[{now:%H:%M:%S}] No data received — retrying…")
                time.sleep(SLEEP_SECONDS)
                continue

            df.columns = df.columns.str.lower()

            # Use the second-to-last candle (last confirmed closed candle)
            trend = calculate_supertrend(df, ATR_PERIOD, ATR_MULTIPLIER)
            if len(trend) < 3:
                print(f"[{now:%H:%M:%S}] Not enough bars yet — waiting…")
                time.sleep(SLEEP_SECONDS)
                continue

            current_trend = bool(trend.iloc[-2])   # True = bullish
            prev_trend    = bool(trend.iloc[-3])

            bullish_crossover = (not prev_trend) and current_trend
            bearish_crossover = prev_trend and (not current_trend)

            ltp = float(df["close"].iloc[-2])
            print(
                f"[{now:%H:%M:%S}]  LTP={ltp:.2f}  "
                f"Trend={'▲ BULL' if current_trend else '▼ BEAR'}  "
                f"Position={position}"
            )

            # ── BUY on bullish crossover ──
            if bullish_crossover and position <= 0:
                response = client.placesmartorder(
                    strategy=STRATEGY_NAME,
                    symbol=SYMBOL,
                    action="BUY",
                    exchange=EXCHANGE,
                    price_type="MARKET",
                    product=PRODUCT,
                    quantity=QUANTITY,
                    position_size=QUANTITY,
                )
                print(f"  → BUY signal | Response: {response}")
                position = QUANTITY

            # ── SELL on bearish crossover ──
            elif bearish_crossover and position > 0:
                response = client.placesmartorder(
                    strategy=STRATEGY_NAME,
                    symbol=SYMBOL,
                    action="SELL",
                    exchange=EXCHANGE,
                    price_type="MARKET",
                    product=PRODUCT,
                    quantity=QUANTITY,
                    position_size=0,
                )
                print(f"  → SELL signal | Response: {response}")
                position = 0

        except KeyboardInterrupt:
            print("\nStrategy stopped by user.")
            break
        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] Error: {exc}")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    run_strategy()
