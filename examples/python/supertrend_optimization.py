# ---------------------------------------------------
# Supertrend Parameter Optimisation using VectorBT
# Strategy: Supertrend — Long/Short signals
# Optimises ATR period and ATR multiplier and plots
# performance heatmaps so you can pick the best set.
#
# Coded by Rajandran R - Creator of OpenAlgo (https://openalgo.in)
# Blog   : www.marketcalls.in
# ---------------------------------------------------
# Prerequisites
#   pip install vectorbt plotly kaleido
# NOTE: This script requires OpenAlgo to be running locally or on a server.
#       Get your API key from your self-hosted OpenAlgo platform.
#       OpenAlgo GitHub: https://github.com/marketcalls/openalgo
# ---------------------------------------------------

print("🔁 OpenAlgo Supertrend Parameter Optimisation running…")

from datetime import datetime, timedelta
from itertools import product as iproduct

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt
from openalgo import api

# ───────────────────────── CONFIG ─────────────────────────
API_KEY  = "your_api_key_here"         # Replace with your OpenAlgo API key
API_HOST = "http://127.0.0.1:5000"

SYMBOL   = "RELIANCE"
EXCHANGE = "NSE"
INTERVAL = "15m"

END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

# Parameter search space
ATR_PERIODS     = range(5,  25, 2)     # 5, 7, 9, 11, … 23
ATR_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Backtest settings
INITIAL_CAPITAL = 100_000
POSITION_SIZE   = 0.5                  # 50 % of equity per trade
FEES            = 0.0011               # 0.11 % round-trip cost

OUTPUT_PREFIX = f"{SYMBOL}_supertrend_optimisation"

# ─────────────────────── INIT CLIENT ──────────────────────
client = api(api_key=API_KEY, host=API_HOST)


# ─────────────────────── FETCH DATA ───────────────────────
def fetch_data() -> pd.DataFrame:
    print(f"\nFetching {SYMBOL} {INTERVAL} data ({START_DATE} → {END_DATE})…")
    resp = client.history(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        interval=INTERVAL,
        start_date=START_DATE,
        end_date=END_DATE,
        source="db",
    )
    if isinstance(resp, pd.DataFrame):
        df = resp.copy()
    else:
        df = pd.DataFrame(resp.get("data", resp))

    if df.empty:
        raise ValueError("No data received from OpenAlgo API")

    if df.index.name == "timestamp" or "timestamp" not in df.columns:
        df.index = pd.to_datetime(df.index)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("datetime")

    df = df.sort_index()
    df.columns = df.columns.str.lower()
    print(f"  ✅ {len(df)} bars loaded  ({df.index[0]} → {df.index[-1]})")
    return df


# ───────────────────── SUPERTREND CALC ────────────────────
def calculate_supertrend(df: pd.DataFrame, atr_period: int, multiplier: float) -> pd.Series:
    """
    Returns a boolean Series: True = price above Supertrend (bullish),
    False = price below Supertrend (bearish).
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (close.shift() - low).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # Rolling Supertrend state
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend       = pd.Series(True, index=df.index)   # True = bullish

    for i in range(1, len(df)):
        # Upper band
        if upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iat[i] = upper.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        # Lower band
        if lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iat[i] = lower.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        # Direction
        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iat[i] = True
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iat[i] = False
        else:
            trend.iat[i] = trend.iat[i - 1]

    return trend


# ─────────────────────── OPTIMISATION ─────────────────────
def run_optimisation(df: pd.DataFrame) -> pd.DataFrame:
    combos = list(iproduct(ATR_PERIODS, ATR_MULTIPLIERS))
    n = len(combos)
    close = df["close"]
    freq_str = pd.infer_freq(df.index) or INTERVAL

    print(f"\n{'='*60}")
    print(f"Running Supertrend parameter sweep — {n} combinations")
    print(f"ATR periods     : {list(ATR_PERIODS)}")
    print(f"ATR multipliers : {ATR_MULTIPLIERS}")
    print(f"{'='*60}\n")

    records = []
    for idx, (period, mult) in enumerate(combos, 1):
        trend = calculate_supertrend(df, period, mult)
        entries = (~trend.shift(1, fill_value=False)) & trend   # False→True cross
        exits   = trend.shift(1, fill_value=True) & (~trend)    # True→False cross

        try:
            pf = vbt.Portfolio.from_signals(
                close,
                entries,
                exits,
                direction="longonly",
                size=POSITION_SIZE,
                size_type="percent",
                fees=FEES,
                init_cash=INITIAL_CAPITAL,
                freq=freq_str,
                min_size=1,
                size_granularity=1,
            )
            ret     = float(pf.total_return()) * 100
            sharpe  = float(pf.sharpe_ratio())
            max_dd  = float(pf.max_drawdown()) * 100
            n_trades = int(pf.trades.count())
            wr      = float(pf.trades.win_rate()) * 100
        except Exception:
            ret = sharpe = max_dd = wr = float("nan")
            n_trades = 0

        records.append(
            {
                "atr_period":    period,
                "atr_multiplier": mult,
                "total_return":  ret,
                "sharpe_ratio":  sharpe,
                "max_drawdown":  max_dd,
                "num_trades":    n_trades,
                "win_rate":      wr,
            }
        )

        if idx % 10 == 0:
            print(f"  Progress: {idx}/{n} combinations tested…")

    results = pd.DataFrame(records)
    results = results.sort_values("sharpe_ratio", ascending=False)
    return results


# ──────────────────────── HEATMAPS ────────────────────────
def _pivot(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    return results.pivot(index="atr_period", columns="atr_multiplier", values=metric)


def plot_heatmaps(results: pd.DataFrame) -> go.Figure:
    metrics = [
        ("sharpe_ratio",  "Sharpe Ratio",    "RdYlGn",   False),
        ("total_return",  "Total Return (%)", "RdYlGn",   False),
        ("max_drawdown",  "Max Drawdown (%)", "RdYlGn_r", False),
        ("win_rate",      "Win Rate (%)",      "RdYlGn",   False),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (metric, title, cscale, rev), (r, c) in zip(metrics, row_col):
        grid = _pivot(results, metric)
        fig.add_trace(
            go.Heatmap(
                z=grid.values,
                x=[str(v) for v in grid.columns],
                y=[str(v) for v in grid.index],
                colorscale=cscale,
                reversescale=rev,
                showscale=True,
                colorbar=dict(len=0.45, thickness=12, x=0.48 if c == 1 else 1.0),
                text=np.round(grid.values, 2),
                texttemplate="%{text}",
                hoverongaps=False,
                name=title,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"{SYMBOL} Supertrend Parameter Optimisation<br>"
                f"<sup>{START_DATE} → {END_DATE} | {INTERVAL} bars | "
                f"Capital ₹{INITIAL_CAPITAL:,}</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        template="plotly_dark",
        height=900,
        width=1400,
    )

    for r, c in row_col:
        fig.update_xaxes(title_text="ATR Multiplier", row=r, col=c)
        fig.update_yaxes(title_text="ATR Period",     row=r, col=c)

    return fig


def plot_equity_curves(df: pd.DataFrame, results: pd.DataFrame, top_n: int = 5) -> go.Figure:
    top   = results.head(top_n)
    close = df["close"]
    freq_str = pd.infer_freq(df.index) or INTERVAL

    fig = go.Figure()

    for _, row in top.iterrows():
        period = int(row["atr_period"])
        mult   = float(row["atr_multiplier"])
        trend  = calculate_supertrend(df, period, mult)
        entries = (~trend.shift(1, fill_value=False)) & trend
        exits   = trend.shift(1, fill_value=True) & (~trend)

        pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            direction="longonly",
            size=POSITION_SIZE,
            size_type="percent",
            fees=FEES,
            init_cash=INITIAL_CAPITAL,
            freq=freq_str,
            min_size=1,
            size_granularity=1,
        )
        equity = pf.value()
        label  = (
            f"ST({period},{mult}) | Sharpe={row['sharpe_ratio']:.2f} "
            f"| Ret={row['total_return']:.1f}%"
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=equity, name=label, mode="lines")
        )

    fig.update_layout(
        title=dict(
            text=f"{SYMBOL} — Top-{top_n} Supertrend Combinations by Sharpe Ratio",
            x=0.5,
        ),
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        height=500,
        width=1400,
        hovermode="x unified",
    )
    return fig


# ──────────────────────── PRINT TOP ───────────────────────
def print_top_results(results: pd.DataFrame, top_n: int = 10) -> None:
    print(f"\n{'='*70}")
    print(f"TOP {top_n} SUPERTREND COMBINATIONS (ranked by Sharpe Ratio)")
    print(f"{'='*70}")
    cols = ["atr_period", "atr_multiplier", "sharpe_ratio", "total_return",
            "max_drawdown", "win_rate", "num_trades"]
    print(results[cols].head(top_n).to_string(index=False))

    best = results.iloc[0]
    print(f"\n{'='*70}")
    print(f"🏆  BEST PARAMETERS  →  Supertrend({int(best['atr_period'])}, {best['atr_multiplier']})")
    print(f"    Sharpe Ratio : {best['sharpe_ratio']:.4f}")
    print(f"    Total Return : {best['total_return']:.2f} %")
    print(f"    Max Drawdown : {best['max_drawdown']:.2f} %")
    print(f"    Win Rate     : {best['win_rate']:.2f} %")
    print(f"    # Trades     : {int(best['num_trades'])}")
    print(f"{'='*70}\n")


# ──────────────────────── MAIN ────────────────────────────
if __name__ == "__main__":
    try:
        # 1. Fetch data
        df = fetch_data()

        # 2. Run optimisation sweep
        results = run_optimisation(df)

        # 3. Print leaderboard
        print_top_results(results, top_n=10)

        # 4. Save CSV
        csv_file = f"{OUTPUT_PREFIX}_results.csv"
        results.to_csv(csv_file, index=False)
        print(f"📄  Full results saved → {csv_file}")

        # 5. Heatmap dashboard
        heatmap_fig = plot_heatmaps(results)
        heatmap_html = f"{OUTPUT_PREFIX}_heatmap.html"
        heatmap_fig.write_html(heatmap_html)
        print(f"📊  Heatmap saved     → {heatmap_html}")
        heatmap_fig.show()

        # 6. Equity curves for top-5
        equity_fig = plot_equity_curves(df, results, top_n=5)
        equity_html = f"{OUTPUT_PREFIX}_equity_curves.html"
        equity_fig.write_html(equity_html)
        print(f"📈  Equity curves     → {equity_html}")
        equity_fig.show()

    except Exception as exc:
        print(f"❌  Error: {exc}")
        import traceback
        traceback.print_exc()
