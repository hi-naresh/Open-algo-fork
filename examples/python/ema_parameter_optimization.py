# ---------------------------------------------------
# EMA Crossover Parameter Optimization using VectorBT
# Strategy: Dual EMA Crossover — Long Only
# Optimises fast/slow EMA periods and plots performance
# heatmaps so you can pick the best parameter set.
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

print("🔁 OpenAlgo EMA Parameter Optimisation running…")

from datetime import datetime, timedelta
from itertools import product as iproduct

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import vectorbt as vbt
from openalgo import api

# ───────────────────────── CONFIG ─────────────────────────
API_KEY  = "your_api_key_here"         # Replace with your OpenAlgo API key
API_HOST = "http://127.0.0.1:5000"

SYMBOL   = "SBIN"
EXCHANGE = "NSE"
INTERVAL = "15m"                       # Candlestick interval

END_DATE   = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

# Parameter search space — adjust ranges to taste
FAST_RANGE = range(5,  30, 5)          # 5, 10, 15, 20, 25
SLOW_RANGE = range(15, 65, 5)          # 15, 20, … 60

# Backtest settings
INITIAL_CAPITAL = 100_000              # ₹1,00,000
POSITION_SIZE   = 0.5                  # 50 % of equity per trade
FEES            = 0.0011               # 0.11 % round-trip cost

# Output file prefix
OUTPUT_PREFIX = f"{SYMBOL}_ema_optimisation"

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


# ─────────────────────── OPTIMISATION ─────────────────────
def run_optimisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test every valid (fast, slow) EMA pair where fast < slow.
    Returns a DataFrame with one row per parameter combination and
    columns: fast, slow, total_return, sharpe_ratio, max_drawdown,
             win_rate, num_trades.
    """
    close = df["close"]

    # Build all valid combinations
    combos = [(f, s) for f, s in iproduct(FAST_RANGE, SLOW_RANGE) if f < s]
    fast_vals = [c[0] for c in combos]
    slow_vals = [c[1] for c in combos]
    n = len(combos)

    print(f"\n{'='*60}")
    print(f"Running parameter sweep — {n} EMA combinations")
    print(f"Fast range : {list(FAST_RANGE)}")
    print(f"Slow range : {list(SLOW_RANGE)}")
    print(f"{'='*60}\n")

    # Calculate all EMAs in one vectorised pass
    fast_ema = vbt.MA.run(close, fast_vals, short_name="fast", ewm=True)
    slow_ema = vbt.MA.run(close, slow_vals, short_name="slow", ewm=True)

    entries = fast_ema.ma_crossed_above(slow_ema)
    exits   = fast_ema.ma_crossed_below(slow_ema)

    # Infer frequency string from data
    freq_str = pd.infer_freq(df.index) or INTERVAL

    portfolio = vbt.Portfolio.from_signals(
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

    # Collect metrics — values() flattens to 1-D array regardless of whether
    # the result is a Series or a single-row DataFrame.
    records = []
    total_return = portfolio.total_return().values.flatten()
    sharpe       = portfolio.sharpe_ratio().values.flatten()
    max_dd       = portfolio.max_drawdown().values.flatten()
    num_trades   = portfolio.trades.count().values.flatten()
    win_rate     = portfolio.trades.win_rate().values.flatten()

    for idx, (f, s) in enumerate(combos):
        records.append(
            {
                "fast":          f,
                "slow":          s,
                "total_return":  float(total_return[idx]) * 100,   # %
                "sharpe_ratio":  float(sharpe[idx]),
                "max_drawdown":  float(max_dd[idx]) * 100,          # %
                "num_trades":    int(num_trades[idx]),
                "win_rate":      float(win_rate[idx]) * 100,        # %
            }
        )

    results = pd.DataFrame(records)
    results = results.sort_values("sharpe_ratio", ascending=False)
    return results


# ──────────────────────── HEATMAPS ────────────────────────
def _pivot(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot results into a fast×slow grid for heatmap plotting."""
    return results.pivot(index="fast", columns="slow", values=metric)


def plot_heatmaps(results: pd.DataFrame) -> go.Figure:
    """
    Create a 2×2 grid of heatmaps:
      • Sharpe Ratio      • Total Return (%)
      • Max Drawdown (%)  • Win Rate (%)
    """
    metrics = [
        ("sharpe_ratio",  "Sharpe Ratio",      "RdYlGn",          False),
        ("total_return",  "Total Return (%)",   "RdYlGn",          False),
        ("max_drawdown",  "Max Drawdown (%)",   "RdYlGn_r",        False),
        ("win_rate",      "Win Rate (%)",        "RdYlGn",          False),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    row_col = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (metric, title, cscale, rev_scale), (r, c) in zip(metrics, row_col):
        grid = _pivot(results, metric)
        fig.add_trace(
            go.Heatmap(
                z=grid.values,
                x=[str(v) for v in grid.columns],
                y=[str(v) for v in grid.index],
                colorscale=cscale,
                reversescale=rev_scale,
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
                f"{SYMBOL} EMA Crossover Parameter Optimisation<br>"
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

    # Label axes
    for r, c in row_col:
        fig.update_xaxes(title_text="Slow EMA Period", row=r, col=c)
        fig.update_yaxes(title_text="Fast EMA Period", row=r, col=c)

    return fig


def plot_equity_curves(df: pd.DataFrame, results: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """
    Plot equity curves for the top-N parameter sets ranked by Sharpe Ratio.
    """
    top = results.head(top_n)
    close = df["close"]

    fig = go.Figure()

    for _, row in top.iterrows():
        f, s = int(row["fast"]), int(row["slow"])
        fast_ema = vbt.MA.run(close, [f], short_name="fast", ewm=True)
        slow_ema = vbt.MA.run(close, [s], short_name="slow", ewm=True)
        entries  = fast_ema.ma_crossed_above(slow_ema)
        exits    = fast_ema.ma_crossed_below(slow_ema)

        freq_str = pd.infer_freq(df.index) or INTERVAL
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
            f"EMA({f},{s}) | Sharpe={row['sharpe_ratio']:.2f} "
            f"| Ret={row['total_return']:.1f}%"
        )
        fig.add_trace(go.Scatter(x=df.index, y=equity.iloc[:, 0], name=label, mode="lines"))

    fig.update_layout(
        title=dict(
            text=f"{SYMBOL} — Top-{top_n} EMA Combinations by Sharpe Ratio",
            x=0.5,
            font=dict(size=15),
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
    print(f"TOP {top_n} PARAMETER COMBINATIONS (ranked by Sharpe Ratio)")
    print(f"{'='*70}")
    cols = ["fast", "slow", "sharpe_ratio", "total_return",
            "max_drawdown", "win_rate", "num_trades"]
    print(results[cols].head(top_n).to_string(index=False))

    best = results.iloc[0]
    print(f"\n{'='*70}")
    print(f"🏆  BEST PARAMETERS  →  EMA({int(best['fast'])}, {int(best['slow'])})")
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
