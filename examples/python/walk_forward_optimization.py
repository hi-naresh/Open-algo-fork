# ---------------------------------------------------
# Walk-Forward Optimisation — EMA Crossover
# Tests whether best in-sample parameters hold out-of-sample.
# Splits the data into N rolling windows (train/test pairs)
# and compares the best IS parameters to OOS performance.
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

print("🔁 OpenAlgo Walk-Forward Optimisation running…")

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

SYMBOL   = "SBIN"
EXCHANGE = "NSE"
INTERVAL = "1d"                        # Daily bars give enough history

# Walk-forward window settings (in trading bars)
IN_SAMPLE_BARS  = 252                  # ~1 year of daily bars for training
OUT_SAMPLE_BARS = 63                   # ~3 months for testing
STEP_BARS       = 63                   # Advance window by this many bars

# Parameter search space (applied during in-sample optimisation)
FAST_RANGE = range(5,  30, 5)
SLOW_RANGE = range(15, 65, 5)

# Backtest settings
INITIAL_CAPITAL = 100_000
POSITION_SIZE   = 0.5
FEES            = 0.0011

OUTPUT_PREFIX = f"{SYMBOL}_walk_forward"

# ─────────────────────── INIT CLIENT ──────────────────────
client = api(api_key=API_KEY, host=API_HOST)


# ─────────────────────── FETCH DATA ───────────────────────
def fetch_data() -> pd.DataFrame:
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    print(f"\nFetching {SYMBOL} {INTERVAL} data ({start_date} → {end_date})…")

    resp = client.history(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        interval=INTERVAL,
        start_date=start_date,
        end_date=end_date,
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


# ─────────────── SINGLE-WINDOW BACKTEST ───────────────────
def best_parameters(close: pd.Series, freq_str: str) -> tuple[int, int]:
    """
    Find the (fast, slow) EMA pair with the best Sharpe Ratio on *close*.
    Returns (fast_period, slow_period).
    """
    combos     = [(f, s) for f, s in iproduct(FAST_RANGE, SLOW_RANGE) if f < s]
    fast_vals  = [c[0] for c in combos]
    slow_vals  = [c[1] for c in combos]

    fast_ema = vbt.MA.run(close, fast_vals, short_name="fast", ewm=True)
    slow_ema = vbt.MA.run(close, slow_vals, short_name="slow", ewm=True)
    entries  = fast_ema.ma_crossed_above(slow_ema)
    exits    = fast_ema.ma_crossed_below(slow_ema)

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
    sharpe  = pf.sharpe_ratio()
    best_idx = int(np.nanargmax(sharpe.values))
    return combos[best_idx]


def backtest_params(close: pd.Series, fast: int, slow: int, freq_str: str) -> dict:
    """Return performance metrics for a single (fast, slow) pair on *close*."""
    fe = vbt.MA.run(close, [fast], short_name="fast", ewm=True)
    se = vbt.MA.run(close, [slow], short_name="slow", ewm=True)
    entries = fe.ma_crossed_above(se)
    exits   = fe.ma_crossed_below(se)

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
    return {
        "total_return": float(pf.total_return()) * 100,
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()) * 100,
        "num_trades":   int(pf.trades.count()),
    }


# ──────────────── WALK-FORWARD ENGINE ─────────────────────
def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    close    = df["close"]
    freq_str = pd.infer_freq(df.index) or INTERVAL
    n        = len(df)

    windows = []
    start   = 0
    while start + IN_SAMPLE_BARS + OUT_SAMPLE_BARS <= n:
        is_end  = start + IN_SAMPLE_BARS
        oos_end = is_end + OUT_SAMPLE_BARS
        windows.append((start, is_end, oos_end))
        start += STEP_BARS

    print(f"\n{'='*60}")
    print(f"Walk-Forward Optimisation  —  {len(windows)} windows")
    print(f"In-sample  : {IN_SAMPLE_BARS} bars")
    print(f"Out-sample : {OUT_SAMPLE_BARS} bars")
    print(f"Step       : {STEP_BARS} bars")
    print(f"{'='*60}\n")

    records = []
    for i, (s, ie, oe) in enumerate(windows, 1):
        is_close  = close.iloc[s:ie]
        oos_close = close.iloc[ie:oe]

        fast, slow = best_parameters(is_close, freq_str)

        is_perf  = backtest_params(is_close,  fast, slow, freq_str)
        oos_perf = backtest_params(oos_close, fast, slow, freq_str)

        records.append(
            {
                "window":           i,
                "is_start":         df.index[s].date(),
                "is_end":           df.index[ie - 1].date(),
                "oos_start":        df.index[ie].date(),
                "oos_end":          df.index[oe - 1].date(),
                "best_fast":        fast,
                "best_slow":        slow,
                "is_sharpe":        is_perf["sharpe_ratio"],
                "is_return":        is_perf["total_return"],
                "oos_sharpe":       oos_perf["sharpe_ratio"],
                "oos_return":       oos_perf["total_return"],
                "oos_max_drawdown": oos_perf["max_drawdown"],
                "oos_num_trades":   oos_perf["num_trades"],
            }
        )

        print(
            f"  Window {i:>2}  IS EMA({fast:>2},{slow:>2}) → "
            f"IS Sharpe={is_perf['sharpe_ratio']:>6.2f}  "
            f"OOS Sharpe={oos_perf['sharpe_ratio']:>6.2f}  "
            f"OOS Ret={oos_perf['total_return']:>7.2f}%"
        )

    return pd.DataFrame(records)


# ─────────────────── VISUALISATION ────────────────────────
def plot_wf_results(results: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "In-Sample vs Out-of-Sample Sharpe Ratio",
            "Out-of-Sample Return (%)",
            "Best EMA Parameters per Window",
        ],
        vertical_spacing=0.10,
        shared_xaxes=True,
    )

    labels = [f"W{r['window']}" for _, r in results.iterrows()]

    # Row 1 – Sharpe
    fig.add_trace(
        go.Bar(x=labels, y=results["is_sharpe"],  name="IS Sharpe",  marker_color="steelblue"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=results["oos_sharpe"], name="OOS Sharpe",
               marker_color=["green" if v >= 0 else "red" for v in results["oos_sharpe"]]),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Row 2 – OOS Return
    fig.add_trace(
        go.Bar(
            x=labels, y=results["oos_return"], name="OOS Return (%)",
            marker_color=["green" if v >= 0 else "red" for v in results["oos_return"]],
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Row 3 – Parameters
    fig.add_trace(
        go.Scatter(x=labels, y=results["best_fast"], name="Fast EMA",
                   mode="lines+markers", marker_color="orange"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=labels, y=results["best_slow"], name="Slow EMA",
                   mode="lines+markers", marker_color="cyan"),
        row=3, col=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{SYMBOL} Walk-Forward Optimisation  |  {INTERVAL} bars  |  "
                f"IS={IN_SAMPLE_BARS} / OOS={OUT_SAMPLE_BARS} bars"
            ),
            x=0.5,
            font=dict(size=15),
        ),
        template="plotly_dark",
        height=900,
        width=1200,
        barmode="group",
        hovermode="x unified",
    )
    return fig


# ──────────────────────── MAIN ────────────────────────────
if __name__ == "__main__":
    try:
        # 1. Fetch data
        df = fetch_data()

        # 2. Run walk-forward
        results = run_walk_forward(df)

        # 3. Summary statistics
        positive_oos = (results["oos_sharpe"] > 0).sum()
        efficiency   = positive_oos / len(results) * 100
        avg_oos_ret  = results["oos_return"].mean()
        avg_oos_sh   = results["oos_sharpe"].mean()

        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY")
        print(f"  Total windows          : {len(results)}")
        print(f"  Positive OOS Sharpe    : {positive_oos}  ({efficiency:.1f}%)")
        print(f"  Average OOS Return     : {avg_oos_ret:.2f}%")
        print(f"  Average OOS Sharpe     : {avg_oos_sh:.4f}")
        print(f"{'='*60}\n")

        # 4. Save CSV
        csv_file = f"{OUTPUT_PREFIX}_results.csv"
        results.to_csv(csv_file, index=False)
        print(f"📄  Full results saved → {csv_file}")

        # 5. Plot
        fig = plot_wf_results(results)
        html_file = f"{OUTPUT_PREFIX}_chart.html"
        fig.write_html(html_file)
        print(f"📊  Chart saved       → {html_file}")
        fig.show()

    except Exception as exc:
        print(f"❌  Error: {exc}")
        import traceback
        traceback.print_exc()
