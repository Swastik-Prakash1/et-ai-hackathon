"""
chart_renderer.py — Annotated Candlestick Chart Generator
==========================================================
Renders publication-quality annotated candlestick charts using mplfinance.

Features:
  - OHLCV candlestick chart with volume bars
  - SMA50 / SMA200 overlay lines
  - Bollinger Bands shaded region
  - Pattern markers (arrows/annotations) at detection points
  - RSI subplot
  - MACD subplot with histogram
  - Clean dark theme optimized for dashboard embedding
  - Saves as PNG (for VLM analysis) and returns path

Every chart is self-contained — a retail investor can understand the
signal just by looking at the annotated chart.
"""

import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT DIRECTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHART_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charts")
os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM DARK THEME — Bloomberg-terminal inspired
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#00c853",        # Green for up candles
        down="#ff1744",      # Red for down candles
        edge="inherit",
        wick="inherit",
        volume={"up": "#00c85366", "down": "#ff174466"},
        ohlc="inherit",
    ),
    facecolor="#0d1117",     # GitHub dark background
    edgecolor="#0d1117",
    figcolor="#0d1117",
    gridcolor="#21262d",
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
    rc={
        "font.size": 10,
        "axes.labelcolor": "#8b949e",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    },
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Render annotated chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_chart(
    df: pd.DataFrame,
    symbol: str,
    patterns: Optional[list[dict]] = None,
    indicators: Optional[dict] = None,
    last_n_days: int = 120,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    show_bollinger: bool = True,
    show_sma: bool = True,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Render an annotated candlestick chart as PNG.

    Args:
        df:              OHLCV DataFrame with computed indicators
                         (from ta_processor.compute_indicators)
        symbol:          Ticker symbol for title
        patterns:        List of detected pattern dicts (from ta_processor.scan_all_patterns)
                         Each must have: pattern_name, pattern_type, detected_date, confidence
        indicators:      Latest indicator snapshot (optional, for subtitle)
        last_n_days:     Number of trading days to show (default 120 = ~6 months)
        show_volume:     Show volume bars
        show_rsi:        Show RSI subplot
        show_macd:       Show MACD subplot
        show_bollinger:  Show Bollinger Bands
        show_sma:        Show SMA50/SMA200
        title:           Custom title (auto-generated if None)
        output_path:     Custom save path (auto-generated if None)

    Returns:
        Absolute path to the saved PNG file.
    """
    if df.empty:
        logger.error("Cannot render chart — empty DataFrame")
        return ""

    # Slice to last N days
    chart_df = df.iloc[-last_n_days:].copy()
    if chart_df.empty:
        chart_df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    # ── Build addplots ──────────────────────────────────────
    addplots = []
    panel_count = 0  # 0 = main chart, panels added below

    # SMA overlays on main chart
    if show_sma:
        if "SMA50" in chart_df.columns:
            sma50 = chart_df["SMA50"]
            addplots.append(mpf.make_addplot(
                sma50, panel=0, color="#42a5f5", width=1.2,
                linestyle="-", secondary_y=False,
            ))
        if "SMA200" in chart_df.columns:
            sma200 = chart_df["SMA200"]
            addplots.append(mpf.make_addplot(
                sma200, panel=0, color="#ffa726", width=1.2,
                linestyle="-", secondary_y=False,
            ))

    # EMA20 on main chart
    if "EMA20" in chart_df.columns:
        addplots.append(mpf.make_addplot(
            chart_df["EMA20"], panel=0, color="#ab47bc", width=0.8,
            linestyle="--", secondary_y=False,
        ))

    # Bollinger Bands — fill between upper and lower
    if show_bollinger and "BB_upper" in chart_df.columns and "BB_lower" in chart_df.columns:
        addplots.append(mpf.make_addplot(
            chart_df["BB_upper"], panel=0, color="#78909c",
            width=0.7, linestyle=":", secondary_y=False,
        ))
        addplots.append(mpf.make_addplot(
            chart_df["BB_lower"], panel=0, color="#78909c",
            width=0.7, linestyle=":", secondary_y=False,
        ))

    # Volume panel is automatic if show_volume=True

    # RSI subplot
    rsi_panel = None
    if show_rsi and "RSI_14" in chart_df.columns:
        panel_count += 1
        rsi_panel = panel_count + 1  # +1 because volume takes panel 1
        addplots.append(mpf.make_addplot(
            chart_df["RSI_14"], panel=rsi_panel, color="#42a5f5",
            width=1.0, ylabel="RSI", secondary_y=False,
        ))
        # Overbought / oversold lines
        rsi_70 = pd.Series(70.0, index=chart_df.index)
        rsi_30 = pd.Series(30.0, index=chart_df.index)
        addplots.append(mpf.make_addplot(
            rsi_70, panel=rsi_panel, color="#ff1744",
            width=0.5, linestyle="--", secondary_y=False,
        ))
        addplots.append(mpf.make_addplot(
            rsi_30, panel=rsi_panel, color="#00c853",
            width=0.5, linestyle="--", secondary_y=False,
        ))

    # MACD subplot
    if show_macd and "MACD" in chart_df.columns and "MACD_Signal" in chart_df.columns:
        panel_count += 1
        macd_panel = panel_count + 1
        addplots.append(mpf.make_addplot(
            chart_df["MACD"], panel=macd_panel, color="#42a5f5",
            width=1.0, ylabel="MACD", secondary_y=False,
        ))
        addplots.append(mpf.make_addplot(
            chart_df["MACD_Signal"], panel=macd_panel, color="#ffa726",
            width=1.0, secondary_y=False,
        ))
        if "MACD_Hist" in chart_df.columns:
            hist = chart_df["MACD_Hist"]
            hist_colors = ["#00c853" if v >= 0 else "#ff1744" for v in hist]
            addplots.append(mpf.make_addplot(
                hist, type="bar", panel=macd_panel,
                color=hist_colors, width=0.7, secondary_y=False,
            ))

    # ── Pattern markers on main chart ───────────────────────
    if patterns:
        _add_pattern_markers(chart_df, patterns, addplots)

    # ── Generate title ──────────────────────────────────────
    if title is None:
        pattern_summary = ""
        if patterns:
            bullish = [p for p in patterns if p.get("pattern_type") == "bullish"]
            bearish = [p for p in patterns if p.get("pattern_type") == "bearish"]
            parts = []
            if bullish:
                names = ", ".join([p["pattern_name"] for p in bullish[:2]])
                parts.append(f"Bullish: {names}")
            if bearish:
                names = ", ".join([p["pattern_name"] for p in bearish[:2]])
                parts.append(f"Bearish: {names}")
            if parts:
                pattern_summary = " | ".join(parts)

        title = f"{symbol}.NS"
        if pattern_summary:
            title += f" — {pattern_summary}"

    # ── Panel ratios ────────────────────────────────────────
    panel_ratios = [4, 1]  # main + volume
    if show_rsi and "RSI_14" in chart_df.columns:
        panel_ratios.append(1.2)
    if show_macd and "MACD" in chart_df.columns:
        panel_ratios.append(1.2)

    # ── Output path ─────────────────────────────────────────
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}.png"
        output_path = os.path.join(CHART_OUTPUT_DIR, filename)

    # ── Render ──────────────────────────────────────────────
    try:
        fig, axes = mpf.plot(
            chart_df,
            type="candle",
            style=DARK_STYLE,
            title=title,
            volume=show_volume,
            addplot=addplots if addplots else None,
            panel_ratios=tuple(panel_ratios),
            figsize=(16, 10),
            tight_layout=True,
            returnfig=True,
            warn_too_much_data=1000,
        )

        # Add legend for overlay lines
        main_ax = axes[0]
        legend_items = []
        if show_sma and "SMA50" in chart_df.columns:
            legend_items.append(("SMA 50", "#42a5f5"))
        if show_sma and "SMA200" in chart_df.columns:
            legend_items.append(("SMA 200", "#ffa726"))
        if "EMA20" in chart_df.columns:
            legend_items.append(("EMA 20", "#ab47bc"))
        if show_bollinger and "BB_upper" in chart_df.columns:
            legend_items.append(("Bollinger Bands", "#78909c"))

        if legend_items:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=color, linewidth=1.5, label=label)
                for label, color in legend_items
            ]
            main_ax.legend(
                handles=handles, loc="upper left",
                fontsize=8, framealpha=0.7,
                facecolor="#161b22", edgecolor="#30363d",
                labelcolor="#c9d1d9",
            )

        # Add current price annotation
        last_close = chart_df["Close"].iloc[-1]
        main_ax.axhline(
            y=last_close, color="#58a6ff", linewidth=0.5,
            linestyle="--", alpha=0.5,
        )
        main_ax.annotate(
            f"₹{last_close:,.0f}",
            xy=(1.01, last_close),
            xycoords=("axes fraction", "data"),
            fontsize=9, color="#58a6ff",
            fontweight="bold",
            va="center",
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#0d1117",
            edgecolor="none",
            pad_inches=0.3,
        )
        plt.close(fig)

        logger.info("Chart saved: %s", output_path)
        return os.path.abspath(output_path)

    except Exception as exc:
        logger.error("Failed to render chart for %s: %s", symbol, exc)
        plt.close("all")
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK RENDER — Minimal args, just give me a chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quick_chart(
    symbol: str,
    last_n_days: int = 120,
    output_path: Optional[str] = None,
) -> str:
    """
    One-liner chart generation: fetch data, compute indicators, render.

    Args:
        symbol:      NSE ticker (e.g. "RELIANCE")
        last_n_days: Trading days to show
        output_path: Custom save path (auto if None)

    Returns:
        Absolute path to saved PNG.
    """
    from data.nse_fetcher import fetch_ohlcv
    from data.ta_processor import compute_indicators, scan_all_patterns

    df = fetch_ohlcv(symbol)
    if df.empty:
        logger.error("No data for %s — cannot render chart", symbol)
        return ""

    df = compute_indicators(df)
    result = scan_all_patterns(df, symbol=symbol)

    return render_chart(
        df=df,
        symbol=symbol,
        patterns=result.get("patterns"),
        indicators=result.get("indicators"),
        last_n_days=last_n_days,
        output_path=output_path,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN MARKERS — Annotate detected patterns on the chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _add_pattern_markers(
    chart_df: pd.DataFrame,
    patterns: list[dict],
    addplots: list,
) -> None:
    """
    Add up/down arrow markers at pattern detection dates.

    Bullish patterns → green up arrows below the low
    Bearish patterns → red down arrows above the high
    """
    bull_markers = pd.Series(np.nan, index=chart_df.index)
    bear_markers = pd.Series(np.nan, index=chart_df.index)

    for p in patterns:
        det_date = p.get("detected_date", "")
        if not det_date:
            continue

        try:
            dt = pd.Timestamp(det_date)
        except Exception:
            continue

        # Find closest date in our chart data
        if dt not in chart_df.index:
            # Find nearest
            idx_pos = chart_df.index.searchsorted(dt)
            if idx_pos >= len(chart_df.index):
                idx_pos = len(chart_df.index) - 1
            dt = chart_df.index[idx_pos]

        if p.get("pattern_type") == "bullish":
            # Place marker below the low
            offset = chart_df.loc[dt, "Low"] * 0.985
            bull_markers.loc[dt] = offset
        elif p.get("pattern_type") == "bearish":
            # Place marker above the high
            offset = chart_df.loc[dt, "High"] * 1.015
            bear_markers.loc[dt] = offset

    # Add bullish markers (green up triangles)
    if bull_markers.notna().any():
        addplots.append(mpf.make_addplot(
            bull_markers, panel=0, type="scatter",
            marker="^", markersize=100, color="#00c853",
            secondary_y=False,
        ))

    # Add bearish markers (red down triangles)
    if bear_markers.notna().any():
        addplots.append(mpf.make_addplot(
            bear_markers, panel=0, type="scatter",
            marker="v", markersize=100, color="#ff1744",
            secondary_y=False,
        ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("CHART RENDERER — Smoke Test")
    print("=" * 60)

    ticker = "RELIANCE"
    print(f"\nGenerating annotated chart for {ticker}...")

    path = quick_chart(ticker, last_n_days=120)

    if path:
        print(f"Chart saved: {path}")
        size_kb = os.path.getsize(path) / 1024
        print(f"File size: {size_kb:.0f} KB")
        print("PASS")
    else:
        print("FAIL — chart not generated")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)
