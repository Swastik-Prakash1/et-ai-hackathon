"""
ta_processor.py — Technical Analysis Pattern Detection Engine
==============================================================
Consumes OHLCV DataFrames from nse_fetcher and detects actionable
technical patterns using pandas-ta + custom rule-based logic.

Patterns detected:
  TREND:        Golden Cross, Death Cross
  MOMENTUM:     RSI Divergence (bullish & bearish), MACD Crossover
  VOLATILITY:   Bollinger Band Squeeze & Breakout
  CANDLESTICK:  Hammer, Bullish Engulfing, Bearish Engulfing,
                Doji, Morning Star, Evening Star

Every detection returns a structured PatternSignal dict with:
  - pattern name, type (bullish/bearish/neutral)
  - confidence score (0-100)
  - plain-English explanation (Class 10 reading level)
  - detection date
  - raw indicator values for charting
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN SIGNAL — Standardized output for every detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PatternSignal:
    """Standard output from every pattern detector."""
    pattern_name: str               # e.g. "Golden Cross"
    pattern_type: str               # "bullish" | "bearish" | "neutral"
    confidence: float               # 0-100
    explanation: str                # plain-English, Class 10 level
    detected_date: str              # ISO date string
    category: str                   # "trend" | "momentum" | "volatility" | "candlestick"
    indicator_values: dict          # raw numbers for charting / RAG citation
    suggested_action: str           # "buy" | "watch" | "avoid"

    def to_dict(self) -> dict:
        return asdict(self)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Compute all technical indicators on a DataFrame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV DataFrame in-place.

    Expects columns: Open, High, Low, Close, Volume
    Adds: SMA50, SMA200, RSI_14, MACD, MACD_Signal, MACD_Hist,
          BB_upper, BB_middle, BB_lower, BB_bandwidth, ATR_14,
          plus candlestick pattern columns.

    Returns the enriched DataFrame (also modifies in-place).
    """
    if df.empty or len(df) < 200:
        logger.warning(
            "DataFrame has %d rows — need 200+ for SMA200. "
            "Some indicators will have NaN values.", len(df)
        )

    # ── Moving Averages ─────────────────────────────────────
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["SMA200"] = ta.sma(df["Close"], length=200)
    df["EMA20"] = ta.ema(df["Close"], length=20)

    # ── RSI ─────────────────────────────────────────────────
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # ── MACD ────────────────────────────────────────────────
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"] = macd.iloc[:, 0]         # MACD line
        df["MACD_Hist"] = macd.iloc[:, 1]    # Histogram
        df["MACD_Signal"] = macd.iloc[:, 2]  # Signal line

    # ── Bollinger Bands ─────────────────────────────────────
    bb = ta.bbands(df["Close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        df["BB_lower"] = bb.iloc[:, 0]
        df["BB_middle"] = bb.iloc[:, 1]
        df["BB_upper"] = bb.iloc[:, 2]
        df["BB_bandwidth"] = bb.iloc[:, 3]   # Bandwidth
        df["BB_pctb"] = bb.iloc[:, 4]        # %B

    # ── ATR (for confidence scaling) ────────────────────────
    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── Volume SMA (for confirmation) ──────────────────────
    df["Vol_SMA20"] = ta.sma(df["Volume"], length=20)

    logger.info("Computed all indicators (%d rows)", len(df))
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN 1: GOLDEN CROSS / DEATH CROSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_golden_death_cross(df: pd.DataFrame) -> list[PatternSignal]:
    """
    Golden Cross: SMA50 crosses ABOVE SMA200 → bullish
    Death Cross:  SMA50 crosses BELOW SMA200 → bearish

    Confidence factors:
    - Volume confirmation (above-average volume on crossover day)
    - Spread between SMAs (wider = stronger conviction)
    - Recent price action (close above SMA50 = confirmation)
    """
    signals = []
    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return signals

    sma50 = df["SMA50"]
    sma200 = df["SMA200"]

    # Find crossover points (only in last 5 trading days for "active" signals)
    lookback = min(5, len(df) - 1)
    for i in range(-lookback, 0):
        idx = len(df) + i
        if idx < 1:
            continue

        prev_50 = sma50.iloc[idx - 1]
        curr_50 = sma50.iloc[idx]
        prev_200 = sma200.iloc[idx - 1]
        curr_200 = sma200.iloc[idx]

        if pd.isna(prev_50) or pd.isna(curr_50) or pd.isna(prev_200) or pd.isna(curr_200):
            continue

        is_golden = prev_50 <= prev_200 and curr_50 > curr_200
        is_death = prev_50 >= prev_200 and curr_50 < curr_200

        if not is_golden and not is_death:
            continue

        # ── Confidence Calculation ──────────────────────────
        confidence = 60.0  # Base confidence for any crossover

        # Volume confirmation (+15 max)
        vol = df["Volume"].iloc[idx]
        vol_avg = df["Vol_SMA20"].iloc[idx] if "Vol_SMA20" in df.columns else vol
        if not pd.isna(vol_avg) and vol_avg > 0:
            vol_ratio = vol / vol_avg
            confidence += min(vol_ratio - 1.0, 1.0) * 15  # Up to +15

        # SMA spread (+15 max) — wider spread = stronger
        spread_pct = abs(curr_50 - curr_200) / curr_200 * 100
        confidence += min(spread_pct, 3.0) * 5  # Up to +15

        # Price above SMA50 on golden (below on death) (+10)
        close = df["Close"].iloc[idx]
        if is_golden and close > curr_50:
            confidence += 10
        elif is_death and close < curr_50:
            confidence += 10

        confidence = min(confidence, 95.0)

        date_str = df.index[idx].strftime("%Y-%m-%d")

        if is_golden:
            signals.append(PatternSignal(
                pattern_name="Golden Cross",
                pattern_type="bullish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock's 50-day average price just crossed above its "
                    f"200-day average price on {date_str}. This is called a Golden Cross — "
                    f"it means the stock's recent trend is getting stronger compared to "
                    f"its long-term trend. Historically, this often leads to further price "
                    f"increases over the next few weeks."
                ),
                detected_date=date_str,
                category="trend",
                indicator_values={
                    "sma50": round(curr_50, 2),
                    "sma200": round(curr_200, 2),
                    "close": round(close, 2),
                    "spread_pct": round(spread_pct, 2),
                    "volume_ratio": round(vol / vol_avg, 2) if vol_avg > 0 else 1.0,
                },
                suggested_action="buy",
            ))
        else:
            signals.append(PatternSignal(
                pattern_name="Death Cross",
                pattern_type="bearish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock's 50-day average price just crossed below its "
                    f"200-day average price on {date_str}. This is called a Death Cross — "
                    f"it means the stock's recent momentum is weakening compared to "
                    f"its long-term trend. This pattern often comes before further "
                    f"price declines."
                ),
                detected_date=date_str,
                category="trend",
                indicator_values={
                    "sma50": round(curr_50, 2),
                    "sma200": round(curr_200, 2),
                    "close": round(close, 2),
                    "spread_pct": round(spread_pct, 2),
                },
                suggested_action="avoid",
            ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN 2: RSI DIVERGENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30) -> list[PatternSignal]:
    """
    Bullish Divergence: Price makes LOWER low, RSI makes HIGHER low
      → price about to reverse upward

    Bearish Divergence: Price makes HIGHER high, RSI makes LOWER high
      → price about to reverse downward

    We compare the most recent swing low/high vs the one before it
    within the lookback window.
    """
    signals = []
    if "RSI_14" not in df.columns or len(df) < lookback + 10:
        return signals

    recent = df.iloc[-lookback:].copy()
    close = recent["Close"]
    rsi = recent["RSI_14"]

    # Find local minima and maxima
    lows_idx = _find_swing_points(close, mode="low")
    highs_idx = _find_swing_points(close, mode="high")

    # ── Bullish divergence (lower price low + higher RSI low) ──
    if len(lows_idx) >= 2:
        prev_low_i = lows_idx[-2]
        curr_low_i = lows_idx[-1]

        prev_price = close.iloc[prev_low_i]
        curr_price = close.iloc[curr_low_i]
        prev_rsi = rsi.iloc[prev_low_i]
        curr_rsi = rsi.iloc[curr_low_i]

        if (
            not pd.isna(prev_rsi) and not pd.isna(curr_rsi)
            and curr_price < prev_price  # Lower low in price
            and curr_rsi > prev_rsi      # Higher low in RSI
            and curr_rsi < 40            # RSI in oversold territory
        ):
            confidence = 55.0
            # Stronger if RSI is deeply oversold
            if curr_rsi < 30:
                confidence += 15
            # Stronger if divergence is large
            rsi_diff = curr_rsi - prev_rsi
            confidence += min(rsi_diff * 2, 15)
            # Volume confirmation
            if "Vol_SMA20" in recent.columns:
                vol = recent["Volume"].iloc[curr_low_i]
                vol_avg = recent["Vol_SMA20"].iloc[curr_low_i]
                if not pd.isna(vol_avg) and vol_avg > 0 and vol > vol_avg:
                    confidence += 10

            confidence = min(confidence, 90.0)
            date_str = recent.index[curr_low_i].strftime("%Y-%m-%d")

            signals.append(PatternSignal(
                pattern_name="Bullish RSI Divergence",
                pattern_type="bullish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock price made a new low, but its momentum indicator (RSI) "
                    f"actually made a higher low. This mismatch — called bullish divergence — "
                    f"suggests that selling pressure is weakening. The RSI is at {curr_rsi:.0f}, "
                    f"which is in the oversold zone. This often signals that the price is "
                    f"about to bounce back up."
                ),
                detected_date=date_str,
                category="momentum",
                indicator_values={
                    "rsi_current": round(curr_rsi, 1),
                    "rsi_previous_low": round(prev_rsi, 1),
                    "price_current_low": round(curr_price, 2),
                    "price_previous_low": round(prev_price, 2),
                },
                suggested_action="buy",
            ))

    # ── Bearish divergence (higher price high + lower RSI high) ──
    if len(highs_idx) >= 2:
        prev_high_i = highs_idx[-2]
        curr_high_i = highs_idx[-1]

        prev_price = close.iloc[prev_high_i]
        curr_price = close.iloc[curr_high_i]
        prev_rsi = rsi.iloc[prev_high_i]
        curr_rsi = rsi.iloc[curr_high_i]

        if (
            not pd.isna(prev_rsi) and not pd.isna(curr_rsi)
            and curr_price > prev_price  # Higher high in price
            and curr_rsi < prev_rsi      # Lower high in RSI
            and curr_rsi > 60            # RSI in overbought territory
        ):
            confidence = 55.0
            if curr_rsi > 70:
                confidence += 15
            rsi_diff = prev_rsi - curr_rsi
            confidence += min(rsi_diff * 2, 15)
            confidence = min(confidence, 90.0)
            date_str = recent.index[curr_high_i].strftime("%Y-%m-%d")

            signals.append(PatternSignal(
                pattern_name="Bearish RSI Divergence",
                pattern_type="bearish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock price made a new high, but its momentum indicator (RSI) "
                    f"made a lower high. This mismatch — called bearish divergence — "
                    f"suggests that buying pressure is fading even though the price is "
                    f"still rising. The RSI is at {curr_rsi:.0f}, which is in the "
                    f"overbought zone. This often signals an upcoming price pullback."
                ),
                detected_date=date_str,
                category="momentum",
                indicator_values={
                    "rsi_current": round(curr_rsi, 1),
                    "rsi_previous_high": round(prev_rsi, 1),
                    "price_current_high": round(curr_price, 2),
                    "price_previous_high": round(prev_price, 2),
                },
                suggested_action="avoid",
            ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN 3: MACD CROSSOVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_macd_crossover(df: pd.DataFrame) -> list[PatternSignal]:
    """
    Bullish MACD Crossover: MACD line crosses ABOVE signal line
    Bearish MACD Crossover: MACD line crosses BELOW signal line

    Strongest when happening near zero line (early trend) or
    in extreme territory (reversal).
    """
    signals = []
    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return signals

    lookback = min(5, len(df) - 1)
    for i in range(-lookback, 0):
        idx = len(df) + i
        if idx < 1:
            continue

        prev_macd = df["MACD"].iloc[idx - 1]
        curr_macd = df["MACD"].iloc[idx]
        prev_sig = df["MACD_Signal"].iloc[idx - 1]
        curr_sig = df["MACD_Signal"].iloc[idx]

        if pd.isna(prev_macd) or pd.isna(curr_macd) or pd.isna(prev_sig) or pd.isna(curr_sig):
            continue

        is_bullish = prev_macd <= prev_sig and curr_macd > curr_sig
        is_bearish = prev_macd >= prev_sig and curr_macd < curr_sig

        if not is_bullish and not is_bearish:
            continue

        confidence = 55.0

        # Histogram strength (+15 max)
        hist = df["MACD_Hist"].iloc[idx] if "MACD_Hist" in df.columns else 0
        if not pd.isna(hist):
            hist_strength = abs(hist) / (df["Close"].iloc[idx] * 0.001 + 0.01)
            confidence += min(hist_strength * 5, 15)

        # Near zero line = early trend signal (+10)
        if abs(curr_macd) < df["Close"].iloc[idx] * 0.005:
            confidence += 10

        # Volume confirmation (+10)
        if "Vol_SMA20" in df.columns:
            vol = df["Volume"].iloc[idx]
            vol_avg = df["Vol_SMA20"].iloc[idx]
            if not pd.isna(vol_avg) and vol_avg > 0 and vol > vol_avg * 1.2:
                confidence += 10

        confidence = min(confidence, 90.0)
        date_str = df.index[idx].strftime("%Y-%m-%d")

        if is_bullish:
            signals.append(PatternSignal(
                pattern_name="Bullish MACD Crossover",
                pattern_type="bullish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The MACD line just crossed above its signal line on {date_str}. "
                    f"Think of MACD as a speedometer for the stock's momentum — when "
                    f"it crosses upward, it means the stock is picking up speed in the "
                    f"upward direction. This is often an early sign that a rally is starting."
                ),
                detected_date=date_str,
                category="momentum",
                indicator_values={
                    "macd": round(curr_macd, 4),
                    "signal": round(curr_sig, 4),
                    "histogram": round(float(hist), 4) if not pd.isna(hist) else 0,
                },
                suggested_action="buy",
            ))
        else:
            signals.append(PatternSignal(
                pattern_name="Bearish MACD Crossover",
                pattern_type="bearish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The MACD line just crossed below its signal line on {date_str}. "
                    f"This means the stock's upward momentum is slowing down and starting "
                    f"to shift downward. It's like a car that was speeding up but is now "
                    f"beginning to brake — a potential early warning of a price decline."
                ),
                detected_date=date_str,
                category="momentum",
                indicator_values={
                    "macd": round(curr_macd, 4),
                    "signal": round(curr_sig, 4),
                    "histogram": round(float(hist), 4) if not pd.isna(hist) else 0,
                },
                suggested_action="avoid",
            ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN 4: BOLLINGER BAND SQUEEZE & BREAKOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_bollinger_squeeze(df: pd.DataFrame) -> list[PatternSignal]:
    """
    Bollinger Squeeze: Bands contract to historically tight levels,
    then price breaks out of the band.

    Squeeze + Upside Breakout → Bullish
    Squeeze + Downside Breakout → Bearish
    Squeeze Active (no breakout yet) → Watch

    Squeeze is detected when bandwidth drops below its 20-period
    rolling percentile (bottom 20%).
    """
    signals = []
    if "BB_bandwidth" not in df.columns or "BB_upper" not in df.columns:
        return signals

    bw = df["BB_bandwidth"]
    if bw.isna().all():
        return signals

    # Rolling 120-day bandwidth percentile to determine "tight"
    bw_pct = bw.rolling(120, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    lookback = min(5, len(df) - 1)
    for i in range(-lookback, 0):
        idx = len(df) + i
        if idx < 1:
            continue

        curr_bw_pct = bw_pct.iloc[idx] if idx < len(bw_pct) else None
        if curr_bw_pct is None or pd.isna(curr_bw_pct):
            continue

        is_squeeze = curr_bw_pct < 0.20  # Bottom 20% = squeeze

        if not is_squeeze:
            continue

        close = df["Close"].iloc[idx]
        upper = df["BB_upper"].iloc[idx]
        lower = df["BB_lower"].iloc[idx]

        if pd.isna(upper) or pd.isna(lower):
            continue

        # Check for breakout
        broke_upper = close > upper
        broke_lower = close < lower

        date_str = df.index[idx].strftime("%Y-%m-%d")
        confidence = 60.0

        if broke_upper:
            # Volume confirmation
            if "Vol_SMA20" in df.columns:
                vol = df["Volume"].iloc[idx]
                vol_avg = df["Vol_SMA20"].iloc[idx]
                if not pd.isna(vol_avg) and vol_avg > 0 and vol > vol_avg * 1.5:
                    confidence += 20
                elif not pd.isna(vol_avg) and vol_avg > 0 and vol > vol_avg:
                    confidence += 10

            # Tighter squeeze = stronger signal
            confidence += (1 - curr_bw_pct) * 15

            confidence = min(confidence, 90.0)

            signals.append(PatternSignal(
                pattern_name="Bollinger Squeeze Breakout (Up)",
                pattern_type="bullish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock's price range had been getting very narrow — like a "
                    f"compressed spring. On {date_str}, the price burst above the upper "
                    f"Bollinger Band. When prices break out after a tight squeeze like "
                    f"this, the move tends to be strong and sustained."
                ),
                detected_date=date_str,
                category="volatility",
                indicator_values={
                    "close": round(close, 2),
                    "upper_band": round(upper, 2),
                    "lower_band": round(lower, 2),
                    "bandwidth_percentile": round(curr_bw_pct * 100, 1),
                },
                suggested_action="buy",
            ))

        elif broke_lower:
            confidence += (1 - curr_bw_pct) * 15
            confidence = min(confidence, 90.0)

            signals.append(PatternSignal(
                pattern_name="Bollinger Squeeze Breakout (Down)",
                pattern_type="bearish",
                confidence=round(confidence, 1),
                explanation=(
                    f"The stock's price range had been getting very narrow, and on "
                    f"{date_str} the price broke below the lower Bollinger Band. "
                    f"This downside breakout after a tight squeeze often leads to a "
                    f"sharp decline."
                ),
                detected_date=date_str,
                category="volatility",
                indicator_values={
                    "close": round(close, 2),
                    "upper_band": round(upper, 2),
                    "lower_band": round(lower, 2),
                    "bandwidth_percentile": round(curr_bw_pct * 100, 1),
                },
                suggested_action="avoid",
            ))

        else:
            # Squeeze active, no breakout yet — WATCH
            signals.append(PatternSignal(
                pattern_name="Bollinger Band Squeeze",
                pattern_type="neutral",
                confidence=round(50 + (1 - curr_bw_pct) * 10, 1),
                explanation=(
                    f"The stock's Bollinger Bands are squeezing tight — the price has "
                    f"been moving in a very narrow range. This usually means a big move "
                    f"is coming soon, but we don't know which direction yet. Watch for "
                    f"a breakout above ₹{upper:.0f} (bullish) or below ₹{lower:.0f} (bearish)."
                ),
                detected_date=date_str,
                category="volatility",
                indicator_values={
                    "close": round(close, 2),
                    "upper_band": round(upper, 2),
                    "lower_band": round(lower, 2),
                    "bandwidth_percentile": round(curr_bw_pct * 100, 1),
                },
                suggested_action="watch",
            ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN 5: CANDLESTICK PATTERNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_candlestick_patterns(df: pd.DataFrame) -> list[PatternSignal]:
    """
    Detect key candlestick patterns on the most recent candles.

    Patterns: Hammer, Bullish Engulfing, Bearish Engulfing,
              Doji, Morning Star, Evening Star
    """
    signals = []
    if len(df) < 5:
        return signals

    # Check last 3 candles for single-candle patterns, last 5 for multi-candle
    for offset in range(1, 4):
        idx = len(df) - offset
        if idx < 2:
            break

        o = df["Open"].iloc[idx]
        h = df["High"].iloc[idx]
        l = df["Low"].iloc[idx]
        c = df["Close"].iloc[idx]
        date_str = df.index[idx].strftime("%Y-%m-%d")

        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            continue

        body_pct = body / total_range

        # Previous candle for context
        prev_o = df["Open"].iloc[idx - 1]
        prev_c = df["Close"].iloc[idx - 1]
        prev_body = abs(prev_c - prev_o)
        prev_bearish = prev_c < prev_o

        # ── HAMMER ──────────────────────────────────────────
        # Small body at top, long lower shadow (2x+ body), tiny upper shadow
        # Must be in a downtrend (price below SMA50 or recent decline)
        if (
            body_pct < 0.35
            and lower_shadow >= body * 2
            and upper_shadow < body * 0.5
        ):
            # Check downtrend context
            in_downtrend = False
            if "SMA50" in df.columns and not pd.isna(df["SMA50"].iloc[idx]):
                in_downtrend = c < df["SMA50"].iloc[idx]
            else:
                # Fallback: check if price declined over last 10 days
                if idx >= 10:
                    in_downtrend = c < df["Close"].iloc[idx - 10]

            if in_downtrend:
                confidence = 55.0
                shadow_ratio = lower_shadow / body if body > 0 else 3
                confidence += min(shadow_ratio * 5, 20)
                confidence = min(confidence, 85.0)

                signals.append(PatternSignal(
                    pattern_name="Hammer",
                    pattern_type="bullish",
                    confidence=round(confidence, 1),
                    explanation=(
                        f"A hammer candlestick appeared on {date_str}. The stock fell "
                        f"significantly during the day but buyers stepped in and pushed "
                        f"the price back up near where it opened. This pattern — a small "
                        f"body with a long lower wick — suggests that the selling pressure "
                        f"may be exhausting and a reversal upward could be coming."
                    ),
                    detected_date=date_str,
                    category="candlestick",
                    indicator_values={
                        "open": round(o, 2), "high": round(h, 2),
                        "low": round(l, 2), "close": round(c, 2),
                        "lower_shadow_ratio": round(shadow_ratio, 1),
                    },
                    suggested_action="buy",
                ))

        # ── DOJI ────────────────────────────────────────────
        # Very small body (< 10% of range), shadows on both sides
        if body_pct < 0.10 and upper_shadow > 0 and lower_shadow > 0:
            confidence = 45.0  # Doji alone is neutral
            rsi = df["RSI_14"].iloc[idx] if "RSI_14" in df.columns else 50

            if not pd.isna(rsi):
                if rsi < 30 or rsi > 70:
                    confidence += 20  # Strong when at extremes
                    pat_type = "bullish" if rsi < 30 else "bearish"
                    action = "buy" if rsi < 30 else "avoid"
                else:
                    pat_type = "neutral"
                    action = "watch"
            else:
                pat_type = "neutral"
                action = "watch"

            signals.append(PatternSignal(
                pattern_name="Doji",
                pattern_type=pat_type,
                confidence=round(confidence, 1),
                explanation=(
                    f"A doji candlestick appeared on {date_str}. The stock opened and "
                    f"closed at nearly the same price (₹{o:.0f}), meaning buyers and "
                    f"sellers were equally matched. A doji signals indecision — the "
                    f"current trend may be about to change direction."
                ),
                detected_date=date_str,
                category="candlestick",
                indicator_values={
                    "open": round(o, 2), "close": round(c, 2),
                    "body_pct": round(body_pct * 100, 1),
                    "rsi": round(float(rsi), 1) if not pd.isna(rsi) else None,
                },
                suggested_action=action,
            ))

        # ── BULLISH ENGULFING ───────────────────────────────
        # Current green candle completely engulfs previous red candle
        if (
            prev_bearish               # Previous was bearish (red)
            and c > o                  # Current is bullish (green)
            and o <= prev_c            # Current open ≤ prev close
            and c >= prev_o            # Current close ≥ prev open
            and body > prev_body       # Current body bigger
        ):
            confidence = 65.0
            engulf_ratio = body / prev_body if prev_body > 0 else 2
            confidence += min(engulf_ratio * 5, 15)

            if "Vol_SMA20" in df.columns:
                vol = df["Volume"].iloc[idx]
                vol_avg = df["Vol_SMA20"].iloc[idx]
                if not pd.isna(vol_avg) and vol_avg > 0 and vol > vol_avg:
                    confidence += 10

            confidence = min(confidence, 90.0)

            signals.append(PatternSignal(
                pattern_name="Bullish Engulfing",
                pattern_type="bullish",
                confidence=round(confidence, 1),
                explanation=(
                    f"A bullish engulfing pattern appeared on {date_str}. "
                    f"Yesterday's small red candle was completely swallowed by today's "
                    f"large green candle — this means buyers have decisively taken control "
                    f"from sellers. This is one of the strongest single-day reversal signals."
                ),
                detected_date=date_str,
                category="candlestick",
                indicator_values={
                    "current_open": round(o, 2), "current_close": round(c, 2),
                    "prev_open": round(prev_o, 2), "prev_close": round(prev_c, 2),
                    "engulf_ratio": round(engulf_ratio, 1),
                },
                suggested_action="buy",
            ))

        # ── BEARISH ENGULFING ───────────────────────────────
        prev_bullish = prev_c > prev_o
        if (
            prev_bullish               # Previous was bullish (green)
            and c < o                  # Current is bearish (red)
            and o >= prev_c            # Current open ≥ prev close
            and c <= prev_o            # Current close ≤ prev open
            and body > prev_body       # Current body bigger
        ):
            confidence = 65.0
            engulf_ratio = body / prev_body if prev_body > 0 else 2
            confidence += min(engulf_ratio * 5, 15)
            confidence = min(confidence, 90.0)

            signals.append(PatternSignal(
                pattern_name="Bearish Engulfing",
                pattern_type="bearish",
                confidence=round(confidence, 1),
                explanation=(
                    f"A bearish engulfing pattern appeared on {date_str}. "
                    f"Yesterday's small green candle was completely swallowed by today's "
                    f"large red candle — sellers have overwhelming taken control. "
                    f"This is a strong reversal signal suggesting the price may decline."
                ),
                detected_date=date_str,
                category="candlestick",
                indicator_values={
                    "current_open": round(o, 2), "current_close": round(c, 2),
                    "prev_open": round(prev_o, 2), "prev_close": round(prev_c, 2),
                    "engulf_ratio": round(engulf_ratio, 1),
                },
                suggested_action="avoid",
            ))

    # ── MORNING STAR (3-candle pattern) ─────────────────────
    _detect_morning_evening_star(df, signals)

    return signals


def _detect_morning_evening_star(df: pd.DataFrame, signals: list[PatternSignal]) -> None:
    """
    Morning Star (bullish): Large red → Small body/doji → Large green
    Evening Star (bearish): Large green → Small body/doji → Large red

    Check the last completed 3-candle window.
    """
    if len(df) < 4:
        return

    for offset in range(0, 2):
        idx = len(df) - 1 - offset
        if idx < 2:
            break

        # Candle 1 (two days ago), Candle 2 (yesterday), Candle 3 (today)
        c1_o, c1_h, c1_l, c1_c = (
            df["Open"].iloc[idx - 2], df["High"].iloc[idx - 2],
            df["Low"].iloc[idx - 2], df["Close"].iloc[idx - 2],
        )
        c2_o, c2_h, c2_l, c2_c = (
            df["Open"].iloc[idx - 1], df["High"].iloc[idx - 1],
            df["Low"].iloc[idx - 1], df["Close"].iloc[idx - 1],
        )
        c3_o, c3_h, c3_l, c3_c = (
            df["Open"].iloc[idx], df["High"].iloc[idx],
            df["Low"].iloc[idx], df["Close"].iloc[idx],
        )

        c1_body = abs(c1_c - c1_o)
        c2_body = abs(c2_c - c2_o)
        c3_body = abs(c3_c - c3_o)

        c1_range = c1_h - c1_l
        c2_range = c2_h - c2_l

        if c1_range == 0 or c2_range == 0:
            continue

        c1_body_pct = c1_body / c1_range
        c2_body_pct = c2_body / c2_range

        date_str = df.index[idx].strftime("%Y-%m-%d")

        # Morning Star: big red + small body + big green
        if (
            c1_c < c1_o                  # Candle 1 bearish
            and c1_body_pct > 0.5        # Large body
            and c2_body_pct < 0.30       # Small body (star)
            and c3_c > c3_o              # Candle 3 bullish
            and c3_body > c1_body * 0.5  # Candle 3 body reasonably large
            and c3_c > (c1_o + c1_c) / 2 # Closes above midpoint of candle 1
        ):
            signals.append(PatternSignal(
                pattern_name="Morning Star",
                pattern_type="bullish",
                confidence=70.0,
                explanation=(
                    f"A morning star pattern completed on {date_str}. This is a 3-day "
                    f"reversal pattern: a large drop, then a day of indecision (small candle), "
                    f"followed by a strong recovery. Like a star appearing just before dawn, "
                    f"it suggests the downtrend is ending and prices may rise."
                ),
                detected_date=date_str,
                category="candlestick",
                indicator_values={
                    "candle1_close": round(c1_c, 2),
                    "candle2_close": round(c2_c, 2),
                    "candle3_close": round(c3_c, 2),
                },
                suggested_action="buy",
            ))

        # Evening Star: big green + small body + big red
        if (
            c1_c > c1_o                  # Candle 1 bullish
            and c1_body_pct > 0.5        # Large body
            and c2_body_pct < 0.30       # Small body (star)
            and c3_c < c3_o              # Candle 3 bearish
            and c3_body > c1_body * 0.5  # Candle 3 body reasonably large
            and c3_c < (c1_o + c1_c) / 2 # Closes below midpoint of candle 1
        ):
            signals.append(PatternSignal(
                pattern_name="Evening Star",
                pattern_type="bearish",
                confidence=70.0,
                explanation=(
                    f"An evening star pattern completed on {date_str}. This is a 3-day "
                    f"reversal pattern: a strong rally, then a day where momentum stalled, "
                    f"followed by a sharp decline. Like a star appearing at dusk, it "
                    f"suggests the uptrend is ending and prices may fall."
                ),
                detected_date=date_str,
                category="candlestick",
                indicator_values={
                    "candle1_close": round(c1_c, 2),
                    "candle2_close": round(c2_c, 2),
                    "candle3_close": round(c3_c, 2),
                },
                suggested_action="avoid",
            ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RSI EXTREME ZONES (bonus — simple but valuable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_rsi_extremes(df: pd.DataFrame) -> list[PatternSignal]:
    """
    Detect when RSI enters extreme oversold (< 30) or overbought (> 70) zones.
    Not a crossover — just the zone entry, as a supporting signal.
    """
    signals = []
    if "RSI_14" not in df.columns:
        return signals

    # Check only the latest candle
    rsi = df["RSI_14"].iloc[-1]
    if pd.isna(rsi):
        return signals

    date_str = df.index[-1].strftime("%Y-%m-%d")

    if rsi < 30:
        signals.append(PatternSignal(
            pattern_name="RSI Oversold",
            pattern_type="bullish",
            confidence=round(50 + (30 - rsi) * 2, 1),
            explanation=(
                f"The RSI indicator is at {rsi:.0f}, which is in the oversold zone "
                f"(below 30). This means the stock has been heavily sold in recent days "
                f"and may be due for a bounce. However, oversold conditions can persist — "
                f"look for confirmation from other signals before acting."
            ),
            detected_date=date_str,
            category="momentum",
            indicator_values={"rsi": round(rsi, 1)},
            suggested_action="watch",
        ))
    elif rsi > 70:
        signals.append(PatternSignal(
            pattern_name="RSI Overbought",
            pattern_type="bearish",
            confidence=round(50 + (rsi - 70) * 2, 1),
            explanation=(
                f"The RSI indicator is at {rsi:.0f}, which is in the overbought zone "
                f"(above 70). This means the stock has rallied sharply and may be "
                f"due for a pullback. However, strong stocks can stay overbought for "
                f"a while — look for other bearish signals to confirm."
            ),
            detected_date=date_str,
            category="momentum",
            indicator_values={"rsi": round(rsi, 1)},
            suggested_action="watch",
        ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER SCANNER — Run ALL pattern detectors on one DataFrame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_all_patterns(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Run every pattern detector on an OHLCV DataFrame.

    Args:
        df:     OHLCV DataFrame (from nse_fetcher.fetch_ohlcv)
        symbol: Ticker symbol (for logging / output tagging)

    Returns:
        {
            "symbol": str,
            "scan_time": str,
            "patterns": list[dict],         # All detected patterns
            "bullish_count": int,
            "bearish_count": int,
            "neutral_count": int,
            "overall_bias": str,            # "bullish" | "bearish" | "neutral"
            "chart_confidence": float,      # 0-100, weighted avg of all signals
            "indicators": dict,             # Latest indicator snapshot
        }
    """
    if df.empty:
        return {
            "symbol": symbol,
            "scan_time": datetime.now().isoformat(),
            "patterns": [],
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "overall_bias": "neutral",
            "chart_confidence": 0.0,
            "indicators": {},
        }

    # Step 1: Compute all indicators
    df = compute_indicators(df)

    # Step 2: Run all detectors
    all_signals: list[PatternSignal] = []
    all_signals.extend(detect_golden_death_cross(df))
    all_signals.extend(detect_rsi_divergence(df))
    all_signals.extend(detect_macd_crossover(df))
    all_signals.extend(detect_bollinger_squeeze(df))
    all_signals.extend(detect_candlestick_patterns(df))
    all_signals.extend(detect_rsi_extremes(df))

    # Deduplicate: keep highest confidence per pattern name
    seen = {}
    for sig in all_signals:
        key = sig.pattern_name
        if key not in seen or sig.confidence > seen[key].confidence:
            seen[key] = sig
    unique_signals = list(seen.values())

    # Step 3: Compute overall bias
    bullish = [s for s in unique_signals if s.pattern_type == "bullish"]
    bearish = [s for s in unique_signals if s.pattern_type == "bearish"]
    neutral = [s for s in unique_signals if s.pattern_type == "neutral"]

    bull_score = sum(s.confidence for s in bullish)
    bear_score = sum(s.confidence for s in bearish)

    if bull_score > bear_score * 1.2:
        overall_bias = "bullish"
    elif bear_score > bull_score * 1.2:
        overall_bias = "bearish"
    else:
        overall_bias = "neutral"

    # Chart confidence: weighted average of pattern confidences
    # Bullish patterns contribute positively, bearish negatively
    if unique_signals:
        weights = []
        for s in unique_signals:
            if s.pattern_type == "bullish":
                weights.append(s.confidence)
            elif s.pattern_type == "bearish":
                weights.append(-s.confidence)
            else:
                weights.append(0)
        # Normalize to 0-100 range (50 = neutral)
        raw = sum(weights) / len(weights)
        chart_confidence = max(0, min(100, 50 + raw / 2))
    else:
        chart_confidence = 50.0  # Neutral if no patterns

    # Step 4: Latest indicator snapshot for charting
    last = df.iloc[-1]
    indicators = {}
    for col in ["SMA50", "SMA200", "EMA20", "RSI_14", "MACD", "MACD_Signal",
                "MACD_Hist", "BB_upper", "BB_middle", "BB_lower",
                "BB_bandwidth", "ATR_14"]:
        val = last.get(col)
        if val is not None and not pd.isna(val):
            indicators[col] = round(float(val), 4)

    indicators["close"] = round(float(last["Close"]), 2)
    indicators["volume"] = int(last["Volume"]) if not pd.isna(last["Volume"]) else 0

    result = {
        "symbol": symbol,
        "scan_time": datetime.now().isoformat(),
        "patterns": [s.to_dict() for s in unique_signals],
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "overall_bias": overall_bias,
        "chart_confidence": round(chart_confidence, 1),
        "indicators": indicators,
    }

    logger.info(
        "%s scan complete: %d patterns (%d bull, %d bear) → %s (confidence=%.1f)",
        symbol, len(unique_signals), len(bullish), len(bearish),
        overall_bias, chart_confidence,
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_swing_points(series: pd.Series, mode: str = "low", window: int = 5) -> list[int]:
    """
    Find local minima (mode='low') or maxima (mode='high') using a
    rolling-window approach. Returns list of integer positions (iloc indices).
    """
    points = []
    vals = series.values
    for i in range(window, len(vals) - window):
        if pd.isna(vals[i]):
            continue
        segment = vals[i - window: i + window + 1]
        if np.any(np.isnan(segment)):
            continue
        if mode == "low" and vals[i] == np.min(segment):
            points.append(i)
        elif mode == "high" and vals[i] == np.max(segment):
            points.append(i)
    return points


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.nse_fetcher import fetch_ohlcv

    print("=" * 60)
    print("TA PROCESSOR — Smoke Test")
    print("=" * 60)

    # Test on a liquid NSE stock
    ticker = "TATAMOTORS"
    print(f"\nFetching OHLCV for {ticker}...")
    df = fetch_ohlcv(ticker)
    print(f"Got {len(df)} candles\n")

    print("Running full pattern scan...")
    result = scan_all_patterns(df, symbol=ticker)

    print(f"\nOverall Bias: {result['overall_bias'].upper()}")
    print(f"Chart Confidence: {result['chart_confidence']}/100")
    print(f"Patterns Found: {len(result['patterns'])} "
          f"({result['bullish_count']} bullish, "
          f"{result['bearish_count']} bearish, "
          f"{result['neutral_count']} neutral)")

    print("\n" + "-" * 60)
    for p in result["patterns"]:
        emoji = "🟢" if p["pattern_type"] == "bullish" else (
            "🔴" if p["pattern_type"] == "bearish" else "🟡"
        )
        print(f"\n{emoji} {p['pattern_name']} ({p['confidence']}% confidence)")
        print(f"   Type: {p['pattern_type']} | Action: {p['suggested_action']}")
        print(f"   {p['explanation'][:120]}...")

    print("\n" + "-" * 60)
    print("Latest Indicators:")
    for k, v in result["indicators"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)
