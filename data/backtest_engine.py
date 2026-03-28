"""
backtest_engine.py — Per-Pattern Per-Stock Historical Win Rate Calculator
=========================================================================
THE MOAT. No other team computes this.

Instead of saying "golden cross works 65% of the time," our system says:
"Golden cross on TATAMOTORS specifically has worked 71% of the time in
the last 2 years, with an average gain of +8.3% on winning trades."

How it works:
  1. Takes 2 years of OHLCV data for a stock
  2. Re-runs pattern detection (using ta_processor) on every historical bar
  3. For each pattern occurrence, measures what actually happened over the
     next N trading days (default: 5, 10, 20, 30)
  4. Computes win rate, avg gain, avg loss, expectancy, max gain, max loss
  5. Returns a BacktestResult per pattern with full trade-level detail

All results include the specific dates when the pattern fired so the
Reasoning Agent can cite them as evidence.
"""

import logging
from dataclasses import dataclass, asdict, field
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
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TradeOutcome:
    """One historical occurrence of a pattern and its subsequent outcome."""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    return_pct: float
    is_win: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Full backtest statistics for one pattern on one stock."""
    symbol: str
    pattern_name: str
    pattern_type: str              # bullish / bearish
    holding_period: int            # days held after signal
    total_occurrences: int
    wins: int
    losses: int
    win_rate: float                # 0.0-1.0 decimal ratio
    avg_gain_pct: float            # average return on winning trades
    avg_loss_pct: float            # average return on losing trades
    overall_avg_return_pct: float  # average across all trades
    expectancy_pct: float          # (win_rate * avg_gain) + ((1-win_rate) * avg_loss)
    max_gain_pct: float
    max_loss_pct: float
    median_return_pct: float
    sharpe_ratio: float            # risk-adjusted return
    trades: list[TradeOutcome] = field(default_factory=list)
    backtest_time: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trades"] = [t.to_dict() if isinstance(t, TradeOutcome) else t for t in self.trades]
        return d

    @property
    def summary_text(self) -> str:
        """Plain-English summary at Class 10 reading level."""
        if self.total_occurrences == 0:
            return f"No historical occurrences of {self.pattern_name} found."

        direction = "gained" if self.overall_avg_return_pct >= 0 else "lost"
        abs_return = abs(self.overall_avg_return_pct)

        return (
            f"Over the last 2 years, the {self.pattern_name} pattern has appeared "
            f"{self.total_occurrences} time{'s' if self.total_occurrences != 1 else ''} "
            f"on this stock. In {self.win_rate:.0%} of those cases, the stock price "
            f"went up within {self.holding_period} trading days. On average, the stock "
            f"{direction} {abs_return:.1f}% after this pattern. "
            f"The best outcome was +{self.max_gain_pct:.1f}%, and the worst was "
            f"{self.max_loss_pct:.1f}%."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN SCANNERS — Vectorized historical detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# These mirror the logic from ta_processor but are designed to
# return ALL historical occurrences (not just recent ones).

def _find_all_golden_cross(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where golden cross fired."""
    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return []
    sma50 = df["SMA50"]
    sma200 = df["SMA200"]
    cross = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    return list(cross[cross].index.map(lambda d: df.index.get_loc(d)))


def _find_all_death_cross(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where death cross fired."""
    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return []
    sma50 = df["SMA50"]
    sma200 = df["SMA200"]
    cross = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))
    return list(cross[cross].index.map(lambda d: df.index.get_loc(d)))


def _find_all_macd_bullish_cross(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where MACD bullish crossover fired."""
    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return []
    macd = df["MACD"]
    sig = df["MACD_Signal"]
    cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    return list(cross[cross].index.map(lambda d: df.index.get_loc(d)))


def _find_all_macd_bearish_cross(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where MACD bearish crossover fired."""
    if "MACD" not in df.columns or "MACD_Signal" not in df.columns:
        return []
    macd = df["MACD"]
    sig = df["MACD_Signal"]
    cross = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    return list(cross[cross].index.map(lambda d: df.index.get_loc(d)))


def _find_all_rsi_oversold(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where RSI crossed below 30 (entering oversold)."""
    if "RSI_14" not in df.columns:
        return []
    rsi = df["RSI_14"]
    entry = (rsi < 30) & (rsi.shift(1) >= 30)
    return list(entry[entry].index.map(lambda d: df.index.get_loc(d)))


def _find_all_rsi_overbought(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where RSI crossed above 70 (entering overbought)."""
    if "RSI_14" not in df.columns:
        return []
    rsi = df["RSI_14"]
    entry = (rsi > 70) & (rsi.shift(1) <= 70)
    return list(entry[entry].index.map(lambda d: df.index.get_loc(d)))


def _find_all_bb_squeeze_breakout_up(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where price broke above upper BB during a squeeze."""
    if "BB_bandwidth" not in df.columns or "BB_upper" not in df.columns:
        return []
    bw = df["BB_bandwidth"]
    # Squeeze: bandwidth below its rolling 25th percentile
    bw_pct = bw.rolling(120, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    squeeze = bw_pct < 0.25
    breakout = df["Close"] > df["BB_upper"]
    signal = squeeze & breakout
    return list(signal[signal].index.map(lambda d: df.index.get_loc(d)))


def _find_all_bb_squeeze_breakout_down(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where price broke below lower BB during a squeeze."""
    if "BB_bandwidth" not in df.columns or "BB_lower" not in df.columns:
        return []
    bw = df["BB_bandwidth"]
    bw_pct = bw.rolling(120, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    squeeze = bw_pct < 0.25
    breakout = df["Close"] < df["BB_lower"]
    signal = squeeze & breakout
    return list(signal[signal].index.map(lambda d: df.index.get_loc(d)))


def _find_all_bullish_engulfing(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where bullish engulfing pattern appeared."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(1, len(df)):
        prev_bearish = c[i - 1] < o[i - 1]
        curr_bullish = c[i] > o[i]
        prev_body = abs(c[i - 1] - o[i - 1])
        curr_body = abs(c[i] - o[i])
        if (
            prev_bearish and curr_bullish
            and o[i] <= c[i - 1]
            and c[i] >= o[i - 1]
            and curr_body > prev_body
        ):
            indices.append(i)
    return indices


def _find_all_bearish_engulfing(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where bearish engulfing pattern appeared."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(1, len(df)):
        prev_bullish = c[i - 1] > o[i - 1]
        curr_bearish = c[i] < o[i]
        prev_body = abs(c[i - 1] - o[i - 1])
        curr_body = abs(c[i] - o[i])
        if (
            prev_bullish and curr_bearish
            and o[i] >= c[i - 1]
            and c[i] <= o[i - 1]
            and curr_body > prev_body
        ):
            indices.append(i)
    return indices


def _find_all_hammer(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where hammer candlestick appeared in downtrend context."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    sma50 = df["SMA50"].values if "SMA50" in df.columns else None
    for i in range(10, len(df)):
        body = abs(c[i] - o[i])
        total_range = h[i] - l[i]
        if total_range == 0:
            continue
        lower_shadow = min(o[i], c[i]) - l[i]
        upper_shadow = h[i] - max(o[i], c[i])
        body_pct = body / total_range
        # Hammer criteria
        if body_pct < 0.35 and lower_shadow >= body * 2 and upper_shadow < body * 0.5:
            # Downtrend context
            in_downtrend = False
            if sma50 is not None and not np.isnan(sma50[i]):
                in_downtrend = c[i] < sma50[i]
            else:
                in_downtrend = c[i] < c[i - 10]
            if in_downtrend:
                indices.append(i)
    return indices


def _find_all_doji(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where doji candlestick appeared."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(len(df)):
        total_range = h[i] - l[i]
        if total_range == 0:
            continue
        body = abs(c[i] - o[i])
        body_pct = body / total_range
        upper_shadow = h[i] - max(o[i], c[i])
        lower_shadow = min(o[i], c[i]) - l[i]
        if body_pct < 0.10 and upper_shadow > 0 and lower_shadow > 0:
            indices.append(i)
    return indices


def _find_all_morning_star(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where morning star (3-candle bullish reversal) completed."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(2, len(df)):
        c1_o, c1_h, c1_l, c1_c = o[i - 2], h[i - 2], l[i - 2], c[i - 2]
        c2_o, c2_h, c2_l, c2_c = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        c3_o, c3_c = o[i], c[i]
        c1_body = abs(c1_c - c1_o)
        c2_body = abs(c2_c - c2_o)
        c3_body = abs(c3_c - c3_o)
        c1_range = c1_h - c1_l
        c2_range = c2_h - c2_l
        if c1_range == 0 or c2_range == 0:
            continue
        if (
            c1_c < c1_o                     # Candle 1 bearish
            and c1_body / c1_range > 0.5    # Large body
            and c2_body / c2_range < 0.30   # Small body (star)
            and c3_c > c3_o                 # Candle 3 bullish
            and c3_body > c1_body * 0.5     # Reasonably large
            and c3_c > (c1_o + c1_c) / 2    # Closes above midpoint of candle 1
        ):
            indices.append(i)
    return indices


def _find_all_evening_star(df: pd.DataFrame) -> list[int]:
    """Return iloc indices where evening star (3-candle bearish reversal) completed."""
    indices = []
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    for i in range(2, len(df)):
        c1_o, c1_h, c1_l, c1_c = o[i - 2], h[i - 2], l[i - 2], c[i - 2]
        c2_o, c2_h, c2_l, c2_c = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        c3_o, c3_c = o[i], c[i]
        c1_body = abs(c1_c - c1_o)
        c2_body = abs(c2_c - c2_o)
        c3_body = abs(c3_c - c3_o)
        c1_range = c1_h - c1_l
        c2_range = c2_h - c2_l
        if c1_range == 0 or c2_range == 0:
            continue
        if (
            c1_c > c1_o                     # Candle 1 bullish
            and c1_body / c1_range > 0.5    # Large body
            and c2_body / c2_range < 0.30   # Small body (star)
            and c3_c < c3_o                 # Candle 3 bearish
            and c3_body > c1_body * 0.5     # Reasonably large
            and c3_c < (c1_o + c1_c) / 2    # Closes below midpoint of candle 1
        ):
            indices.append(i)
    return indices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN REGISTRY — maps pattern name → (finder, type)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN_REGISTRY: dict[str, tuple] = {
    "Golden Cross":                  (_find_all_golden_cross,          "bullish"),
    "Death Cross":                   (_find_all_death_cross,           "bearish"),
    "Bullish MACD Crossover":        (_find_all_macd_bullish_cross,    "bullish"),
    "Bearish MACD Crossover":        (_find_all_macd_bearish_cross,    "bearish"),
    "RSI Oversold Entry":            (_find_all_rsi_oversold,          "bullish"),
    "RSI Overbought Entry":          (_find_all_rsi_overbought,        "bearish"),
    "BB Squeeze Breakout (Up)":      (_find_all_bb_squeeze_breakout_up,   "bullish"),
    "BB Squeeze Breakout (Down)":    (_find_all_bb_squeeze_breakout_down, "bearish"),
    "Bullish Engulfing":             (_find_all_bullish_engulfing,     "bullish"),
    "Bearish Engulfing":             (_find_all_bearish_engulfing,     "bearish"),
    "Hammer":                        (_find_all_hammer,                "bullish"),
    "Doji":                          (_find_all_doji,                  "neutral"),
    "Morning Star":                  (_find_all_morning_star,          "bullish"),
    "Evening Star":                  (_find_all_evening_star,          "bearish"),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Compute outcome for pattern occurrences
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_outcomes(
    df: pd.DataFrame,
    signal_indices: list[int],
    holding_days: int,
    pattern_type: str,
) -> list[TradeOutcome]:
    """
    For each signal index, measure the return over the next `holding_days`
    trading days.

    A "win" for bullish patterns = price went up.
    A "win" for bearish patterns = price went down (i.e., the signal
    correctly predicted the direction).
    """
    trades = []
    close = df["Close"].values
    dates = df.index

    for idx in signal_indices:
        exit_idx = idx + holding_days
        if exit_idx >= len(df):
            continue  # Not enough future data to measure

        entry_price = close[idx]
        exit_price = close[exit_idx]

        if entry_price == 0:
            continue

        return_pct = ((exit_price - entry_price) / entry_price) * 100

        # Win definition depends on pattern type
        if pattern_type == "bullish":
            is_win = return_pct > 0
        elif pattern_type == "bearish":
            is_win = return_pct < 0  # Bearish prediction correct if price fell
        else:
            is_win = abs(return_pct) < 2  # Neutral patterns: no big move = correct

        trades.append(TradeOutcome(
            entry_date=dates[idx].strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 2),
            exit_date=dates[exit_idx].strftime("%Y-%m-%d"),
            exit_price=round(exit_price, 2),
            holding_days=holding_days,
            return_pct=round(return_pct, 2),
            is_win=is_win,
        ))

    return trades


def _build_result(
    symbol: str,
    pattern_name: str,
    pattern_type: str,
    holding_days: int,
    trades: list[TradeOutcome],
) -> BacktestResult:
    """Compute aggregate statistics from a list of trade outcomes."""
    total = len(trades)
    if total == 0:
        return BacktestResult(
            symbol=symbol,
            pattern_name=pattern_name,
            pattern_type=pattern_type,
            holding_period=holding_days,
            total_occurrences=0,
            wins=0, losses=0,
            win_rate=0.0,
            avg_gain_pct=0.0, avg_loss_pct=0.0,
            overall_avg_return_pct=0.0,
            expectancy_pct=0.0,
            max_gain_pct=0.0, max_loss_pct=0.0,
            median_return_pct=0.0,
            sharpe_ratio=0.0,
            trades=[],
            backtest_time=datetime.now().isoformat(),
        )

    returns = [t.return_pct for t in trades]
    winning = [r for r in returns if r > 0]
    losing = [r for r in returns if r <= 0]

    wins = sum(1 for t in trades if t.is_win)
    losses = total - wins
    win_rate = wins / total

    avg_gain = np.mean(winning) if winning else 0.0
    avg_loss = np.mean(losing) if losing else 0.0
    overall_avg = np.mean(returns)
    median_return = float(np.median(returns))

    # Expectancy: expected $ return per trade (as %)
    wr = win_rate
    expectancy = (wr * avg_gain) + ((1 - wr) * avg_loss)

    # Sharpe-like ratio: mean / std (annualized from holding period)
    returns_arr = np.array(returns)
    std = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 1.0
    sharpe = (overall_avg / std) if std > 0 else 0.0

    return BacktestResult(
        symbol=symbol,
        pattern_name=pattern_name,
        pattern_type=pattern_type,
        holding_period=holding_days,
        total_occurrences=total,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 4),
        avg_gain_pct=round(float(avg_gain), 2),
        avg_loss_pct=round(float(avg_loss), 2),
        overall_avg_return_pct=round(float(overall_avg), 2),
        expectancy_pct=round(float(expectancy), 2),
        max_gain_pct=round(float(max(returns)), 2),
        max_loss_pct=round(float(min(returns)), 2),
        median_return_pct=round(median_return, 2),
        sharpe_ratio=round(float(sharpe), 3),
        trades=trades,
        backtest_time=datetime.now().isoformat(),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API: Backtest a single pattern
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_pattern(
    df: pd.DataFrame,
    pattern_name: str,
    symbol: str = "",
    holding_days: int = 20,
) -> BacktestResult:
    """
    Backtest a SINGLE pattern on a stock's historical data.

    Args:
        df:            OHLCV DataFrame (from nse_fetcher.fetch_ohlcv)
        pattern_name:  Must match a key in PATTERN_REGISTRY
        symbol:        Ticker symbol (for labeling)
        holding_days:  Days to hold after signal (default 20 = ~1 month)

    Returns:
        BacktestResult with full statistics and trade list
    """
    if pattern_name not in PATTERN_REGISTRY:
        logger.error("Unknown pattern '%s'. Available: %s",
                      pattern_name, list(PATTERN_REGISTRY.keys()))
        return BacktestResult(
            symbol=symbol, pattern_name=pattern_name, pattern_type="unknown",
            holding_period=holding_days, total_occurrences=0,
            wins=0, losses=0, win_rate=0, avg_gain_pct=0, avg_loss_pct=0,
            overall_avg_return_pct=0, expectancy_pct=0, max_gain_pct=0,
            max_loss_pct=0, median_return_pct=0, sharpe_ratio=0,
            backtest_time=datetime.now().isoformat(),
        )

    finder_fn, pattern_type = PATTERN_REGISTRY[pattern_name]

    # Ensure indicators are computed
    df = _ensure_indicators(df)

    # Find all historical occurrences
    signal_indices = finder_fn(df)

    # Compute outcomes
    trades = _compute_outcomes(df, signal_indices, holding_days, pattern_type)

    result = _build_result(symbol, pattern_name, pattern_type, holding_days, trades)

    logger.info(
        "Backtest %s on %s: %d occurrences, %.0f%% win rate, %.2f%% avg return (%d-day hold)",
        pattern_name, symbol, result.total_occurrences,
        result.win_rate * 100, result.overall_avg_return_pct, holding_days,
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API: Backtest ALL patterns on a stock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest_all_patterns(
    df: pd.DataFrame,
    symbol: str = "",
    holding_days_list: Optional[list[int]] = None,
) -> dict:
    """
    Backtest EVERY registered pattern on a stock across multiple
    holding periods.

    Args:
        df:                 OHLCV DataFrame
        symbol:             Ticker symbol
        holding_days_list:  List of holding periods to test
                            Default: [5, 10, 20, 30]

    Returns:
        {
            "symbol": str,
            "backtest_time": str,
            "results": {
                "Golden Cross": {
                    5:  BacktestResult,
                    10: BacktestResult,
                    20: BacktestResult,
                    30: BacktestResult,
                },
                ...
            },
            "best_patterns": list[dict],  # Top patterns sorted by win rate
            "summary": dict,              # Aggregate stats
        }
    """
    if holding_days_list is None:
        holding_days_list = [5, 10, 20, 30]

    # Compute indicators once
    df = _ensure_indicators(df)

    results = {}
    all_results_flat = []

    for pattern_name, (finder_fn, pattern_type) in PATTERN_REGISTRY.items():
        signal_indices = finder_fn(df)
        results[pattern_name] = {}

        for days in holding_days_list:
            trades = _compute_outcomes(df, signal_indices, days, pattern_type)
            bt = _build_result(symbol, pattern_name, pattern_type, days, trades)
            results[pattern_name][days] = bt
            all_results_flat.append(bt)

    # Find best patterns at primary holding period (20 days)
    primary = holding_days_list[2] if len(holding_days_list) >= 3 else holding_days_list[0]
    primary_results = [
        results[p][primary] for p in results if primary in results[p]
    ]
    # Filter to patterns with at least 3 occurrences for statistical relevance
    meaningful = [r for r in primary_results if r.total_occurrences >= 3]
    best_patterns = sorted(meaningful, key=lambda r: r.win_rate, reverse=True)

    # Summary
    total_patterns = len([r for r in primary_results if r.total_occurrences > 0])
    total_trades = sum(r.total_occurrences for r in primary_results)
    avg_wr = np.mean([r.win_rate for r in meaningful]) if meaningful else 0

    output = {
        "symbol": symbol,
        "backtest_time": datetime.now().isoformat(),
        "results": {
            pname: {
                days: results[pname][days].to_dict()
                for days in holding_days_list
            }
            for pname in results
        },
        "best_patterns": [
            {
                "pattern": r.pattern_name,
                "type": r.pattern_type,
                "occurrences": r.total_occurrences,
                "win_rate": r.win_rate,
                "avg_return": r.overall_avg_return_pct,
                "expectancy": r.expectancy_pct,
                "holding_days": primary,
                "summary": r.summary_text,
            }
            for r in best_patterns[:5]
        ],
        "summary": {
            "patterns_with_data": total_patterns,
            "total_trade_signals": total_trades,
            "avg_win_rate": round(float(avg_wr), 1),
            "holding_periods_tested": holding_days_list,
        },
    }

    logger.info(
        "Full backtest for %s: %d patterns with data, %d total signals, %.0f%% avg WR",
        symbol, total_patterns, total_trades, avg_wr * 100,
    )
    return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC API: Quick win rate lookup (used by Chart Agent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_win_rate(
    df: pd.DataFrame,
    pattern_name: str,
    symbol: str = "",
    holding_days: int = 20,
) -> dict:
    """
    Quick lookup: get the win rate for a specific pattern on a stock.
    Returns a lightweight dict (no full trade list).

    This is what the Chart Agent calls when it detects a pattern and
    needs to attach the back-tested win rate to the signal.

    Returns:
        {
            "symbol": str,
            "pattern": str,
            "win_rate": float,       # 0-100
            "occurrences": int,
            "avg_return_pct": float,
            "holding_days": int,
            "summary": str,          # Plain English
            "reliable": bool,        # True if >= 5 occurrences
        }
    """
    bt = backtest_pattern(df, pattern_name, symbol, holding_days)

    return {
        "symbol": symbol,
        "pattern": pattern_name,
        "win_rate": bt.win_rate,
        "occurrences": bt.total_occurrences,
        "avg_return_pct": bt.overall_avg_return_pct,
        "avg_gain_pct": bt.avg_gain_pct,
        "avg_loss_pct": bt.avg_loss_pct,
        "max_gain_pct": bt.max_gain_pct,
        "max_loss_pct": bt.max_loss_pct,
        "expectancy_pct": bt.expectancy_pct,
        "holding_days": holding_days,
        "summary": bt.summary_text,
        "reliable": bt.total_occurrences >= 5,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators if not already present."""
    if "SMA50" not in df.columns:
        from data.ta_processor import compute_indicators
        df = compute_indicators(df)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.nse_fetcher import fetch_ohlcv

    print("=" * 70)
    print("BACKTEST ENGINE — Smoke Test")
    print("=" * 70)

    ticker = "RELIANCE"
    print(f"\n1. Fetching 2Y OHLCV for {ticker}...")
    df = fetch_ohlcv(ticker)
    print(f"   → {len(df)} candles\n")

    # Single pattern backtest
    print("2. Backtesting 'Bullish MACD Crossover' (20-day hold)...")
    wr = get_win_rate(df, "Bullish MACD Crossover", symbol=ticker)
    print(f"   → Occurrences: {wr['occurrences']}")
    print(f"   → Win Rate: {wr['win_rate']:.0f}%")
    print(f"   → Avg Return: {wr['avg_return_pct']:.2f}%")
    print(f"   → Reliable: {wr['reliable']}")
    print(f"   → {wr['summary']}\n")

    # Full backtest
    print("3. Full backtest — all patterns × all holding periods...")
    full = backtest_all_patterns(df, symbol=ticker)
    print(f"   → Patterns with data: {full['summary']['patterns_with_data']}")
    print(f"   → Total signals: {full['summary']['total_trade_signals']}")
    print(f"   → Avg WR: {full['summary']['avg_win_rate']:.0f}%\n")

    print("   TOP 5 PATTERNS (20-day hold, ≥3 occurrences):")
    print("   " + "-" * 65)
    for i, bp in enumerate(full["best_patterns"], 1):
        print(f"   {i}. {bp['pattern']} — "
              f"WR: {bp['win_rate']:.0f}% | "
              f"Avg: {bp['avg_return']:.2f}% | "
              f"Signals: {bp['occurrences']} | "
              f"Type: {bp['type']}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
