"""convergence_agent.py
=================================
Convergence scoring agent.

Combines chart, radar, and sentiment agent outputs into a single score:
(0.35 * chart_confidence)
+ (0.30 * insider_signal_strength)
+ (0.20 * earnings_beat_normalized)
+ (0.15 * sentiment_normalized)
"""

import logging
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CHART_WEIGHT = 0.35
INSIDER_WEIGHT = 0.30
EARNINGS_WEIGHT = 0.20
SENTIMENT_WEIGHT = 0.15

STRONG_BUY_THRESHOLD = 0.75
WATCH_THRESHOLD = 0.50

logger = logging.getLogger(__name__)

BASE_WEIGHTS = {
    "chart": CHART_WEIGHT,
    "insider": INSIDER_WEIGHT,
    "earnings": EARNINGS_WEIGHT,
    "sentiment": SENTIMENT_WEIGHT,
}


def _clamp_01(value: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _normalize_chart_confidence(chart_data: dict[str, Any]) -> float:
    """Normalize chart confidence to [0, 1], supporting both 0-1 and 0-100 sources."""
    raw = float(chart_data.get("chart_confidence", 0.0))
    if raw > 1.0:
        raw = raw / 100.0
    return _clamp_01(raw)


def _normalize_earnings_beat_pct(radar_data: dict[str, Any]) -> float:
    """Normalize earnings beat into [0, 1] for convergence scoring.

    Expected format in CONTEXT.md is fractional percent (e.g., 0.204 for 20.4%).
    If upstream sends whole percent (20.4), this function scales it safely.
    """
    beat = float(radar_data.get("earnings_beat_pct", 0.0))
    if beat > 1.0:
        beat = beat / 100.0
    if beat < 0:
        return 0.0
    return _clamp_01(beat)


def _normalize_sentiment(sentiment_data: dict[str, Any]) -> float:
    """Convert sentiment score in [-1, 1] to normalized [0, 1]."""
    score = float(sentiment_data.get("sentiment_score", 0.0))
    score = max(-1.0, min(1.0, score))
    return _clamp_01((score + 1.0) / 2.0)


def _compute_label(score: float) -> str:
    """Map convergence score to required label buckets."""
    if score >= STRONG_BUY_THRESHOLD:
        return "STRONG BUY SIGNAL"
    if score >= WATCH_THRESHOLD:
        return "WATCH"
    return "WEAK"


def _adaptive_weights(
    chart_conf: float,
    insider_strength: float,
    earnings_norm: float,
    sentiment_raw_score: float,
) -> dict[str, float]:
    """Compute adaptive weights over available signals.

    Availability follows the requested rules:
      - chart available when chart_confidence > 0
      - insider available when insider_signal_strength > 0
      - earnings available when earnings_beat_pct normalized > 0
      - sentiment available when raw sentiment_score > 0
    """
    available: list[str] = []
    if chart_conf > 0:
        available.append("chart")
    if insider_strength > 0:
        available.append("insider")
    if earnings_norm > 0:
        available.append("earnings")
    if sentiment_raw_score > 0:
        available.append("sentiment")

    # Fallback: no positive signals available.
    if not available:
        return {
            "chart": 0.0,
            "insider": 0.0,
            "earnings": 0.0,
            "sentiment": 0.0,
        }

    # Requested special case: only chart + earnings => 65/35.
    if set(available) == {"chart", "earnings"}:
        return {
            "chart": 0.65,
            "insider": 0.0,
            "earnings": 0.35,
            "sentiment": 0.0,
        }

    # If all 4 are available, keep original base weights.
    if len(available) == 4:
        return dict(BASE_WEIGHTS)

    # General redistribution: normalize base weights over available signals.
    total = sum(BASE_WEIGHTS[k] for k in available)
    if total <= 0:
        return {
            "chart": 0.0,
            "insider": 0.0,
            "earnings": 0.0,
            "sentiment": 0.0,
        }

    adaptive = {
        "chart": 0.0,
        "insider": 0.0,
        "earnings": 0.0,
        "sentiment": 0.0,
    }
    for key in available:
        adaptive[key] = BASE_WEIGHTS[key] / total
    return adaptive


def _build_signals_present(
    chart_data: dict[str, Any],
    radar_data: dict[str, Any],
    sentiment_data: dict[str, Any],
) -> list[str]:
    """Build the signals_present list required by the output schema."""
    signals: list[str] = []

    if _normalize_chart_confidence(chart_data) > 0:
        signals.append("chart_pattern")

    if bool(radar_data.get("has_insider_buying", False)):
        signals.append("insider_buying")

    if bool(radar_data.get("has_earnings_beat", False)):
        signals.append("earnings_beat")

    sentiment_label = str(sentiment_data.get("sentiment_label", "Neutral"))
    if sentiment_label == "Positive" or float(sentiment_data.get("sentiment_score", 0.0)) > 0:
        signals.append("positive_sentiment")

    return signals


def build_convergence(
    ticker: str,
    chart_data: dict[str, Any],
    radar_data: dict[str, Any],
    sentiment_data: dict[str, Any],
) -> dict[str, Any]:
    """Compute convergence score from three upstream agent outputs.

    Args:
        ticker: Stock ticker (without .NS).
        chart_data: Output dict from chart agent.
        radar_data: Output dict from radar agent.
        sentiment_data: Output dict from sentiment agent.

    Returns:
        Dict matching the exact convergence_agent schema defined in CONTEXT.md.
    """
    normalized_ticker = ticker.strip().upper().replace(".NS", "")

    patterns = chart_data.get("patterns", [])
    raw_chart_conf = float(chart_data.get("chart_confidence", 0.0))
    chart_conf_for_guard = raw_chart_conf / 100.0 if raw_chart_conf > 1.0 else raw_chart_conf
    radar_composite = float(radar_data.get("composite_score", 0.0))

    # Detect silent pipeline failure — no real data retrieved.
    if (
        len(patterns) == 0
        and abs(chart_conf_for_guard - 0.50) < 0.001
        and radar_composite == 0.0
    ):
        return {
            "ticker": normalized_ticker,
            "convergence_score": 0.0,
            "convergence_label": "NO DATA",
            "signal_breakdown": {},
            "signals_present": [],
            "chart_data": chart_data,
            "radar_data": radar_data,
            "sentiment_data": sentiment_data,
            "timestamp": datetime.now().isoformat(),
            "pipeline_failed": True,
        }

    chart_conf = _normalize_chart_confidence(chart_data)
    insider_strength = _clamp_01(float(radar_data.get("insider_signal_strength", 0.0)))
    earnings_norm = _normalize_earnings_beat_pct(radar_data)
    sentiment_raw_score = float(sentiment_data.get("sentiment_score", 0.0))
    sentiment_norm = _normalize_sentiment(sentiment_data)

    weights = _adaptive_weights(
        chart_conf=chart_conf,
        insider_strength=insider_strength,
        earnings_norm=earnings_norm,
        sentiment_raw_score=sentiment_raw_score,
    )

    chart_contribution = weights["chart"] * chart_conf
    insider_contribution = weights["insider"] * insider_strength
    earnings_contribution = weights["earnings"] * earnings_norm
    sentiment_contribution = weights["sentiment"] * sentiment_norm

    convergence_score = _clamp_01(
        chart_contribution
        + insider_contribution
        + earnings_contribution
        + sentiment_contribution
    )

    label = _compute_label(convergence_score)

    # Score and label must agree — never allow contradiction.
    if convergence_score >= 0.75 and "SELL" in label:
        label = "WATCH"
    if convergence_score == 0.0:
        label = "NO DATA"

    result = {
        "ticker": normalized_ticker,
        "convergence_score": round(convergence_score, 4),
        "convergence_label": label,
        "signal_breakdown": {
            "chart_contribution": round(chart_contribution, 4),
            "insider_contribution": round(insider_contribution, 4),
            "earnings_contribution": round(earnings_contribution, 4),
            "sentiment_contribution": round(sentiment_contribution, 4),
        },
        "signals_present": _build_signals_present(chart_data, radar_data, sentiment_data),
        "chart_data": chart_data,
        "radar_data": radar_data,
        "sentiment_data": sentiment_data,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info(
        "Convergence %s: %.4f (%s) [w chart=%.2f insider=%.2f earnings=%.2f sentiment=%.2f]",
        normalized_ticker,
        result["convergence_score"],
        result["convergence_label"],
        weights["chart"],
        weights["insider"],
        weights["earnings"],
        weights["sentiment"],
    )
    return result


def smoke_test() -> dict[str, Any]:
    """Run a local smoke test with synthetic but schema-valid upstream payloads."""
    chart_data = {
        "ticker": "RELIANCE",
        "chart_confidence": 0.80,
        "patterns": [{"name": "Golden Cross", "type": "bullish", "confidence": 80}],
    }
    radar_data = {
        "ticker": "RELIANCE",
        "composite_score": 0.72,
        "has_insider_buying": True,
        "insider_signal_strength": 0.85,
        "has_earnings_beat": True,
        "earnings_beat_pct": 0.204,
        "signals": [],
    }
    sentiment_data = {
        "ticker": "RELIANCE",
        "sentiment_score": 0.72,
        "sentiment_label": "Positive",
        "tone_shift": True,
        "tone_shift_direction": "positive",
        "key_sentences": [],
        "document_type": "earnings_call",
    }
    return build_convergence("RELIANCE", chart_data, radar_data, sentiment_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(smoke_test())
