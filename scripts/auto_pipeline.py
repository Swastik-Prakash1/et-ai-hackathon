"""Autonomous signal pipeline runner.

Runs continuously every 60 seconds with ZERO human input:
  STEP 1 — Detect:  Scan all 6 stocks, detect patterns, compute convergence scores
  STEP 2 — Enrich:  Query ChromaDB RAG for each stock with signals found
  STEP 3 — Alert:   Call Groq to generate plain-English alert for stocks scoring > 50

Demonstrates multi-step agentic execution with timestamped proof.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve project root so imports work when invoked as `python -m scripts.auto_pipeline`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.chart_agent import analyze_stock
from agents.convergence_agent import build_convergence
from agents.radar_agent import scan_stock
from agents.reasoning_agent import generate_reasoning
from agents.sentiment_agent import analyze_sentiment
from data.nse_fetcher import fetch_company_info, fetch_earnings
from rag.retriever import build_reasoning_context

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WATCHLIST = ["SBIN", "OIL", "BHARTIARTL", "HINDUNILVR", "HCLTECH", "MARUTI"]
SCAN_INTERVAL_SECONDS = 60
CONVERGENCE_ALERT_THRESHOLD = 0.50  # score > 50 triggers Groq alert
ALERTS_FILE = PROJECT_ROOT / "alerts_log.json"

logger = logging.getLogger("auto_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    """Current timestamp formatted for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    """Print a timestamped pipeline log line to stdout AND logger."""
    line = f"[{_ts()}] {message}"
    print(line, flush=True)
    logger.info(message)


import math

def _sanitize_value(v: Any) -> Any:
    """Recursively sanitize: replace NaN/Inf floats with None at any depth."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_sanitize_value(i) for i in v]
    return v


def _sanitize_float(v: Any) -> Any:
    """Sanitize a single float value: replace NaN/Inf with None."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _load_alerts() -> list[dict[str, Any]]:
    """Load the full alert history from alerts_log.json."""
    if not ALERTS_FILE.exists():
        return []
    try:
        with ALERTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_alerts(alerts: list[dict[str, Any]]) -> None:
    """Atomically save the full alert list to alerts_log.json."""
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sanitized = [_sanitize_value(a) for a in alerts]
    tmp = ALERTS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(ALERTS_FILE)


def _build_sentiment_text(ticker: str) -> tuple[str, str | None]:
    """Build sentiment text from live company + earnings data."""
    info = fetch_company_info(ticker)
    earnings = fetch_earnings(ticker)

    company = str(info.get("longName") or info.get("shortName") or ticker)
    sector = str(info.get("sector") or "Unknown sector")
    pe = info.get("trailingPE")
    rev_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")

    parts = [
        f"{company} operates in {sector}.",
        f"Trailing PE is {pe}." if pe is not None else "Trailing PE is not available.",
        f"Revenue growth is {rev_growth}." if rev_growth is not None else "Revenue growth is not available.",
        f"Earnings growth is {earnings_growth}." if earnings_growth is not None else "Earnings growth is not available.",
    ]

    latest_beat = earnings.get("latest_beat")
    latest_beat_pct = earnings.get("latest_beat_pct")
    if latest_beat is not None and latest_beat_pct is not None:
        outcome = "beat" if bool(latest_beat) else "missed"
        parts.append(f"Latest quarter {outcome} estimates by {float(latest_beat_pct):.2f} percent.")

    current_text = " ".join(parts)

    previous_text: str | None = None
    qdf = earnings.get("quarterly_earnings")
    if hasattr(qdf, "empty") and not qdf.empty and getattr(qdf, "shape", (0, 0))[1] >= 2:
        cols = list(qdf.columns)
        prev_col = cols[1]
        prev_parts = [f"{company} previous quarter summary."]
        for row_name, label in (("Total Revenue", "revenue"), ("Net Income", "net income")):
            try:
                val = float(qdf.loc[row_name, prev_col])
                prev_parts.append(f"Previous quarter {label} was {val}.")
            except Exception:
                continue
        previous_text = " ".join(prev_parts)

    return current_text, previous_text


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DETECT
# ═══════════════════════════════════════════════════════════════════════════

def _step1_detect(cycle: int) -> list[dict[str, Any]]:
    """Scan all watchlist stocks: patterns + convergence scores.

    Returns a list of per-stock result dicts.
    """
    results: list[dict[str, Any]] = []

    for ticker in WATCHLIST:
        try:
            # Radar scan
            radar_profile = scan_stock(ticker)
            radar_data = radar_profile.to_dict() if hasattr(radar_profile, "to_dict") else dict(radar_profile)

            # Chart analysis (fast, no VLM)
            chart_data = analyze_stock(ticker, holding_days=20, chart_days=120, use_vlm=False)

            # Sentiment
            sentiment_text, previous_text = _build_sentiment_text(ticker)
            sentiment_data = analyze_sentiment(
                ticker=ticker,
                text=sentiment_text,
                document_type="earnings_call",
                previous_text=previous_text,
            )

            # Convergence
            convergence_data = build_convergence(
                ticker=ticker,
                chart_data=chart_data,
                radar_data=radar_data,
                sentiment_data=sentiment_data,
            )

            # Extract top pattern for log line
            patterns = chart_data.get("patterns", [])
            top_pattern_name = "No pattern"
            top_confidence = 0
            top_win_rate = 0.0
            if patterns:
                top = patterns[0]
                top_pattern_name = str(top.get("name", top.get("pattern_name", "Unknown")))
                top_confidence = int(top.get("confidence", 0))
                top_win_rate = float(top.get("win_rate", 0))

            conv_score = convergence_data.get("convergence_score", 0)
            conv_pct = round(conv_score * 100)

            _log(
                f"STEP 1 — Detected: {ticker} "
                f"{top_pattern_name} ({top_confidence}% conf, {top_win_rate:.0%} win rate) "
                f"| Convergence: {conv_pct}%"
            )

            results.append({
                "ticker": ticker,
                "chart_data": chart_data,
                "radar_data": radar_data,
                "sentiment_data": sentiment_data,
                "convergence_data": convergence_data,
                "top_pattern": top_pattern_name,
                "top_confidence": top_confidence,
                "top_win_rate": top_win_rate,
            })

        except Exception as exc:
            _log(f"STEP 1 — {ticker} scan failed: {exc}")
            logger.exception("STEP 1 failure for %s", ticker)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — ENRICH
# ═══════════════════════════════════════════════════════════════════════════

def _step2_enrich(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Query ChromaDB RAG for each stock with detected signals."""
    enriched: list[dict[str, Any]] = []

    for item in results:
        ticker = item["ticker"]
        try:
            top_pattern = item.get("top_pattern", "")
            rag_query = f"{ticker} {top_pattern} insider buying earnings sentiment chart pattern"
            rag_context = build_reasoning_context(rag_query, top_k=5)
            doc_count = len(rag_context)

            _log(f"STEP 2 — Enriched: {ticker} — Retrieved {doc_count} relevant documents from ChromaDB")

            enriched.append({**item, "rag_context": rag_context, "rag_doc_count": doc_count})

        except Exception as exc:
            _log(f"STEP 2 — {ticker} RAG enrichment failed: {exc}")
            logger.exception("STEP 2 failure for %s", ticker)
            enriched.append({**item, "rag_context": [], "rag_doc_count": 0})

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — ALERT
# ═══════════════════════════════════════════════════════════════════════════

def _step3_alert(cycle: int, enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate Groq alerts for stocks with convergence_score > 50%.

    Returns list of generated alert dicts.
    """
    alerts_generated: list[dict[str, Any]] = []

    for item in enriched:
        ticker = item["ticker"]
        convergence_data = item["convergence_data"]
        conv_score = convergence_data.get("convergence_score", 0)

        if conv_score <= CONVERGENCE_ALERT_THRESHOLD:
            _log(f"STEP 3 — {ticker} score {round(conv_score * 100)}% ≤ 50% — skipped alert generation")
            continue

        try:
            rag_context = item.get("rag_context", [])
            reasoning_data = generate_reasoning(ticker, convergence_data, rag_context=rag_context)

            # Extract action plan details for log line
            action_plan = reasoning_data.get("action_plan", {})
            entry_range = action_plan.get("entry_price_range", {})
            entry_price = entry_range.get("min") or entry_range.get("max") or "N/A"
            stop_loss = (action_plan.get("stop_loss") or {}).get("price", "N/A")
            target_price = (action_plan.get("target_price") or {}).get("price", "N/A")
            time_horizon = action_plan.get("time_horizon_days", "N/A")

            # Format for log
            entry_str = f"₹{entry_price:,.0f}" if isinstance(entry_price, (int, float)) else str(entry_price)
            sl_str = f"₹{stop_loss:,.0f}" if isinstance(stop_loss, (int, float)) else str(stop_loss)
            tgt_str = f"₹{target_price:,.0f}" if isinstance(target_price, (int, float)) else str(target_price)

            _log(
                f"STEP 3 — Alert generated: {ticker} "
                f"Entry {entry_str}, Stop Loss {sl_str}, Target {tgt_str}, "
                f"Horizon {time_horizon} days"
            )

            # Build full alert record
            signals = item.get("radar_data", {}).get("signals", [])
            top_headline = ""
            if isinstance(signals, list) and signals:
                first = signals[0]
                if isinstance(first, dict):
                    top_headline = str(first.get("headline", ""))

            alert = {
                "timestamp": _ts(),
                "cycle": cycle,
                "ticker": ticker,
                "convergence_score": round(conv_score, 3),
                "convergence_label": convergence_data.get("convergence_label", ""),
                "action": reasoning_data.get("action", "WATCH"),
                "confidence_plain": reasoning_data.get("confidence_plain", ""),
                "entry_price": _sanitize_float(entry_price if isinstance(entry_price, (int, float)) else None),
                "stop_loss": _sanitize_float(stop_loss if isinstance(stop_loss, (int, float)) else None),
                "target_price": _sanitize_float(target_price if isinstance(target_price, (int, float)) else None),
                "time_horizon": _sanitize_float(time_horizon if isinstance(time_horizon, (int, float)) else None),
                "patterns": [
                    {
                        "name": item.get("top_pattern", ""),
                        "confidence": item.get("top_confidence", 0),
                        "win_rate": item.get("top_win_rate", 0),
                    }
                ],
                "explanation": reasoning_data.get("explanation", ""),
                "key_points": reasoning_data.get("key_points", []),
                "sources": reasoning_data.get("sources", []),
                "top_signal_headline": top_headline,
                "rag_context_count": item.get("rag_doc_count", 0),
                "pipeline_steps": ["detect", "enrich", "alert"],
                "autonomous": True,
            }

            alerts_generated.append(alert)

        except Exception as exc:
            _log(f"STEP 3 — {ticker} alert generation failed: {exc}")
            logger.exception("STEP 3 failure for %s", ticker)

    return alerts_generated


# ═══════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_autonomous_pipeline() -> None:
    """Continuously execute autonomous 3-step pipeline every 60 seconds."""
    cycle = 0
    _log("Auto pipeline booted — fully autonomous, zero human input")
    _log(f"Watchlist: {', '.join(WATCHLIST)} | Interval: {SCAN_INTERVAL_SECONDS}s | Alert threshold: >{int(CONVERGENCE_ALERT_THRESHOLD * 100)}%")

    while True:
        cycle += 1
        cycle_started = time.perf_counter()

        # ── Cycle header ──────────────────────────────────────────────
        stock_list = ", ".join(WATCHLIST)
        _log(f"═══ AUTONOMOUS CYCLE #{cycle} STARTING ═══")
        _log(f"Scanning {len(WATCHLIST)} stocks: {stock_list}")

        # ── STEP 1 — Detect ──────────────────────────────────────────
        detect_results = _step1_detect(cycle)

        # ── STEP 2 — Enrich ──────────────────────────────────────────
        enriched_results = _step2_enrich(detect_results)

        # ── STEP 3 — Alert ───────────────────────────────────────────
        new_alerts = _step3_alert(cycle, enriched_results)

        # ── Persist alerts ────────────────────────────────────────────
        if new_alerts:
            all_alerts = _load_alerts()
            all_alerts.extend(new_alerts)
            _save_alerts(all_alerts)

        # ── Cycle footer ──────────────────────────────────────────────
        elapsed = time.perf_counter() - cycle_started
        alert_count = len(new_alerts)
        sleep_for = max(0.0, SCAN_INTERVAL_SECONDS - elapsed)
        _log(
            f"═══ CYCLE #{cycle} COMPLETE — {alert_count} alert{'s' if alert_count != 1 else ''} "
            f"generated, next cycle in {int(sleep_for)}s ═══"
        )

        time.sleep(sleep_for)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_autonomous_pipeline()
