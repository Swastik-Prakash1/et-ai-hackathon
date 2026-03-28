"""
chart_agent.py — Chart Pattern Intelligence Agent
===================================================
Agent 1 of 5. The entry point for technical analysis on any NSE stock.

Pipeline:
  1. Fetch 2Y OHLCV data (nse_fetcher)
  2. Compute all technical indicators (ta_processor)
  3. Detect active patterns with confidence scores (ta_processor)
  4. Render annotated candlestick chart as PNG (chart_renderer)
  5. Send chart PNG to Gemini Vision API for visual pattern confirmation
  6. Run backtest_engine to get per-stock historical win rate for each pattern
  7. Return structured JSON with everything the Convergence Agent needs

Output schema:
  {
    "symbol": str,
    "analysis_time": str,
    "chart_path": str,           # path to annotated PNG
    "patterns": [
      {
        "pattern_name": str,
        "pattern_type": str,
        "confidence": float,     # 0-100 from ta_processor
        "explanation": str,      # plain English
        "win_rate": float,       # back-tested on THIS stock
        "occurrences": int,
        "avg_return_pct": float,
        "suggested_action": str,
        "reliable": bool,        # True if >= 5 historical occurrences
      }
    ],
    "vlm_analysis": dict,        # Gemini Vision output
    "overall_bias": str,
    "chart_confidence": float,   # 0-100
    "indicators": dict,
    "backtest_summary": dict,
  }
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # Exact string — do NOT change

VLM_PROMPT = """You are a senior technical analyst examining an NSE (India) stock chart.
Analyze this candlestick chart carefully. Identify ALL technical patterns visible in the chart.

Return your analysis as valid JSON with this exact structure:
{
  "patterns_visible": [
    {
      "name": "pattern name",
      "type": "bullish or bearish or neutral",
      "confidence": 0 to 100,
      "location": "describe where on the chart this pattern appears"
    }
  ],
  "trend_direction": "uptrend or downtrend or sideways",
  "key_observations": [
    "observation 1",
    "observation 2"
  ],
  "support_level": null or price number,
  "resistance_level": null or price number,
  "overall_signal": "bullish or bearish or neutral",
  "risk_level": "low or medium or high"
}

Be specific. Reference the moving averages (blue=SMA50, orange=SMA200), 
Bollinger Bands (gray dotted), RSI subplot, and MACD subplot if visible.
Return ONLY valid JSON, no markdown, no explanation outside the JSON."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Run the full Chart Agent pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_stock(
    symbol: str,
    holding_days: int = 20,
    chart_days: int = 120,
    use_vlm: bool = True,
) -> dict:
    """
    Run the FULL Chart Agent pipeline on a single NSE stock.

    Args:
        symbol:       NSE ticker (e.g. "RELIANCE", "TATAMOTORS")
        holding_days: Days for win rate calculation (default 20 = ~1 month)
        chart_days:   Trading days to show on chart (default 120 = ~6 months)
        use_vlm:      Whether to call Gemini Vision (set False to skip, saves quota)

    Returns:
        Complete analysis dict — patterns, win rates, chart path, VLM analysis.
    """
    from data.nse_fetcher import fetch_ohlcv
    from data.ta_processor import compute_indicators, scan_all_patterns
    from data.backtest_engine import get_win_rate
    from data.chart_renderer import render_chart

    logger.info("Chart Agent: analyzing %s", symbol)

    # ── Step 1: Fetch OHLCV ─────────────────────────────────
    df = fetch_ohlcv(symbol)
    if df.empty:
        return _error_result(symbol, "Failed to fetch OHLCV data")

    # ── Step 2: Compute indicators ──────────────────────────
    df = compute_indicators(df)

    # ── Step 3: Detect patterns ─────────────────────────────
    scan = scan_all_patterns(df, symbol=symbol)
    patterns = scan.get("patterns", [])

    # ── Step 4: Backtest each detected pattern ──────────────
    enriched_patterns = []
    for p in patterns:
        pname = p["pattern_name"]
        # Map ta_processor pattern names to backtest_engine registry names
        bt_name = _map_pattern_name(pname)
        if bt_name:
            wr = get_win_rate(df, bt_name, symbol=symbol, holding_days=holding_days)
            p["win_rate"] = wr["win_rate"]
            p["occurrences"] = wr["occurrences"]
            p["avg_return_pct"] = wr["avg_return_pct"]
            p["avg_gain_pct"] = wr["avg_gain_pct"]
            p["avg_loss_pct"] = wr["avg_loss_pct"]
            p["max_gain_pct"] = wr["max_gain_pct"]
            p["expectancy_pct"] = wr["expectancy_pct"]
            p["reliable"] = wr["reliable"]
            p["win_rate_summary"] = wr["summary"]
        else:
            # Pattern not in backtest registry — mark as untested
            p["win_rate"] = 0.0
            p["occurrences"] = 0
            p["avg_return_pct"] = 0.0
            p["reliable"] = False
            p["win_rate_summary"] = "No backtest data available for this pattern."

        enriched_patterns.append(p)

    # ── Step 5: Render annotated chart ──────────────────────
    chart_path = render_chart(
        df=df,
        symbol=symbol,
        patterns=enriched_patterns,
        indicators=scan.get("indicators"),
        last_n_days=chart_days,
    )

    # ── Step 6: VLM analysis (optional) ─────────────────────
    vlm_analysis = {}
    if use_vlm and chart_path and GEMINI_API_KEY:
        vlm_analysis = _run_vlm_analysis(chart_path)
    elif use_vlm and not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — skipping VLM analysis")
        vlm_analysis = {"skipped": True, "reason": "GEMINI_API_KEY not configured"}

    # ── Step 7: Assemble final result ───────────────────────
    current_price = float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0
    recent_low_20 = float(df["Low"].iloc[-20:].min()) if len(df) >= 20 else current_price * 0.95
    recent_high_20 = float(df["High"].iloc[-20:].max()) if len(df) >= 20 else current_price * 1.08

    result = {
        "symbol": symbol.upper(),
        "analysis_time": datetime.now().isoformat(),
        "chart_path": chart_path,
        "data_staleness_days": df.attrs.get("days_stale", 0),
        "last_trading_date": df.attrs.get("last_trading_date", "unknown"),
        "is_stale_data": df.attrs.get("is_stale", False),
        "current_price": round(current_price, 2),
        "recent_low_20": round(recent_low_20, 2),
        "recent_high_20": round(recent_high_20, 2),
        "patterns": enriched_patterns,
        "pattern_count": len(enriched_patterns),
        "bullish_count": scan.get("bullish_count", 0),
        "bearish_count": scan.get("bearish_count", 0),
        "overall_bias": scan.get("overall_bias", "neutral"),
        "chart_confidence": scan.get("chart_confidence", 50.0),
        "vlm_analysis": vlm_analysis,
        "indicators": scan.get("indicators", {}),
        "backtest_summary": {
            "holding_days": holding_days,
            "patterns_with_backtest": sum(1 for p in enriched_patterns if p.get("occurrences", 0) > 0),
            "reliable_patterns": sum(1 for p in enriched_patterns if p.get("reliable", False)),
        },
    }

    logger.info(
        "Chart Agent complete for %s: %d patterns, bias=%s, confidence=%.1f",
        symbol, len(enriched_patterns), result["overall_bias"], result["chart_confidence"],
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH: Analyze multiple stocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_multiple(
    symbols: list[str],
    holding_days: int = 20,
    use_vlm: bool = False,
) -> list[dict]:
    """
    Run Chart Agent on multiple stocks. VLM disabled by default for
    batch mode (rate limits). Returns list of analysis results sorted
    by chart_confidence descending.
    """
    import time
    results = []
    for i, sym in enumerate(symbols, 1):
        logger.info("Batch analysis %d/%d: %s", i, len(symbols), sym)
        try:
            r = analyze_stock(sym, holding_days=holding_days, use_vlm=use_vlm)
            results.append(r)
        except Exception as exc:
            logger.error("Failed to analyze %s: %s", sym, exc)
            results.append(_error_result(sym, str(exc)))
        time.sleep(0.5)  # Rate limit

    # Sort by signal strength
    results.sort(key=lambda r: r.get("chart_confidence", 0), reverse=True)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM: Gemini Vision chart analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_vlm_analysis(chart_path: str) -> dict:
    """
    Send chart PNG to Gemini Vision for visual pattern confirmation.

    Returns parsed JSON from the VLM, or error dict on failure.
    """
    text = ""
    try:
        from google import genai
        from google.genai import types
        import PIL.Image

        client = genai.Client(api_key=GEMINI_API_KEY)

        img = PIL.Image.open(chart_path)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[img, VLM_PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
        )

        # Parse response
        text = str(getattr(response, "text", "") or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        # Repair truncated output by clipping to the last complete object terminator.
        first_obj = text.find("{")
        last_obj = text.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            text = text[first_obj : last_obj + 1]

        vlm_result = json.loads(text)
        logger.info("VLM analysis complete: signal=%s", vlm_result.get("overall_signal", "?"))
        return vlm_result

    except json.JSONDecodeError as exc:
        logger.warning("VLM returned non-JSON response: %s", exc)
        return {"error": "VLM response was not valid JSON", "raw_text": text[:500]}
    except ImportError:
        logger.warning("google-genai not installed — skipping VLM")
        return {"skipped": True, "reason": "google-genai not installed"}
    except Exception as exc:
        logger.error("VLM analysis failed: %s", exc)
        return {"error": str(exc)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATTERN NAME MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ta_processor pattern names → backtest_engine PATTERN_REGISTRY names
_PATTERN_NAME_MAP = {
    "Golden Cross": "Golden Cross",
    "Death Cross": "Death Cross",
    "Bullish MACD Crossover": "Bullish MACD Crossover",
    "Bearish MACD Crossover": "Bearish MACD Crossover",
    "RSI Oversold": "RSI Oversold Entry",
    "RSI Overbought": "RSI Overbought Entry",
    "Bullish RSI Divergence": None,  # Not in backtest registry (needs swing detection)
    "Bearish RSI Divergence": None,
    "Bollinger Squeeze Breakout (Up)": "BB Squeeze Breakout (Up)",
    "Bollinger Squeeze Breakout (Down)": "BB Squeeze Breakout (Down)",
    "Bollinger Band Squeeze": None,  # Neutral squeeze, no directional backtest
    "Bullish Engulfing": "Bullish Engulfing",
    "Bearish Engulfing": "Bearish Engulfing",
    "Hammer": "Hammer",
    "Doji": "Doji",
    "Morning Star": "Morning Star",
    "Evening Star": "Evening Star",
}


def _map_pattern_name(ta_name: str) -> Optional[str]:
    """Map ta_processor pattern name to backtest_engine registry name."""
    return _PATTERN_NAME_MAP.get(ta_name)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _error_result(symbol: str, error_msg: str) -> dict:
    """Return a standardized error result."""
    return {
        "symbol": symbol.upper(),
        "analysis_time": datetime.now().isoformat(),
        "error": error_msg,
        "chart_path": "",
        "patterns": [],
        "pattern_count": 0,
        "bullish_count": 0,
        "bearish_count": 0,
        "overall_bias": "neutral",
        "chart_confidence": 0.0,
        "vlm_analysis": {},
        "indicators": {},
        "backtest_summary": {},
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("CHART AGENT — Smoke Test")
    print("=" * 70)

    ticker = "RELIANCE"
    print(f"\nRunning full Chart Agent pipeline on {ticker}...")
    print("(VLM disabled for smoke test — set GEMINI_API_KEY to enable)\n")

    result = analyze_stock(ticker, use_vlm=False)

    print(f"Symbol: {result['symbol']}")
    print(f"Overall Bias: {result['overall_bias']}")
    print(f"Chart Confidence: {result['chart_confidence']}/100")
    print(f"Chart Saved: {result['chart_path']}")
    print(f"Patterns Found: {result['pattern_count']}")
    print(f"  Backtest coverage: {result['backtest_summary'].get('patterns_with_backtest', 0)} "
          f"with data, {result['backtest_summary'].get('reliable_patterns', 0)} reliable")

    print("\n" + "-" * 70)
    for p in result["patterns"]:
        emoji = "🟢" if p["pattern_type"] == "bullish" else (
            "🔴" if p["pattern_type"] == "bearish" else "🟡"
        )
        wr = p.get("win_rate", 0)
        occ = p.get("occurrences", 0)
        print(f"\n{emoji} {p['pattern_name']} (Confidence: {p['confidence']}%)")
        print(f"   Action: {p['suggested_action']}")
        if occ > 0:
            print(f"   Win Rate: {wr * 100:.0f}% over {occ} occurrences "
                  f"(avg return: {p.get('avg_return_pct', 0):.2f}%)")
            print(f"   {p.get('win_rate_summary', '')[:120]}")
        else:
            print(f"   No backtest data available")

    print("\n" + "-" * 70)
    print("Key Indicators:")
    for k, v in result["indicators"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
