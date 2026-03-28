"""reasoning_agent.py
=================================
Plain-English reasoning agent for investor-friendly explanations.

Primary LLM:
- Groq model: llama-3.3-70b-versatile
Fallback LLM:
- OpenRouter model: meta-llama/llama-3.3-70b-instruct:free

Output schema (exact):
{
    "ticker": str,
    "explanation": str,
    "action": "BUY" | "WATCH" | "AVOID",
    "confidence_plain": "Very High" | "High" | "Medium" | "Low",
    "key_points": list[str],
    "sources": list[str],
    "rag_context_used": bool,
    "action_plan": {
        "entry_price_range": {"min": float, "max": float},
        "stop_loss": {"price": float, "basis": str},
        "target_price": {"price": float, "basis": str},
        "time_horizon_days": 5 | 10 | 20,
        "risk_rating": "Low" | "Medium" | "High",
        "rationale": list[str]
    }
}
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

logger = logging.getLogger(__name__)


def _is_quota_or_rate_error(exc: Exception) -> bool:
    """Return True when an exception likely represents quota/rate/credit exhaustion."""
    message = str(exc).lower()
    markers = ["quota", "rate", "429", "limit", "credits", "insufficient_quota"]
    return any(marker in message for marker in markers)


def _confidence_plain_from_score(convergence_score: float) -> str:
    """Map convergence score to user-facing confidence wording."""
    score = float(convergence_score)
    if score >= 0.80:
        return "Very High"
    if score >= 0.65:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def _action_from_label(convergence_label: str) -> str:
    """Map convergence label to required action keyword."""
    label = str(convergence_label).upper()
    if "STRONG BUY" in label:
        return "BUY"
    if "WATCH" in label:
        return "WATCH"
    return "AVOID"


def _build_prompt(
    ticker: str,
    convergence_data: dict[str, Any],
    rag_context: list[dict[str, Any]] | None,
    portfolio: dict[str, int] | None = None,
) -> str:
    """Build instruction prompt for reasoning generation with strict JSON output."""
    rag_text = "No additional RAG context provided."
    if rag_context:
        rag_lines = []
        for item in rag_context[:8]:
            source = item.get("source", "Unknown Source")
            text = item.get("text", "")
            rag_lines.append(f"- Source: {source} | Text: {text}")
        rag_text = "\n".join(rag_lines)

    portfolio_text = ""
    if portfolio:
        qty = portfolio.get(ticker, 0)
        if qty and qty > 0:
            portfolio_text = (
                f"\nPortfolio Context:\n"
                f"User holds {qty} shares of {ticker}.\n"
                f"Personalise the explanation to mention the impact on their specific holding.\n"
                f"Include a sentence about what this signal means for their {qty} shares.\n"
            )

    return f"""
You are a financial AI assistant for Indian retail investors.
Write at Class 10 reading level and avoid jargon.
Use only the provided data; do not invent facts.
Return ONLY valid JSON with keys exactly:
explanation, action, confidence_plain, key_points, sources

Ticker: {ticker}
Convergence Data JSON:
{json.dumps(convergence_data, ensure_ascii=True)}

RAG Context:
{rag_text}
{portfolio_text}
Rules:
- Keep explanation to 4-6 short sentences.
- action must be BUY, WATCH, or AVOID.
- confidence_plain must be Very High, High, Medium, or Low.
- key_points should contain 3-5 concise bullet-style strings.
- sources should include explicit source strings from convergence_data and RAG context.
""".strip()


def _safe_json_parse(text: str) -> dict[str, Any]:
    """Parse a JSON string safely, including extraction from fenced blocks."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _to_float(value: Any) -> float | None:
    """Safely coerce numeric-like values to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_risk_reward(entry: float, stop_loss: float, target: float) -> float:
    """Compute risk/reward ratio: (target - entry) / (entry - stop_loss)."""
    risk = entry - stop_loss
    if risk <= 0:
        return 0.0
    reward = target - entry
    return round(reward / risk, 1)


def _extract_action_plan_inputs(convergence_data: dict[str, Any]) -> dict[str, Any]:
    """Extract current price, support, resistance, and pattern hints from chart data."""
    chart_data = convergence_data.get("chart_data", {}) if isinstance(convergence_data, dict) else {}
    indicators = chart_data.get("indicators", {}) if isinstance(chart_data, dict) else {}
    patterns = chart_data.get("patterns", []) if isinstance(chart_data, dict) else []
    vlm = chart_data.get("vlm_analysis", {}) if isinstance(chart_data, dict) else {}

    # Priority 1: current_price directly from chart_agent (OHLCV last close)
    current_price = _to_float(chart_data.get("current_price"))
    # Priority 2: indicators close
    if not current_price:
        current_price = _to_float(indicators.get("close"))
    # Priority 3: pattern indicator values
    top_pattern = patterns[0] if isinstance(patterns, list) and patterns else {}
    if not current_price and isinstance(top_pattern, dict):
        iv = top_pattern.get("indicator_values", {})
        if isinstance(iv, dict):
            current_price = _to_float(iv.get("close"))
    if current_price is None or current_price <= 0:
        current_price = 0.0

    # Support/resistance: use 20-day OHLCV range from chart_agent first
    support_level = _to_float(chart_data.get("recent_low_20"))
    resistance_level = _to_float(chart_data.get("recent_high_20"))

    # Fallback to BB bands and VLM
    if not support_level:
        support_level = _to_float(indicators.get("BB_lower"))
    if not resistance_level:
        resistance_level = _to_float(indicators.get("BB_upper"))
    if isinstance(vlm, dict):
        support_level = support_level or _to_float(vlm.get("support_level"))
        resistance_level = resistance_level or _to_float(vlm.get("resistance_level"))

    # Final fallback: percentage of current price
    if not support_level and current_price > 0:
        support_level = round(current_price * 0.95, 2)
    if not resistance_level and current_price > 0:
        resistance_level = round(current_price * 1.08, 2)
    if support_level is None:
        support_level = 0.0
    if resistance_level is None:
        resistance_level = 0.0

    # Sanity guards
    if support_level > current_price and current_price > 0:
        support_level = round(current_price * 0.95, 2)
    if resistance_level < current_price and current_price > 0:
        resistance_level = round(current_price * 1.08, 2)

    pattern_name = str(top_pattern.get("pattern_name", "")) if isinstance(top_pattern, dict) else ""
    pattern_category = str(top_pattern.get("category", "")) if isinstance(top_pattern, dict) else ""
    pattern_type = str(top_pattern.get("pattern_type", "")) if isinstance(top_pattern, dict) else ""

    return {
        "current_price": round(float(current_price), 2),
        "support_level": round(float(support_level), 2),
        "resistance_level": round(float(resistance_level), 2),
        "pattern_name": pattern_name,
        "pattern_category": pattern_category,
        "pattern_type": pattern_type,
    }


def _build_action_plan_prompt(
    ticker: str,
    action: str,
    convergence_data: dict[str, Any],
    plan_inputs: dict[str, Any],
) -> str:
    """Build strict JSON prompt for action plan generation."""
    return f"""
You are a precise trading plan assistant.
Create an execution-ready action plan for ticker {ticker} using only provided inputs.
Return ONLY valid JSON with keys exactly:
entry_price_range, stop_loss, target_price, time_horizon_days, risk_rating, risk_reward_ratio, rationale

Inputs:
- ticker: {ticker}
- action: {action}
- convergence_label: {convergence_data.get('convergence_label', 'WEAK')}
- convergence_score: {convergence_data.get('convergence_score', 0.0)}
- current_price: {plan_inputs.get('current_price')}
- support_level: {plan_inputs.get('support_level')}
- resistance_level: {plan_inputs.get('resistance_level')}
- pattern_name: {plan_inputs.get('pattern_name')}
- pattern_category: {plan_inputs.get('pattern_category')}
- pattern_type: {plan_inputs.get('pattern_type')}

Rules:
- entry_price_range min/max must be current_price +/- 2% exactly (rounded to 2 decimals).
- stop_loss.price should align with support_level (or a slightly lower buffer).
- target_price.price should align with resistance_level.
- time_horizon_days must be one of 5, 10, 20.
- Choose horizon using pattern category:
  candlestick -> 5, momentum -> 10, trend/volatility -> 20.
- risk_rating must be Low, Medium, or High.
- risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss). Round to 1 decimal.
- rationale should be 3 concise strings.
- Use numeric values only for price fields.
""".strip()


def _deterministic_action_plan(
    ticker: str,
    action: str,
    convergence_data: dict[str, Any],
    plan_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build non-LLM action plan fallback using deterministic math and chart hints."""
    current_price = float(plan_inputs.get("current_price", 0.0) or 0.0)
    support = float(plan_inputs.get("support_level", 0.0) or 0.0)
    resistance = float(plan_inputs.get("resistance_level", 0.0) or 0.0)

    # Ensure support/resistance are never zero when we have a price
    if current_price > 0 and support <= 0:
        support = round(current_price * 0.95, 2)
    if current_price > 0 and resistance <= 0:
        resistance = round(current_price * 1.08, 2)

    entry_min = round(current_price * 0.99, 2) if current_price > 0 else 0.0
    entry_max = round(current_price * 1.01, 2) if current_price > 0 else 0.0

    stop_loss = round(support, 2) if support > 0 else round(current_price * 0.95, 2)
    target_price = round(resistance, 2) if resistance > 0 else round(current_price * 1.08, 2)

    pattern_category = str(plan_inputs.get("pattern_category", "")).lower()
    if pattern_category == "candlestick":
        horizon = 5
    elif pattern_category == "momentum":
        horizon = 10
    else:
        horizon = 20

    score = float(convergence_data.get("convergence_score", 0.0))
    if score >= 0.75:
        risk = "Low"
    elif score >= 0.50:
        risk = "Medium"
    else:
        risk = "High"

    basis_text = f"20-day low support near {support:.2f}"
    target_basis = f"20-day high resistance near {resistance:.2f}"

    return {
        "entry_price_range": {"min": entry_min, "max": entry_max},
        "stop_loss": {"price": stop_loss, "basis": basis_text},
        "target_price": {"price": target_price, "basis": target_basis},
        "time_horizon_days": horizon,
        "risk_rating": risk,
        "risk_reward_ratio": _compute_risk_reward(current_price, stop_loss, target_price),
        "rationale": [
            f"Action stance is {action} with convergence score {score:.2f}.",
            f"Entry band ±1% around current price {current_price:.2f}.",
            f"Stop loss at 20-day low {stop_loss:.2f}, target at 20-day high {target_price:.2f}.",
        ],
    }


def _normalize_action_plan(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize model output into strict action plan schema."""
    plan = dict(fallback)

    entry = parsed.get("entry_price_range", {}) if isinstance(parsed, dict) else {}
    if isinstance(entry, dict):
        entry_min = _to_float(entry.get("min"))
        entry_max = _to_float(entry.get("max"))
        if entry_min is not None and entry_max is not None and entry_min <= entry_max:
            plan["entry_price_range"] = {"min": round(entry_min, 2), "max": round(entry_max, 2)}

    stop = parsed.get("stop_loss", {}) if isinstance(parsed, dict) else {}
    if isinstance(stop, dict):
        stop_price = _to_float(stop.get("price"))
        if stop_price is not None and stop_price >= 0:
            plan["stop_loss"] = {
                "price": round(stop_price, 2),
                "basis": str(stop.get("basis") or plan["stop_loss"]["basis"]),
            }

    target = parsed.get("target_price", {}) if isinstance(parsed, dict) else {}
    if isinstance(target, dict):
        target_val = _to_float(target.get("price"))
        if target_val is not None and target_val >= 0:
            plan["target_price"] = {
                "price": round(target_val, 2),
                "basis": str(target.get("basis") or plan["target_price"]["basis"]),
            }

    horizon = parsed.get("time_horizon_days") if isinstance(parsed, dict) else None
    if horizon in {5, 10, 20}:
        plan["time_horizon_days"] = int(horizon)

    risk = str(parsed.get("risk_rating", "")).strip().title() if isinstance(parsed, dict) else ""
    if risk in {"Low", "Medium", "High"}:
        plan["risk_rating"] = risk

    rationale = parsed.get("rationale", []) if isinstance(parsed, dict) else []
    if isinstance(rationale, list):
        cleaned = [str(item).strip() for item in rationale if str(item).strip()]
        if cleaned:
            plan["rationale"] = cleaned[:4]

    # Compute risk_reward_ratio from final validated prices
    entry_mid = (plan["entry_price_range"]["min"] + plan["entry_price_range"]["max"]) / 2
    plan["risk_reward_ratio"] = _compute_risk_reward(
        entry_mid, plan["stop_loss"]["price"], plan["target_price"]["price"]
    )

    return plan


def _call_groq(prompt: str) -> dict[str, Any]:
    """Call Groq chat completion and return parsed JSON response."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return _safe_json_parse(content)
    except Exception:
        raise


def _call_openrouter(prompt: str) -> dict[str, Any]:
    """Call OpenRouter fallback model and return parsed JSON response."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return _safe_json_parse(content)
    except Exception:
        raise


def _generate_action_plan(
    ticker: str,
    action: str,
    convergence_data: dict[str, Any],
) -> dict[str, Any]:
    """Generate action plan via a second Groq call, with deterministic fallback."""
    plan_inputs = _extract_action_plan_inputs(convergence_data)
    fallback = _deterministic_action_plan(ticker, action, convergence_data, plan_inputs)

    prompt = _build_action_plan_prompt(ticker, action, convergence_data, plan_inputs)

    try:
        parsed = _call_groq(prompt)
        logger.info("Action plan generated via Groq for %s", ticker)
        return _normalize_action_plan(parsed, fallback)
    except Exception as exc:
        logger.warning("Action plan Groq call failed for %s: %s", ticker, exc)
        return fallback


def _deterministic_fallback(
    ticker: str,
    convergence_data: dict[str, Any],
    rag_context: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Generate a safe, non-LLM fallback explanation when API calls fail."""
    convergence_score = float(convergence_data.get("convergence_score", 0.0))
    convergence_label = str(convergence_data.get("convergence_label", "WEAK"))
    action = _action_from_label(convergence_label)
    confidence_plain = _confidence_plain_from_score(convergence_score)

    breakdown = convergence_data.get("signal_breakdown", {})
    chart = float(breakdown.get("chart_contribution", 0.0))
    insider = float(breakdown.get("insider_contribution", 0.0))
    earnings = float(breakdown.get("earnings_contribution", 0.0))
    sentiment = float(breakdown.get("sentiment_contribution", 0.0))

    explanation = (
        f"{ticker} currently has a convergence score of {convergence_score:.2f}, "
        f"which falls in the '{convergence_label}' zone. "
        f"The strongest push is from chart and insider signals, while earnings and sentiment add supporting strength. "
        f"This suggests a {action.lower()} stance right now, but you should keep tracking new filings and quarterly updates."
    )

    sources = [
        "Chart agent output",
        "Radar agent output",
        "Sentiment agent output",
    ]
    if rag_context:
        for item in rag_context[:3]:
            src = item.get("source")
            if src:
                sources.append(str(src))

    return {
        "ticker": ticker,
        "explanation": explanation,
        "action": action,
        "confidence_plain": confidence_plain,
        "key_points": [
            f"Chart contribution: {chart:.3f}",
            f"Insider contribution: {insider:.3f}",
            f"Earnings contribution: {earnings:.3f}",
            f"Sentiment contribution: {sentiment:.3f}",
        ],
        "sources": sources,
        "rag_context_used": bool(rag_context),
        "action_plan": _generate_action_plan(ticker, action, convergence_data),
    }


def generate_reasoning(
    ticker: str,
    convergence_data: dict[str, Any],
    rag_context: list[dict[str, Any]] | None = None,
    portfolio: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Generate class-10-level explanation from convergence data, with API fallback chain.

    Args:
        ticker: Stock ticker without .NS suffix.
        convergence_data: Output payload from convergence_agent.
        rag_context: Optional retrieved context chunks, each containing text/source.
        portfolio: Optional user portfolio dict mapping ticker -> quantity.

    Returns:
        Dict matching reasoning_agent output schema exactly.
    """
    normalized_ticker = ticker.strip().upper().replace(".NS", "")
    prompt = _build_prompt(normalized_ticker, convergence_data, rag_context, portfolio=portfolio)

    parsed: dict[str, Any]

    try:
        parsed = _call_groq(prompt)
        logger.info("Reasoning generated via Groq for %s", normalized_ticker)
    except Exception as groq_exc:
        logger.warning("Groq call failed for %s: %s", normalized_ticker, groq_exc)
        try_openrouter = _is_quota_or_rate_error(groq_exc) or True
        if try_openrouter:
            try:
                parsed = _call_openrouter(prompt)
                logger.info("Reasoning generated via OpenRouter fallback for %s", normalized_ticker)
            except Exception as or_exc:
                logger.error("OpenRouter fallback failed for %s: %s", normalized_ticker, or_exc)
                return _deterministic_fallback(normalized_ticker, convergence_data, rag_context)
        else:
            return _deterministic_fallback(normalized_ticker, convergence_data, rag_context)

    convergence_score = float(convergence_data.get("convergence_score", 0.0))
    convergence_label = str(convergence_data.get("convergence_label", "WEAK"))

    explanation = str(parsed.get("explanation", "")).strip()
    action = str(parsed.get("action", _action_from_label(convergence_label))).strip().upper()
    confidence_plain = str(
        parsed.get("confidence_plain", _confidence_plain_from_score(convergence_score))
    ).strip()
    key_points = parsed.get("key_points", [])
    sources = parsed.get("sources", [])

    if action not in {"BUY", "WATCH", "AVOID"}:
        action = _action_from_label(convergence_label)

    if confidence_plain not in {"Very High", "High", "Medium", "Low"}:
        confidence_plain = _confidence_plain_from_score(convergence_score)

    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(item) for item in key_points if str(item).strip()][:5]

    if not isinstance(sources, list):
        sources = []
    sources = [str(item) for item in sources if str(item).strip()][:6]

    if not explanation:
        return _deterministic_fallback(normalized_ticker, convergence_data, rag_context)

    if not key_points:
        breakdown = convergence_data.get("signal_breakdown", {})
        key_points = [
            f"Chart contribution: {float(breakdown.get('chart_contribution', 0.0)):.3f}",
            f"Insider contribution: {float(breakdown.get('insider_contribution', 0.0)):.3f}",
            f"Earnings contribution: {float(breakdown.get('earnings_contribution', 0.0)):.3f}",
            f"Sentiment contribution: {float(breakdown.get('sentiment_contribution', 0.0)):.3f}",
        ]

    if not sources:
        sources = [
            "Chart agent output",
            "Radar agent output",
            "Sentiment agent output",
        ]

    action_plan = _generate_action_plan(normalized_ticker, action, convergence_data)

    return {
        "ticker": normalized_ticker,
        "explanation": explanation,
        "action": action,
        "confidence_plain": confidence_plain,
        "key_points": key_points,
        "sources": sources,
        "rag_context_used": bool(rag_context),
        "action_plan": action_plan,
    }


def smoke_test() -> dict[str, Any]:
    """Run a local smoke test on synthetic convergence payload."""
    convergence_data = {
        "ticker": "RELIANCE",
        "convergence_score": 0.84,
        "convergence_label": "STRONG BUY SIGNAL",
        "signal_breakdown": {
            "chart_contribution": 0.28,
            "insider_contribution": 0.255,
            "earnings_contribution": 0.16,
            "sentiment_contribution": 0.108,
        },
    }
    rag_context = [
        {"source": "NSE bulk deal data, March 15 2026", "text": "Promoter entity bought shares in open market."},
        {"source": "SEBI PIT disclosure, March 14 2026", "text": "Net insider buying observed this week."},
    ]
    return generate_reasoning("RELIANCE", convergence_data, rag_context)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(smoke_test())
