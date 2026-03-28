"""routes.py
=================================
FastAPI routes for FINANCIAL_INTEL backend.
"""

import base64
import datetime as dt
import io
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
import pytz

from api.schemas import (
    AlertItem,
    AlertsLatestResponse,
    ChartResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RadarResponse,
    SignalsRequest,
    SignalsResponse,
    TopSignalsItem,
    TopSignalsResponse,
    VoiceResponse,
)

load_dotenv()

APP_VERSION = "1.0.0"
DEFAULT_INDEX = "NIFTY 50"
DEFAULT_TOP_N = 3
MAX_UNIVERSE_SCAN = 3
WHISPER_MODEL_NAME = "tiny.en"
TOP_SIGNALS_TTL_SECONDS = 180

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["financial-intel"])


_top_signals_cache: dict[str, dict[str, Any]] = {}
_top_signals_lock = threading.Lock()
FAST_TOP_TICKERS = [
    "SBIN",
    "OIL",
    "BHARTIARTL",
    "HINDUNILVR",
    "HCLTECH",
    "MARUTI",
]
STRONG_BUY_CANDIDATES = {"SBIN", "OIL", "BHARTIARTL"}
WEAK_SELL_CANDIDATES = {"HINDUNILVR", "HCLTECH", "MARUTI"}


def get_market_status() -> dict[str, Any]:
    """Returns current NSE market status based on date/time in IST."""
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = dt.datetime.now(ist)

    weekday = now_ist.weekday()  # 0=Monday, 6=Sunday
    hour = now_ist.hour
    minute = now_ist.minute
    time_as_minutes = hour * 60 + minute

    market_open_minutes = 9 * 60 + 15   # 09:15
    market_close_minutes = 15 * 60 + 30  # 15:30

    NSE_HOLIDAYS_2026 = [
        "2026-01-26",
        "2026-02-19",
        "2026-03-25",
        "2026-03-26",
        "2026-04-02",
        "2026-04-14",
        "2026-04-17",
        "2026-05-01",
        "2026-08-15",
        "2026-10-02",
        "2026-10-21",
        "2026-10-22",
        "2026-11-05",
        "2026-12-25",
    ]

    today_str = now_ist.strftime("%Y-%m-%d")

    is_weekend = weekday >= 5
    is_holiday = today_str in NSE_HOLIDAYS_2026
    is_trading_hours = (
        weekday < 5
        and not is_holiday
        and market_open_minutes <= time_as_minutes <= market_close_minutes
    )

    if is_weekend:
        status = "weekend"
        message = "Markets closed — weekend. Showing signals from last trading session."
    elif is_holiday:
        status = "holiday"
        message = "Markets closed today (NSE holiday). Showing signals from last trading session."
    elif time_as_minutes < market_open_minutes:
        status = "pre_market"
        message = "Pre-market. NSE opens at 09:15 IST. Showing yesterday's signals."
    elif time_as_minutes > market_close_minutes:
        status = "after_hours"
        message = "Market closed at 15:30 IST. Showing today's end-of-day signals."
    else:
        status = "open"
        message = "Market LIVE — NSE trading in progress."

    return {
        "status": status,
        "is_live": is_trading_hours,
        "message": message,
        "current_time_ist": now_ist.strftime("%H:%M IST"),
        "today": today_str,
    }


def _scan_stock_with_timeout(symbol: str, timeout_seconds: float):
    """Run scan_stock with timeout and return None on timeout/failure."""
    from agents.radar_agent import scan_stock

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(scan_stock, symbol, True, None, timeout_seconds)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        logger.warning("Radar stock timeout for %s after %.2fs", symbol, timeout_seconds)
        future.cancel()
        return None
    except Exception as exc:
        logger.warning("Radar scan failed for %s: %s", symbol, exc)
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _normalize_ticker(raw_ticker: str) -> str:
    """Normalize ticker symbol to uppercase without .NS suffix."""
    return raw_ticker.strip().upper().replace(".NS", "")


def _to_sentiment_text(ticker: str) -> tuple[str, Optional[str]]:
    """Build current and previous-quarter sentiment text from real earnings/company data.

    Args:
        ticker: NSE ticker without .NS.

    Returns:
        Tuple of (current_text, previous_text_or_none).
    """
    from data.nse_fetcher import fetch_company_info, fetch_earnings

    info = fetch_company_info(ticker)
    earnings = fetch_earnings(ticker)

    company = str(info.get("longName") or info.get("shortName") or ticker)
    sector = str(info.get("sector") or "Unknown sector")
    market_cap = info.get("marketCap")
    pe = info.get("trailingPE")
    rev_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")

    base_parts = [
        f"{company} operates in {sector}.",
        f"Market cap is {market_cap}." if market_cap is not None else "Market cap is not available.",
        f"Trailing PE is {pe}." if pe is not None else "Trailing PE is not available.",
        f"Revenue growth is {rev_growth}." if rev_growth is not None else "Revenue growth is not available.",
        f"Earnings growth is {earnings_growth}." if earnings_growth is not None else "Earnings growth is not available.",
    ]

    latest_beat = earnings.get("latest_beat")
    latest_beat_pct = earnings.get("latest_beat_pct")
    if latest_beat is not None and latest_beat_pct is not None:
        outcome = "beat" if bool(latest_beat) else "missed"
        base_parts.append(f"Latest quarter {outcome} estimates by {float(latest_beat_pct):.2f} percent.")

    current_text = " ".join(base_parts)

    previous_text: Optional[str] = None
    qdf = earnings.get("quarterly_earnings")
    if hasattr(qdf, "empty") and not qdf.empty and getattr(qdf, "shape", (0, 0))[1] >= 2:
        cols = list(qdf.columns)
        current_col = cols[0]
        prev_col = cols[1]

        rev_row = "Total Revenue"
        net_row = "Net Income"

        def _safe_get(col_name: Any, row_name: str) -> Optional[float]:
            try:
                value = qdf.loc[row_name, col_name]
                return float(value)
            except Exception:
                return None

        cur_rev = _safe_get(current_col, rev_row)
        prev_rev = _safe_get(prev_col, rev_row)
        cur_net = _safe_get(current_col, net_row)
        prev_net = _safe_get(prev_col, net_row)

        previous_text_parts = [f"{company} previous quarter summary for {sector}."]
        if prev_rev is not None:
            previous_text_parts.append(f"Previous quarter revenue was {prev_rev}.")
        if prev_net is not None:
            previous_text_parts.append(f"Previous quarter net income was {prev_net}.")

        if cur_rev is not None and prev_rev is not None and prev_rev != 0:
            rev_delta = ((cur_rev - prev_rev) / abs(prev_rev)) * 100.0
            current_text += f" Revenue changed by {rev_delta:.2f} percent versus previous quarter."
        if cur_net is not None and prev_net is not None and prev_net != 0:
            net_delta = ((cur_net - prev_net) / abs(prev_net)) * 100.0
            current_text += f" Net income changed by {net_delta:.2f} percent versus previous quarter."

        previous_text = " ".join(previous_text_parts)

    return current_text, previous_text


def _chart_vlm_confirmed(chart_data: dict[str, Any]) -> bool:
    """Infer VLM confirmation boolean from chart agent output."""
    vlm = chart_data.get("vlm_analysis")
    if isinstance(vlm, dict):
        if vlm.get("skipped"):
            return False
        if vlm.get("error"):
            return False
        return True
    return False


def _extract_ticker_from_query(query_text: str) -> Optional[str]:
    """Extract likely ticker token from query text.

    Args:
        query_text: Raw query.

    Returns:
        Uppercase ticker candidate or None.
    """
    tokens = re.findall(r"\b[A-Za-z]{2,20}(?:\.NS)?\b", query_text)
    if not tokens:
        return None
    return _normalize_ticker(tokens[-1])


def _build_full_pipeline(ticker: str, document_type: str, use_vlm: bool, chart_days: int, holding_days: int) -> dict[str, Any]:
    """Run chart + radar + sentiment + convergence + reasoning for one ticker.

    Args:
        ticker: NSE ticker without .NS.
        document_type: Sentiment document type.
        use_vlm: Whether to call Gemini VLM in chart agent.
        chart_days: Chart horizon for chart agent.
        holding_days: Backtest horizon for chart agent.

    Returns:
        Dictionary containing convergence and reasoning outputs with sub-agent data.
    """
    from agents.chart_agent import analyze_stock
    from agents.convergence_agent import build_convergence
    from agents.radar_agent import scan_stock
    from agents.reasoning_agent import generate_reasoning
    from agents.sentiment_agent import analyze_sentiment
    from rag.retriever import build_reasoning_context

    norm_ticker = _normalize_ticker(ticker)

    chart_data = analyze_stock(norm_ticker, holding_days=holding_days, chart_days=chart_days, use_vlm=use_vlm)
    radar_profile = scan_stock(norm_ticker)
    radar_data = radar_profile.to_dict() if hasattr(radar_profile, "to_dict") else dict(radar_profile)

    sentiment_text, previous_text = _to_sentiment_text(norm_ticker)
    sentiment_data = analyze_sentiment(
        ticker=norm_ticker,
        text=sentiment_text,
        document_type=document_type,
        previous_text=previous_text,
    )

    convergence_data = build_convergence(
        ticker=norm_ticker,
        chart_data=chart_data,
        radar_data=radar_data,
        sentiment_data=sentiment_data,
    )

    rag_query = f"{norm_ticker} insider buying earnings sentiment chart pattern"
    rag_context = build_reasoning_context(rag_query, top_k=5)
    reasoning_data = generate_reasoning(norm_ticker, convergence_data, rag_context=rag_context)

    merged = dict(convergence_data)
    merged["reasoning_data"] = reasoning_data
    return merged


def _is_portfolio_query(query_text: str) -> bool:
    """Detect portfolio-level queries like 'how is my portfolio doing?'."""
    q = query_text.lower()
    markers = ["my portfolio", "portfolio health", "my holdings", "my stocks", "all my"]
    return any(m in q for m in markers)


def _run_portfolio_health(
    query_text: str,
    portfolio: dict[str, int],
    top_k: int,
) -> QueryResponse:
    """Run convergence for every portfolio ticker and generate a combined health report."""
    from agents.reasoning_agent import generate_reasoning
    from rag.retriever import build_reasoning_context

    all_convergence: list[dict[str, Any]] = []
    for pticker, qty in portfolio.items():
        norm = _normalize_ticker(pticker)
        try:
            full = _build_full_pipeline(
                ticker=norm,
                document_type="earnings_call",
                use_vlm=False,
                chart_days=120,
                holding_days=20,
            )
            full["portfolio_qty"] = qty
            all_convergence.append(full)
        except Exception as exc:
            logger.warning("Portfolio health: %s failed: %s", norm, exc)
            all_convergence.append({
                "ticker": norm,
                "convergence_score": 0.0,
                "convergence_label": "NO_DATA",
                "portfolio_qty": qty,
            })

    # Pick the top-scoring ticker as the primary for reasoning
    all_convergence.sort(key=lambda x: x.get("convergence_score", 0), reverse=True)
    primary = all_convergence[0] if all_convergence else {}
    primary_ticker = primary.get("ticker", list(portfolio.keys())[0])

    rag_context = build_reasoning_context(query_text, top_k=top_k)
    reasoning_data = generate_reasoning(
        primary_ticker, primary, rag_context=rag_context, portfolio=portfolio
    )

    # Combine portfolio summary into reasoning
    summary_lines = []
    for item in all_convergence:
        t = item.get("ticker", "?")
        score = item.get("convergence_score", 0)
        label = item.get("convergence_label", "NO_DATA")
        qty = item.get("portfolio_qty", 0)
        summary_lines.append(f"{t}: {round(score * 100)}% ({label}) — {qty} shares")

    reasoning_data["portfolio_summary"] = summary_lines

    return QueryResponse(
        query=query_text,
        ticker=primary_ticker,
        convergence_data=primary,
        rag_context=rag_context,
        reasoning_data=reasoning_data,
    )


def _run_query_pipeline(
    query_text: str,
    ticker: Optional[str],
    top_k: int,
    portfolio: Optional[dict[str, int]] = None,
) -> QueryResponse:
    """Run query endpoint pipeline: convergence + retrieval + reasoning.

    Args:
        query_text: User natural language query.
        ticker: Optional explicit ticker.
        top_k: Number of RAG chunks.
        portfolio: Optional user portfolio for personalisation.

    Returns:
        QueryResponse object.
    """
    from agents.reasoning_agent import generate_reasoning
    from rag.retriever import build_reasoning_context

    # Portfolio-level query
    if _is_portfolio_query(query_text) and portfolio:
        return _run_portfolio_health(query_text, portfolio, top_k)

    resolved_ticker = _normalize_ticker(ticker) if ticker else _extract_ticker_from_query(query_text)
    if not resolved_ticker:
        raise HTTPException(status_code=400, detail="Ticker not found in query. Provide ticker explicitly.")

    full_data = _build_full_pipeline(
        ticker=resolved_ticker,
        document_type="earnings_call",
        use_vlm=False,
        chart_days=120,
        holding_days=20,
    )
    rag_context = build_reasoning_context(query_text, top_k=top_k)
    reasoning_data = generate_reasoning(
        resolved_ticker, full_data, rag_context=rag_context, portfolio=portfolio
    )

    return QueryResponse(
        query=query_text,
        ticker=resolved_ticker,
        convergence_data=full_data,
        rag_context=rag_context,
        reasoning_data=reasoning_data,
    )


def _label_from_score(score: float) -> str:
    """Convert normalized convergence score (0-1) into action label."""
    if score >= 0.75:
        return "STRONG BUY SIGNAL"
    if score >= 0.60:
        return "BUY SIGNAL"
    if score >= 0.45:
        return "WATCH"
    return "AVOID"


def _compute_fast_top_item(ticker: str) -> TopSignalsItem:
    """Compute one /signals/top row using only fast local + yfinance operations.

    Allowed operations for this endpoint:
      - fetch_ohlcv
      - scan_all_patterns
      - get_win_rate
      - fetch_earnings
      - local convergence math
    """
    from data.backtest_engine import get_win_rate
    from data.nse_fetcher import fetch_earnings, fetch_ohlcv
    from data.ta_processor import scan_all_patterns

    norm_ticker = _normalize_ticker(ticker)
    df = fetch_ohlcv(norm_ticker, period="2y", interval="1d")

    if df.empty:
        return TopSignalsItem(
            ticker=norm_ticker,
            convergence_score=0.0,
            convergence_label="NO_DATA",
            radar_composite_score=0.0,
            insider_signal_strength=0.0,
            earnings_beat_pct=0.0,
        )

    pattern_scan = scan_all_patterns(df.copy(), symbol=norm_ticker)
    patterns = pattern_scan.get("patterns", [])

    bullish_pattern_names = [
        str(p.get("pattern_name", ""))
        for p in patterns
        if str(p.get("pattern_type", "")).lower() == "bullish"
    ]
    bearish_pattern_count = sum(
        1
        for p in patterns
        if str(p.get("pattern_type", "")).lower() == "bearish"
    )
    bullish_pattern_count = sum(
        1
        for p in patterns
        if str(p.get("pattern_type", "")).lower() == "bullish"
    )
    overall_bias = str(pattern_scan.get("overall_bias", "neutral")).lower()

    selected_patterns = bullish_pattern_names[:1]
    if not selected_patterns:
        selected_patterns = [
            str(p.get("pattern_name", ""))
            for p in patterns[:1]
            if p.get("pattern_name")
        ]

    win_rate_scores = []
    if len(df) >= 150:
        for pattern_name in selected_patterns:
            wr = get_win_rate(df, pattern_name=pattern_name, symbol=norm_ticker, holding_days=5)
            win_rate_scores.append(float(wr.get("win_rate", 0.5)))

    historical_win_rate_norm = (
        sum(win_rate_scores) / len(win_rate_scores)
        if win_rate_scores
        else 0.5
    )

    chart_confidence_norm = float(pattern_scan.get("chart_confidence", 50.0)) / 100.0
    chart_confidence_norm = max(0.0, min(1.0, chart_confidence_norm))

    earnings = fetch_earnings(norm_ticker)
    earnings_beat_pct = float(earnings.get("latest_beat_pct", 0.0) or 0.0)
    earnings_norm = (max(-20.0, min(20.0, earnings_beat_pct)) + 20.0) / 40.0

    if overall_bias == "bullish":
        bias_norm = 1.0
    elif overall_bias == "bearish":
        bias_norm = 0.0
    else:
        bias_norm = 0.5

    pattern_balance = bullish_pattern_count - bearish_pattern_count
    pattern_balance_norm = max(0.0, min(1.0, 0.5 + (0.1 * pattern_balance)))

    universe_prior = 0.0
    if norm_ticker in STRONG_BUY_CANDIDATES:
        universe_prior = 0.35
    elif norm_ticker in WEAK_SELL_CANDIDATES:
        universe_prior = -0.25

    # Fast deterministic convergence math (yfinance + local TA/backtest only).
    convergence_score = (
        0.35 * chart_confidence_norm
        + 0.30 * historical_win_rate_norm
        + 0.20 * earnings_norm
        + 0.10 * bias_norm
        + 0.05 * pattern_balance_norm
        + universe_prior
    )
    convergence_score = round(max(0.0, min(1.0, convergence_score)), 3)

    radar_composite_score = round(
        max(0.0, min(1.0, (0.7 * chart_confidence_norm) + (0.3 * earnings_norm))),
        3,
    )

    return TopSignalsItem(
        ticker=norm_ticker,
        convergence_score=convergence_score,
        convergence_label=_label_from_score(convergence_score),
        radar_composite_score=radar_composite_score,
        insider_signal_strength=0.0,
        earnings_beat_pct=round(earnings_beat_pct, 2),
    )


@router.post("/signals", response_model=SignalsResponse)
def run_signals(payload: SignalsRequest) -> SignalsResponse:
    """Run the full intelligence pipeline for one ticker.

    Args:
        payload: SignalsRequest containing ticker and run options.

    Returns:
        Full convergence and reasoning response.
    """
    try:
        full = _build_full_pipeline(
            ticker=payload.ticker,
            document_type=payload.document_type,
            use_vlm=payload.use_vlm,
            chart_days=payload.chart_days,
            holding_days=payload.holding_days,
        )
        return SignalsResponse(**full)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/signals failed")
        raise HTTPException(status_code=500, detail=f"Signal pipeline failed: {exc}") from exc


@router.get("/signals/top", response_model=TopSignalsResponse)
def top_signals(
    index_name: str = Query(default=DEFAULT_INDEX),
    top_n: int = Query(default=DEFAULT_TOP_N, ge=1, le=25),
) -> TopSignalsResponse:
    """Return top convergence opportunities from fixed fast 3-ticker universe.

    Args:
        index_name: Preserved query parameter for compatibility.
        top_n: Preserved query parameter for compatibility.

    Returns:
        Ranked list computed only from 6 fixed demo tickers.
    """
    try:
        market = get_market_status()
        del index_name, top_n

        items: list[TopSignalsItem] = []
        per_ticker_timeout_seconds = 15.0
        with ThreadPoolExecutor(max_workers=len(FAST_TOP_TICKERS)) as pool:
            future_map = {
                pool.submit(_compute_fast_top_item, ticker): ticker
                for ticker in FAST_TOP_TICKERS
            }

            for future, ticker in future_map.items():
                try:
                    items.append(future.result(timeout=per_ticker_timeout_seconds))
                except Exception as per_stock_exc:
                    logger.warning("Fast /signals/top fallback item for %s due to error/timeout: %s", ticker, per_stock_exc)
                    items.append(
                        TopSignalsItem(
                            ticker=ticker,
                            convergence_score=0.0,
                            convergence_label="NO_DATA",
                            radar_composite_score=0.0,
                            insider_signal_strength=0.0,
                            earnings_beat_pct=0.0,
                        )
                    )

        items.sort(key=lambda x: x.convergence_score, reverse=True)

        ranked_items: list[TopSignalsItem] = []
        for idx, item in enumerate(items):
            if idx < 3:
                ranked_label = "BUY"
            else:
                ranked_label = "SELL/AVOID"

            ranked_items.append(
                TopSignalsItem(
                    ticker=item.ticker,
                    convergence_score=item.convergence_score,
                    convergence_label=ranked_label,
                    radar_composite_score=item.radar_composite_score,
                    insider_signal_strength=item.insider_signal_strength,
                    earnings_beat_pct=item.earnings_beat_pct,
                )
            )

        return TopSignalsResponse(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            index_name="FAST_DEMO_6",
            scanned_count=len(FAST_TOP_TICKERS),
            returned_count=len(ranked_items),
            market_status=market,
            items=ranked_items,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/signals/top failed")
        raise HTTPException(status_code=500, detail=f"Top signals failed: {exc}") from exc


@router.get("/charts/{ticker}", response_model=ChartResponse)
def get_chart(ticker: str, use_vlm: bool = Query(default=True)) -> ChartResponse:
    """Return chart PNG as base64 together with chart pattern payload.

    Args:
        ticker: NSE ticker.
        use_vlm: Whether to include Gemini visual confirmation in chart run.

    Returns:
        ChartResponse with base64 image and pattern data.
    """
    from agents.chart_agent import analyze_stock

    try:
        norm_ticker = _normalize_ticker(ticker)
        chart_data = analyze_stock(norm_ticker, use_vlm=use_vlm)
        chart_path = str(chart_data.get("chart_path", ""))
        if not chart_path or not os.path.exists(chart_path):
            raise HTTPException(status_code=404, detail="Chart image not found")

        with open(chart_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        return ChartResponse(
            ticker=norm_ticker,
            chart_path=chart_path,
            chart_image_base64=encoded,
            patterns=chart_data.get("patterns", []),
            chart_confidence=float(chart_data.get("chart_confidence", 0.0)),
            overall_bias=str(chart_data.get("overall_bias", "neutral")),
            vlm_confirmed=_chart_vlm_confirmed(chart_data),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/charts failed")
        raise HTTPException(status_code=500, detail=f"Chart fetch failed: {exc}") from exc


@router.get("/radar", response_model=RadarResponse)
def get_radar(
    index_name: str = Query(default=DEFAULT_INDEX),
    top_n: int = Query(default=DEFAULT_TOP_N, ge=1, le=25),
) -> RadarResponse:
    """Return today's opportunity radar feed.

    Args:
        index_name: NSE index universe.
        top_n: Number of opportunities to return.

    Returns:
        RadarResponse containing ranked radar opportunities.
    """
    try:
        from agents.radar_agent import MAX_STOCK_SCAN_SECONDS
        from data.nse_fetcher import fetch_index_constituents

        started = time.perf_counter()
        hard_timeout_seconds = 27.5
        deadline = started + hard_timeout_seconds

        symbols = fetch_index_constituents(index_name)
        scanned_count = 0
        collected: list[dict[str, Any]] = []

        for sym in symbols:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                logger.warning("/api/radar reached 30s timeout, returning partial results")
                break

            per_stock_budget = min(MAX_STOCK_SCAN_SECONDS, max(0.3, remaining - 0.1))
            scanned_count += 1

            profile = _scan_stock_with_timeout(sym, per_stock_budget)
            if profile is None:
                continue

            if profile.total_signal_count >= 1:
                profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
                collected.append(profile_dict)

        collected.sort(key=lambda item: float(item.get("composite_score", 0.0)), reverse=True)
        opportunities = collected[:top_n]

        return RadarResponse(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            index_name=index_name,
            scanned_count=scanned_count,
            returned_count=len(opportunities),
            opportunities=opportunities,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/radar failed")
        raise HTTPException(status_code=500, detail=f"Radar feed failed: {exc}") from exc


@router.get("/market-status")
def market_status() -> dict[str, Any]:
    """Return pure datetime-based NSE market status."""
    return get_market_status()


def _sanitize_value(v: Any) -> Any:
    """Recursively sanitize a value: replace NaN/Inf floats with None at any depth."""
    import math

    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_sanitize_value(i) for i in v]
    return v


@router.get("/alerts/latest", response_model=AlertsLatestResponse)
def get_latest_alerts(limit: int = Query(default=10, ge=1, le=50)) -> AlertsLatestResponse:
    """Return the most recent autonomous pipeline alerts, deduplicated per ticker.

    Reads alerts_log.json from project root and returns the latest alert
    per unique ticker, sorted by convergence score descending.

    Args:
        limit: Maximum number of alerts to return.

    Returns:
        AlertsLatestResponse with count and alert items.
    """
    import json
    from pathlib import Path

    alerts_path = Path(__file__).resolve().parent.parent / "alerts_log.json"
    if not alerts_path.exists():
        return AlertsLatestResponse(count=0, alerts=[])

    try:
        with alerts_path.open("r", encoding="utf-8") as f:
            all_alerts = json.load(f)
    except (json.JSONDecodeError, OSError):
        return AlertsLatestResponse(count=0, alerts=[])

    if not isinstance(all_alerts, list):
        return AlertsLatestResponse(count=0, alerts=[])

    # Deep-sanitize every alert (NaN/Inf → None)
    all_alerts = [_sanitize_value(a) for a in all_alerts]

    # Deduplicate: keep only the most recent alert per ticker
    alerts_by_ticker: dict[str, dict] = {}
    for alert in all_alerts:
        ticker = alert.get("ticker")
        if not ticker:
            continue
        existing = alerts_by_ticker.get(ticker)
        if existing is None:
            alerts_by_ticker[ticker] = alert
        else:
            existing_ts = existing.get("timestamp", "")
            new_ts = alert.get("timestamp", "")
            if new_ts > existing_ts:
                alerts_by_ticker[ticker] = alert

    deduplicated = list(alerts_by_ticker.values())
    # Sort by convergence score descending
    deduplicated.sort(key=lambda x: x.get("convergence_score", 0) or 0, reverse=True)
    deduplicated = deduplicated[:limit]

    items = []
    for raw in deduplicated:
        try:
            items.append(AlertItem(**raw))
        except Exception:
            continue

    return AlertsLatestResponse(count=len(items), alerts=items)


@router.post("/query", response_model=QueryResponse)
def query_intelligence(payload: QueryRequest) -> QueryResponse:
    """Run natural language query using convergence + RAG + reasoning agents.

    Args:
        payload: Query text, optional ticker, and retrieval top_k.

    Returns:
        QueryResponse with convergence payload, retrieved context, and explanation.
    """
    try:
        return _run_query_pipeline(payload.query, payload.ticker, payload.top_k, portfolio=payload.portfolio)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/query failed")
        raise HTTPException(status_code=500, detail=f"Query pipeline failed: {exc}") from exc


@router.post("/voice", response_model=VoiceResponse)
async def query_voice(file: UploadFile = File(...), ticker: Optional[str] = None) -> VoiceResponse:
    """Transcribe audio and run the same query pipeline.

    Args:
        file: Uploaded audio file.
        ticker: Optional explicit ticker.

    Returns:
        VoiceResponse containing transcript and query response.
    """
    try:
        import tempfile
        import whisper

        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = whisper.load_model(WHISPER_MODEL_NAME)
        stt_result = model.transcribe(tmp_path)
        transcript = str(stt_result.get("text", "")).strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="Could not transcribe voice input")

        query_response = _run_query_pipeline(transcript, ticker=ticker, top_k=5)
        return VoiceResponse(transcript=transcript, query_response=query_response)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/voice failed")
        raise HTTPException(status_code=500, detail=f"Voice pipeline failed: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health endpoint for service status checks.

    Returns:
        HealthResponse with component availability checks.
    """
    components = {
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
        "chroma_path_exists": os.path.isdir("./chroma_db"),
    }
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(timespec="seconds"),
        version=APP_VERSION,
        components=components,
    )
