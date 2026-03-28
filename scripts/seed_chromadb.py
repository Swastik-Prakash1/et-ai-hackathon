"""seed_chromadb.py — Populate ChromaDB with financial content for 6 stocks.

For each stock, stores 4 document types:
  1. EARNINGS SUMMARY   — fetched via yfinance quarterly_income_stmt
  2. TECHNICAL SUMMARY   — from ta_processor scan_all_patterns
  3. SECTOR CONTEXT      — hardcoded sector-specific context
  4. RISK FACTORS        — company-specific risk paragraph

Run:  python -m scripts.seed_chromadb
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Resolve project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.nse_fetcher import fetch_company_info, fetch_earnings, fetch_ohlcv
from data.ta_processor import scan_all_patterns
from rag.embedder import embed_text
from rag.vector_store import add_documents, collection_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_chromadb")

WATCHLIST = ["SBIN", "OIL", "BHARTIARTL", "HINDUNILVR", "HCLTECH", "MARUTI"]
TODAY = datetime.now().strftime("%Y-%m-%d")


# ── Sector context (hardcoded) ─────────────────────────────────────────────

SECTOR_CONTEXT: dict[str, str] = {
    "SBIN": (
        "Indian banking sector is seeing strong credit growth driven by retail and MSME lending. "
        "PSU banks benefit from government capex push and improving asset quality. "
        "NIM compression remains a risk as RBI rate cycle peaks."
    ),
    "OIL": (
        "Indian energy sector outlook hinges on global crude prices and government subsidy policy. "
        "Upstream explorers like OIL benefit from higher realizations but face production plateau risks. "
        "Energy transition and renewables push could reshape long-term demand."
    ),
    "BHARTIARTL": (
        "Indian telecom sector is a 3-player oligopoly with stable ARPU growth trajectory. "
        "5G rollout and fixed broadband expansion drive capex but also future revenue streams. "
        "Regulatory clarity on spectrum pricing supports long-term visibility."
    ),
    "HINDUNILVR": (
        "FMCG sector faces rural demand recovery as a key growth lever after two weak years. "
        "Input cost deflation in palm oil and packaging is expanding gross margins. "
        "Premium portfolio mix shift supports operating leverage despite volume pressure."
    ),
    "HCLTECH": (
        "Indian IT services sector navigating client spending caution amid macro uncertainty. "
        "Deal pipelines remain strong in cloud migration and AI/GenAI transformation projects. "
        "Rupee depreciation provides natural tailwind to margins and competitiveness."
    ),
    "MARUTI": (
        "Indian auto sector is in a multi-year upcycle driven by SUV premiumization and rural recovery. "
        "EV transition timeline remains extended for mass market, benefiting ICE-dominant players. "
        "Raw material cost moderation and operating leverage support margin expansion."
    ),
}


# ── Risk factors (company-specific) ────────────────────────────────────────

RISK_FACTORS: dict[str, str] = {
    "SBIN": (
        "Key risks for SBI include exposure to stressed infrastructure loans and potential slippage from "
        "MSME restructured book. Government stake dilution could pressure stock price. Rising deposit costs "
        "may compress net interest margins. Asset quality deterioration in unsecured retail lending and "
        "agricultural NPAs during monsoon shortfall years remain cyclical risks."
    ),
    "OIL": (
        "Oil India faces production stagnation in mature Assam fields with declining reserves. "
        "Global crude price volatility directly impacts realizations and profitability. Government windfall "
        "tax imposition creates earnings unpredictability. Geopolitical tensions in the Northeast and "
        "environmental activism around fossil fuel expansion pose operational risks."
    ),
    "BHARTIARTL": (
        "Bharti Airtel carries significant debt from spectrum acquisitions and 5G capex. Aggressive pricing "
        "by Jio could limit ARPU growth. Africa operations face currency volatility and regulatory risks "
        "across multiple jurisdictions. Satellite broadband entrants like Starlink could disrupt rural "
        "connectivity plans."
    ),
    "HINDUNILVR": (
        "HUL faces persistent volume growth challenges as consumers downgrade to regional brands. "
        "Palm oil price spikes can rapidly compress gross margins. Increasing GST scrutiny and trademark "
        "disputes add regulatory overhead. D2C disruptors and quick commerce platforms are fragmenting "
        "market share in urban metros."
    ),
    "HCLTECH": (
        "HCL Tech is exposed to discretionary IT spending cuts during global recession scenarios. Client "
        "concentration risk with top accounts contributing disproportionate revenue. Visa restrictions and "
        "onshore hiring costs in the US and Europe pressure margins. GenAI-driven automation could reduce "
        "billable headcount in maintenance and support contracts."
    ),
    "MARUTI": (
        "Maruti Suzuki faces margin pressure from rising steel and precious metal costs for emission "
        "compliance. EV disruption threatens the core petrol/diesel small car segment long-term. Intense "
        "competition from Hyundai, Tata, and Mahindra in the SUV segment is eroding market share. "
        "Export markets face currency risks and region-specific regulatory challenges."
    ),
}


# ── Document builders ──────────────────────────────────────────────────────

def _build_earnings_summary(ticker: str) -> str:
    """Build earnings summary text from yfinance quarterly data."""
    info = fetch_company_info(ticker)
    earnings = fetch_earnings(ticker)

    company = str(info.get("longName") or info.get("shortName") or ticker)

    parts = [f"{company} ({ticker})"]

    # Latest beat/miss
    latest_beat = earnings.get("latest_beat")
    latest_beat_pct = earnings.get("latest_beat_pct")
    if latest_beat is not None and latest_beat_pct is not None:
        outcome = "beat" if bool(latest_beat) else "missed"
        parts.append(f"Latest quarter: EPS {outcome} estimates by {abs(float(latest_beat_pct)):.1f}%.")
    else:
        parts.append("Latest quarter earnings data not available from yfinance.")

    # Revenue and net income from quarterly_income_stmt
    qdf = earnings.get("quarterly_earnings")
    if hasattr(qdf, "empty") and not qdf.empty:
        cols = list(qdf.columns)
        if len(cols) >= 2:
            latest_col = cols[0]
            prev_col = cols[1]
            try:
                latest_rev = float(qdf.loc["Total Revenue", latest_col])
                prev_rev = float(qdf.loc["Total Revenue", prev_col])
                if prev_rev > 0:
                    rev_growth = ((latest_rev - prev_rev) / abs(prev_rev)) * 100
                    parts.append(f"Revenue grew {rev_growth:.1f}% QoQ.")
            except Exception:
                pass
            try:
                latest_ni = float(qdf.loc["Net Income", latest_col])
                prev_ni = float(qdf.loc["Net Income", prev_col])
                if prev_ni > 0:
                    ni_growth = ((latest_ni - prev_ni) / abs(prev_ni)) * 100
                    parts.append(f"Net income changed {ni_growth:.1f}% QoQ.")
            except Exception:
                pass

    # Fundamentals
    pe = info.get("trailingPE")
    if pe is not None:
        parts.append(f"Trailing PE: {float(pe):.1f}.")

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        parts.append(f"Annual revenue growth: {float(rev_growth) * 100:.1f}%.")

    return " ".join(parts)


def _build_technical_summary(ticker: str) -> str:
    """Build technical summary from ta_processor scan."""
    try:
        df = fetch_ohlcv(ticker, period="2y", interval="1d")
        if df.empty:
            return f"{ticker}: No OHLCV data available for technical analysis."

        scan = scan_all_patterns(df.copy(), symbol=ticker)
        patterns = scan.get("patterns", [])
        bias = scan.get("overall_bias", "neutral")
        chart_conf = scan.get("chart_confidence", 0)

        if not patterns:
            return (
                f"{ticker}: No active chart patterns detected. "
                f"Overall bias: {bias}. Chart confidence: {chart_conf:.0f}%."
            )

        lines = [f"{ticker} Technical Summary (as of {TODAY}):"]
        for p in patterns[:3]:
            name = p.get("pattern_name", "Unknown")
            ptype = p.get("pattern_type", "neutral")
            conf = p.get("confidence", 0)
            explanation = p.get("explanation", "")
            win_rate_val = p.get("win_rate")
            occurrences = p.get("occurrences")

            line = f"{name} ({ptype}, {conf:.0f}% confidence)"
            if win_rate_val is not None and occurrences is not None:
                line += f" - Win rate: {float(win_rate_val) * 100:.0f}% over {occurrences} occurrences"
            if explanation:
                line += f". {explanation}"
            lines.append(line)

        # Add indicator snapshot
        indicators = scan.get("indicators", {})
        rsi = indicators.get("RSI_14")
        if rsi is not None:
            zone = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
            lines.append(f"RSI(14): {rsi:.1f} ({zone}).")

        lines.append(f"Overall bias: {bias}. Chart confidence: {chart_conf:.0f}%.")
        return " ".join(lines)

    except Exception as exc:
        logger.warning("Technical summary failed for %s: %s", ticker, exc)
        return f"{ticker}: Technical analysis unavailable due to data error."


# ── Main seeding logic ─────────────────────────────────────────────────────

def seed_all() -> int:
    """Seed ChromaDB with 4 document types for each watchlist stock.

    Returns the total number of documents seeded.
    """
    all_docs: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict[str, Any]] = []

    for ticker in WATCHLIST:
        print(f"  Preparing documents for {ticker}...")

        # 1. Earnings summary
        earnings_text = _build_earnings_summary(ticker)
        all_docs.append(earnings_text)
        all_ids.append(f"seed_{ticker.lower()}_earnings")
        all_metas.append({
            "stock": ticker,
            "type": "earnings_summary",
            "date": TODAY,
            "source": "yfinance quarterly data",
        })
        print(f"    [1/4] Earnings summary: {len(earnings_text)} chars")

        # 2. Technical summary
        tech_text = _build_technical_summary(ticker)
        all_docs.append(tech_text)
        all_ids.append(f"seed_{ticker.lower()}_technical")
        all_metas.append({
            "stock": ticker,
            "type": "technical_summary",
            "date": TODAY,
            "source": "ta_processor pattern scan",
        })
        print(f"    [2/4] Technical summary: {len(tech_text)} chars")

        # 3. Sector context
        sector_text = SECTOR_CONTEXT.get(ticker, f"{ticker}: No sector context available.")
        all_docs.append(sector_text)
        all_ids.append(f"seed_{ticker.lower()}_sector")
        all_metas.append({
            "stock": ticker,
            "type": "sector_context",
            "date": TODAY,
            "source": "curated sector analysis",
        })
        print(f"    [3/4] Sector context: {len(sector_text)} chars")

        # 4. Risk factors
        risk_text = RISK_FACTORS.get(ticker, f"{ticker}: No specific risk factors documented.")
        all_docs.append(risk_text)
        all_ids.append(f"seed_{ticker.lower()}_risk")
        all_metas.append({
            "stock": ticker,
            "type": "risk_factors",
            "date": TODAY,
            "source": "curated risk analysis",
        })
        print(f"    [4/4] Risk factors: {len(risk_text)} chars")

    # Embed all documents
    print(f"\n  Embedding {len(all_docs)} documents...")
    all_embeddings = [embed_text(doc).tolist() for doc in all_docs]
    print(f"  Embeddings computed ({len(all_embeddings)} vectors)")

    # Upsert into ChromaDB
    print("  Upserting into ChromaDB (market_intelligence collection)...")
    result = add_documents(
        documents=all_docs,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metas,
        collection_name="market_intelligence",
    )

    if result.get("ok"):
        total = collection_count("market_intelligence")
        print(f"\n  Seeded {len(all_docs)} documents for {len(WATCHLIST)} stocks")
        print(f"  Collection total: {total} documents")
        return len(all_docs)
    else:
        print(f"\n  ERROR: Upsert failed — {result.get('error', 'unknown')}")
        return 0


if __name__ == "__main__":
    print("=" * 60)
    print("  ChromaDB Seeder — FINANCIAL_INTEL")
    print(f"  Stocks: {', '.join(WATCHLIST)}")
    print(f"  Date: {TODAY}")
    print("=" * 60)
    print()

    count = seed_all()

    print()
    print("=" * 60)
    if count > 0:
        print(f"  SUCCESS: Seeded {count} documents for {len(WATCHLIST)} stocks")
    else:
        print("  FAILED: No documents were seeded")
    print("=" * 60)
