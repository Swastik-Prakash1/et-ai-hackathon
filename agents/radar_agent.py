"""
radar_agent.py — Opportunity Radar Agent
==========================================
Agent 2 of 5. Continuously scans the NSE universe for actionable
fundamental signals that most retail investors miss.

Signal Sources:
  1. Bulk Deals   — Institutional activity (qty > 0.5% of listed shares)
  2. Block Deals  — Large trades (5L+ shares or ₹10cr+ value)
  3. Insider Trades — SEBI PIT disclosures (PROMOTER BUYING = strongest signal)
  4. Earnings     — Quarterly results vs analyst estimates (beat/miss %)

For each stock with active signals, the agent computes a
signal_strength score (0.0 to 1.0) and ranks all opportunities
by actionability. This ranked output feeds directly into the
Convergence Agent.

This is NOT a summarizer — it's a SIGNAL FINDER that surfaces
missed opportunities retail investors would never see.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict, field

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

MAX_STOCK_SCAN_SECONDS = 5.0
NSE_PROBE_TIMEOUT_SECONDS = 0.75


def _is_http_403_error(exc: Exception) -> bool:
    """Return True if exception maps to an HTTP 403 response."""
    if isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) == 403:
        return True
    text = str(exc).lower()
    return "403" in text and "forbidden" in text


def _probe_nse_for_403() -> bool:
    """Single no-retry probe to determine if NSE API is blocking requests."""
    from data.nse_fetcher import NSE_HEADERS

    url = "https://www.nseindia.com/"
    try:
        resp = requests.get(url, headers=NSE_HEADERS, timeout=NSE_PROBE_TIMEOUT_SECONDS)
        return resp.status_code == 403
    except requests.RequestException:
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RadarSignal:
    """A single actionable signal detected by the Radar Agent."""
    symbol: str
    signal_type: str          # "bulk_deal" | "block_deal" | "insider_buy" | "insider_sell" | "earnings_beat" | "earnings_miss"
    signal_category: str      # "deals" | "insider" | "earnings"
    direction: str            # "bullish" | "bearish" | "neutral"
    strength: float           # 0.0 to 1.0
    headline: str             # Short one-liner for the feed
    detail: str               # Plain-English explanation
    data: dict                # Raw source data for RAG citation
    source: str               # "NSE Bulk Deals", "SEBI PIT", "yfinance Earnings"
    detected_date: str        # ISO date

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StockRadarProfile:
    """Aggregated radar profile for a single stock — all signals combined."""
    symbol: str
    company_name: str
    signals: list[RadarSignal] = field(default_factory=list)
    total_signal_count: int = 0
    bullish_signals: int = 0
    bearish_signals: int = 0
    composite_score: float = 0.0       # Weighted sum of all signal strengths
    has_insider_buying: bool = False
    has_earnings_beat: bool = False
    has_deal_activity: bool = False
    earnings_beat_pct: float = 0.0
    insider_signal_strength: float = 0.0
    scan_time: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [s if isinstance(s, dict) else s.to_dict() for s in self.signals]
        return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE: Scan a single stock for all radar signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_stock(
    symbol: str,
    allow_nse_checks: bool = True,
    nse_blocked_override: Optional[bool] = None,
    max_scan_seconds: float = MAX_STOCK_SCAN_SECONDS,
) -> StockRadarProfile:
    """
    Run all radar checks on a single stock.

    Checks:
      1. Recent bulk/block deals involving this stock
      2. Insider/promoter trade activity (SEBI PIT)
      3. Latest quarterly earnings beat/miss

    Returns:
        StockRadarProfile with all detected signals and composite score.
    """
    from data.nse_fetcher import (
        fetch_company_info,
        fetch_earnings,
        detect_promoter_buying,
        fetch_all_deals,
    )

    clean = symbol.strip().upper().replace(".NS", "")
    logger.info("Radar scanning %s...", clean)

    # Get company name
    info = fetch_company_info(clean)
    company_name = info.get("longName", info.get("shortName", clean))

    profile = StockRadarProfile(
        symbol=clean,
        company_name=company_name,
        scan_time=datetime.now().isoformat(),
    )
    signals = []

    scan_start = time.perf_counter()
    if nse_blocked_override is not None:
        nse_blocked = bool(nse_blocked_override)
    else:
        nse_blocked = _probe_nse_for_403() if allow_nse_checks else True

    # ── 1. Deals + 2. Insider (NSE-backed, skipped on 403) ──
    if nse_blocked:
        logger.info("NSE returned 403 for %s; using earnings-only radar path", clean)
    else:
        if (time.perf_counter() - scan_start) < max_scan_seconds:
            try:
                deal_signals = _check_deals(clean)
                signals.extend(deal_signals)
                if deal_signals:
                    profile.has_deal_activity = True
            except Exception as exc:
                if _is_http_403_error(exc):
                    nse_blocked = True
                    logger.info("NSE 403 during deals for %s; skipping remaining NSE checks", clean)
                else:
                    logger.warning("Deals check failed for %s: %s", clean, exc)

        if not nse_blocked and (time.perf_counter() - scan_start) < max_scan_seconds:
            try:
                insider_signals = _check_insider_trades(clean)
                signals.extend(insider_signals)
                for sig in insider_signals:
                    if sig.signal_type == "insider_buy":
                        profile.has_insider_buying = True
                        profile.insider_signal_strength = max(
                            profile.insider_signal_strength, sig.strength
                        )
            except Exception as exc:
                if _is_http_403_error(exc):
                    logger.info("NSE 403 during insider checks for %s; forcing earnings-only", clean)
                else:
                    logger.warning("Insider check failed for %s: %s", clean, exc)

    # ── 3. Earnings (yfinance-backed, always attempted) ─────
    earnings_signals = _check_earnings(clean)
    signals.extend(earnings_signals)
    for sig in earnings_signals:
        if sig.signal_type == "earnings_beat":
            profile.has_earnings_beat = True
            profile.earnings_beat_pct = sig.data.get("beat_pct", 0.0)

    # ── Aggregate ───────────────────────────────────────────
    profile.signals = signals
    profile.total_signal_count = len(signals)
    profile.bullish_signals = sum(1 for s in signals if s.direction == "bullish")
    profile.bearish_signals = sum(1 for s in signals if s.direction == "bearish")

    # Composite score: weighted sum of signal strengths
    # Insider buying gets extra weight (it's the strongest fundamental signal)
    if signals:
        weighted = []
        for s in signals:
            weight = 1.5 if s.signal_category == "insider" else 1.0
            if s.direction == "bearish":
                weighted.append(-s.strength * weight)
            else:
                weighted.append(s.strength * weight)
        raw_score = sum(weighted) / len(weighted)
        # Normalize to 0-1
        profile.composite_score = round(max(0, min(1, (raw_score + 1) / 2)), 3)

    logger.info(
        "Radar %s: %d signals (%dB %dR), composite=%.3f, elapsed=%.2fs",
        clean, len(signals), profile.bullish_signals, profile.bearish_signals,
        profile.composite_score, time.perf_counter() - scan_start,
    )
    return profile


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL CHECK 1: Bulk & Block Deals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_deals(symbol: str) -> list[RadarSignal]:
    """Check for recent bulk and block deal activity for this stock."""
    from data.nse_fetcher import fetch_all_deals

    signals = []
    try:
        deals = fetch_all_deals()
    except Exception as exc:
        logger.warning("Failed to fetch deals: %s", exc)
        return signals

    if deals.empty:
        return signals

    # Filter to this stock
    mask = deals["symbol"].str.upper() == symbol.upper()
    stock_deals = deals[mask]

    if stock_deals.empty:
        return signals

    for _, row in stock_deals.iterrows():
        deal_class = row.get("deal_class", "BULK")
        deal_type = row.get("deal_type", "").strip().upper()
        qty = row.get("quantity", 0)
        price = row.get("price", 0)
        client = row.get("client_name", "Unknown")
        trade_date = row.get("trade_date", "")
        value_cr = (qty * price) / 1e7  # Convert to crores

        is_buy = deal_type in ("BUY", "B")
        direction = "bullish" if is_buy else "bearish"

        # Strength: based on deal value (₹50cr+ = max)
        strength = min(value_cr / 50.0, 1.0)

        headline = (
            f"{'Bulk' if deal_class == 'BULK' else 'Block'} "
            f"{'Buy' if is_buy else 'Sell'}: {client} — "
            f"{qty:,.0f} shares @ ₹{price:,.0f} (₹{value_cr:.1f} Cr)"
        )

        detail = (
            f"{client} executed a {deal_class.lower()} deal "
            f"{'buying' if is_buy else 'selling'} {qty:,.0f} shares of {symbol} "
            f"at ₹{price:,.0f} per share, worth approximately ₹{value_cr:.1f} crore. "
            f"{'Large institutional buying like this often signals confidence in the stock.' if is_buy else 'Large institutional selling may indicate bearish sentiment.'}"
        )

        signals.append(RadarSignal(
            symbol=symbol,
            signal_type=f"{'bulk' if deal_class == 'BULK' else 'block'}_deal",
            signal_category="deals",
            direction=direction,
            strength=round(strength, 3),
            headline=headline,
            detail=detail,
            data={
                "deal_class": deal_class,
                "deal_type": deal_type,
                "client": client,
                "quantity": qty,
                "price": price,
                "value_crores": round(value_cr, 2),
                "trade_date": trade_date,
            },
            source=f"NSE {deal_class.title()} Deals",
            detected_date=trade_date or datetime.now().strftime("%Y-%m-%d"),
        ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL CHECK 2: Insider / Promoter Trades
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_insider_trades(symbol: str) -> list[RadarSignal]:
    """
    Check for recent insider/promoter trading activity.
    PROMOTER BUYING is the single strongest fundamental signal
    in our convergence model (weight 0.30).
    """
    from data.nse_fetcher import detect_promoter_buying, fetch_insider_trades

    signals = []

    # High-level promoter signal
    try:
        promo = detect_promoter_buying(symbol)
    except Exception as exc:
        logger.warning("Failed to check insider trades for %s: %s", symbol, exc)
        return signals

    if promo["promoter_buying"]:
        net_val_cr = promo["net_buy_value_lakhs"] / 100  # lakhs → crores
        signals.append(RadarSignal(
            symbol=symbol,
            signal_type="insider_buy",
            signal_category="insider",
            direction="bullish",
            strength=promo["signal_strength"],
            headline=(
                f"Promoter Net Buying: ₹{net_val_cr:.1f} Cr "
                f"({promo['net_buy_quantity']:,.0f} shares)"
            ),
            detail=(
                f"Company promoters have been net buyers of {symbol} shares recently, "
                f"purchasing a net {promo['net_buy_quantity']:,.0f} shares worth approximately "
                f"₹{net_val_cr:.1f} crore. When company insiders buy with their own money, "
                f"it's one of the strongest signals that they believe the stock is undervalued. "
                f"This is the single most reliable fundamental indicator in market research."
            ),
            data={
                "net_buy_quantity": promo["net_buy_quantity"],
                "net_buy_value_crores": round(net_val_cr, 2),
                "signal_strength": promo["signal_strength"],
                "recent_trades_count": len(promo.get("recent_trades", [])),
            },
            source="SEBI PIT Disclosures (via NSE)",
            detected_date=datetime.now().strftime("%Y-%m-%d"),
        ))

    elif promo["net_buy_quantity"] < 0:
        # Net selling by promoters — bearish signal
        net_val_cr = abs(promo["net_buy_value_lakhs"]) / 100
        sell_strength = min(net_val_cr / 10.0, 1.0)

        signals.append(RadarSignal(
            symbol=symbol,
            signal_type="insider_sell",
            signal_category="insider",
            direction="bearish",
            strength=round(sell_strength, 3),
            headline=(
                f"Promoter Net Selling: ₹{net_val_cr:.1f} Cr "
                f"({abs(promo['net_buy_quantity']):,.0f} shares)"
            ),
            detail=(
                f"Company promoters have been net sellers of {symbol} shares, "
                f"disposing of {abs(promo['net_buy_quantity']):,.0f} shares worth "
                f"approximately ₹{net_val_cr:.1f} crore. While promoter selling doesn't "
                f"always mean bad news (they may need liquidity), it's worth watching "
                f"alongside other signals."
            ),
            data={
                "net_sell_quantity": abs(promo["net_buy_quantity"]),
                "net_sell_value_crores": round(net_val_cr, 2),
            },
            source="SEBI PIT Disclosures (via NSE)",
            detected_date=datetime.now().strftime("%Y-%m-%d"),
        ))

    # Also check for notable individual insider trades
    try:
        insider_df = fetch_insider_trades(symbol)
        if not insider_df.empty:
            # Surface large individual trades (top 3 by value)
            sorted_trades = insider_df.nlargest(
                min(3, len(insider_df)), "value_lakhs"
            )
            for _, row in sorted_trades.iterrows():
                val_lakhs = row.get("value_lakhs", 0)
                if val_lakhs < 10:  # Skip tiny trades (< ₹10 lakh)
                    continue

                name = row.get("acquirer_name", "Unknown")
                category = row.get("category", "")
                txn_type = row.get("transaction_type", "")
                qty = row.get("quantity", 0)
                date = row.get("date_of_trade", "")
                if isinstance(date, pd.Timestamp):
                    date = date.strftime("%Y-%m-%d")

                is_buy = "buy" in str(txn_type).lower() or "acquisition" in str(txn_type).lower()
                val_cr = val_lakhs / 100

                signals.append(RadarSignal(
                    symbol=symbol,
                    signal_type="insider_buy" if is_buy else "insider_sell",
                    signal_category="insider",
                    direction="bullish" if is_buy else "bearish",
                    strength=round(min(val_cr / 5.0, 0.8), 3),
                    headline=(
                        f"{category}: {name} {'bought' if is_buy else 'sold'} "
                        f"{qty:,.0f} shares (₹{val_cr:.1f} Cr) on {date}"
                    ),
                    detail=(
                        f"{name} ({category}) {'acquired' if is_buy else 'sold'} "
                        f"{qty:,.0f} shares of {symbol} worth ₹{val_cr:.1f} crore "
                        f"on {date}."
                    ),
                    data={
                        "acquirer": name,
                        "category": category,
                        "transaction_type": txn_type,
                        "quantity": qty,
                        "value_crores": round(val_cr, 2),
                        "date": date,
                    },
                    source="SEBI PIT Disclosures (via NSE)",
                    detected_date=str(date),
                ))
    except Exception as exc:
        logger.warning("Failed to fetch detailed insider trades for %s: %s", symbol, exc)

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIGNAL CHECK 3: Earnings Beat / Miss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_earnings(symbol: str) -> list[RadarSignal]:
    """
    Check if the latest quarterly earnings beat or missed estimates.
    Beat% is normalized for the convergence score.
    """
    from data.nse_fetcher import fetch_earnings

    signals = []
    try:
        earn = fetch_earnings(symbol)
    except Exception as exc:
        logger.warning("Failed to fetch earnings for %s: %s", symbol, exc)
        return signals

    if "latest_beat" not in earn:
        return signals

    is_beat = earn["latest_beat"]
    beat_pct = earn.get("latest_beat_pct", 0.0)

    if is_beat:
        # Strength: 10%+ beat = max, proportional below
        strength = min(abs(beat_pct) / 15.0, 1.0)
        signals.append(RadarSignal(
            symbol=symbol,
            signal_type="earnings_beat",
            signal_category="earnings",
            direction="bullish",
            strength=round(strength, 3),
            headline=f"Earnings Beat: +{beat_pct:.1f}% vs estimates",
            detail=(
                f"{symbol} reported quarterly earnings that beat analyst estimates "
                f"by {beat_pct:.1f}%. When a company earns more than expected, it "
                f"often leads to positive price momentum as the market adjusts its "
                f"valuation upward."
            ),
            data={
                "beat": True,
                "beat_pct": round(beat_pct, 2),
            },
            source="yfinance Quarterly Earnings",
            detected_date=datetime.now().strftime("%Y-%m-%d"),
        ))
    else:
        # Earnings miss
        strength = min(abs(beat_pct) / 15.0, 1.0)
        signals.append(RadarSignal(
            symbol=symbol,
            signal_type="earnings_miss",
            signal_category="earnings",
            direction="bearish",
            strength=round(strength, 3),
            headline=f"Earnings Miss: {beat_pct:.1f}% vs estimates",
            detail=(
                f"{symbol} reported quarterly earnings that missed analyst estimates "
                f"by {abs(beat_pct):.1f}%. Earnings misses can lead to negative price "
                f"pressure as the market reassesses the company's growth trajectory."
            ),
            data={
                "beat": False,
                "beat_pct": round(beat_pct, 2),
            },
            source="yfinance Quarterly Earnings",
            detected_date=datetime.now().strftime("%Y-%m-%d"),
        ))

    return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIVERSE SCAN — Run radar on many stocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_universe(
    symbols: Optional[list[str]] = None,
    index_name: str = "NIFTY 500",
    top_n: int = 20,
    min_signals: int = 1,
    delay_between: float = 0.5,
) -> dict:
    """
    Run radar on every stock in the universe and return the top
    opportunities ranked by composite score.

    Args:
        symbols:       Explicit list of tickers. If None, fetches index.
        index_name:    Index to use if symbols is None.
        top_n:         Number of top opportunities to return.
        min_signals:   Minimum signal count to include in results.
        delay_between: Seconds between scans (rate limiting).

    Returns:
        {
            "scan_time": str,
            "universe_size": int,
            "stocks_scanned": int,
            "stocks_with_signals": int,
            "top_opportunities": list[dict],  # Ranked by composite_score
            "deal_activity": list[dict],      # All deal signals
            "insider_activity": list[dict],   # All insider signals
            "earnings_signals": list[dict],   # All earnings signals
        }
    """
    from data.nse_fetcher import fetch_index_constituents

    if symbols is None:
        symbols = fetch_index_constituents(index_name)

    logger.info("Radar scanning %d stocks...", len(symbols))
    nse_blocked = _probe_nse_for_403()
    if nse_blocked:
        logger.info("NSE probe returned 403; scanning in earnings-only mode")

    all_profiles = []
    deal_signals = []
    insider_signals = []
    earnings_signals = []

    for i, sym in enumerate(symbols, 1):
        try:
            profile = scan_stock(
                sym,
                allow_nse_checks=not nse_blocked,
                nse_blocked_override=nse_blocked,
                max_scan_seconds=MAX_STOCK_SCAN_SECONDS,
            )
            if profile.total_signal_count >= min_signals:
                all_profiles.append(profile)

            # Categorize signals
            for sig in profile.signals:
                sig_dict = sig.to_dict() if isinstance(sig, RadarSignal) else sig
                if sig_dict["signal_category"] == "deals":
                    deal_signals.append(sig_dict)
                elif sig_dict["signal_category"] == "insider":
                    insider_signals.append(sig_dict)
                elif sig_dict["signal_category"] == "earnings":
                    earnings_signals.append(sig_dict)

        except Exception as exc:
            logger.warning("Radar failed for %s: %s", sym, exc)

        if i % 25 == 0:
            logger.info("Radar progress: %d/%d stocks", i, len(symbols))

        time.sleep(delay_between)

    # Rank by composite score
    all_profiles.sort(key=lambda p: p.composite_score, reverse=True)
    top = all_profiles[:top_n]

    result = {
        "scan_time": datetime.now().isoformat(),
        "universe_size": len(symbols),
        "stocks_scanned": len(symbols),
        "stocks_with_signals": len(all_profiles),
        "top_opportunities": [p.to_dict() for p in top],
        "deal_activity": deal_signals,
        "insider_activity": insider_signals,
        "earnings_signals": earnings_signals,
    }

    logger.info(
        "Radar scan complete: %d/%d stocks with signals, top composite=%.3f",
        len(all_profiles), len(symbols),
        top[0].composite_score if top else 0,
    )
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK SCAN — Scan a small list for demo / testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def quick_scan(
    symbols: Optional[list[str]] = None,
) -> list[dict]:
    """
    Quick radar scan on a small list of stocks (default: NIFTY50 top 10).
    Returns list of StockRadarProfile dicts sorted by composite score.
    """
    if symbols is None:
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "TATAMOTORS", "SBIN", "BHARTIARTL", "ITC", "LT",
        ]

    profiles = []
    for sym in symbols:
        try:
            p = scan_stock(sym)
            profiles.append(p)
        except Exception as exc:
            logger.warning("Quick scan failed for %s: %s", sym, exc)

    profiles.sort(key=lambda p: p.composite_score, reverse=True)
    return [p.to_dict() for p in profiles]


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
    print("RADAR AGENT — Smoke Test")
    print("=" * 70)

    # Test single stock scan
    tickers = ["RELIANCE", "TATAMOTORS", "SBIN"]

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"Scanning {ticker}...")
        profile = scan_stock(ticker)

        print(f"  Company: {profile.company_name}")
        print(f"  Signals: {profile.total_signal_count} "
              f"({profile.bullish_signals}B {profile.bearish_signals}R)")
        print(f"  Composite Score: {profile.composite_score:.3f}")
        print(f"  Insider Buying: {profile.has_insider_buying}")
        print(f"  Earnings Beat: {profile.has_earnings_beat}")
        print(f"  Deal Activity: {profile.has_deal_activity}")

        for sig in profile.signals:
            emoji = "🟢" if sig.direction == "bullish" else "🔴"
            print(f"    {emoji} [{sig.signal_type}] {sig.headline}")

    print("\n" + "=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
