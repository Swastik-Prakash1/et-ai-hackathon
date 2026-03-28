"""
nse_fetcher.py — Complete NSE Market Data Pipeline
===================================================
Fetches OHLCV history, fundamentals, earnings, bulk/block deals,
insider trades, and live quotes for the entire NSE universe.

Data Sources:
  - yfinance: OHLCV history, company info, quarterly earnings
  - nsepython: Bulk deals, block deals, live quotes, index constituents

All functions return clean DataFrames or dicts. Every API call is
wrapped in retry logic with exponential backoff. No placeholders.
"""

import os
import time
import logging
import datetime as dt
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pytz
import yfinance as yf
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NSE_SUFFIX = ".NS"
DEFAULT_PERIOD = "2y"  # 2 years of history for backtesting
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds — doubles each retry
REQUEST_TIMEOUT = 30  # seconds

_ohlcv_cache = {}  # {ticker: {"df": DataFrame, "timestamp": float}}
CACHE_TTL = 300    # 5 minutes

# NSE API headers (public endpoints, no auth needed)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# NIFTY 500 — the tradable NSE universe
# This list can be refreshed via fetch_index_constituents("NIFTY 500")
# Hardcoded top-50 as fallback so the system always has a working universe
NIFTY50_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "HCLTECH", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO",
    "NESTLEIND", "TECHM", "NTPC", "TATAMOTORS", "POWERGRID",
    "M&M", "ONGC", "JSWSTEEL", "TATASTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "GRASIM", "BRITANNIA",
    "CIPLA", "EICHERMOT", "DIVISLAB", "DRREDDY", "BPCL",
    "APOLLOHOSP", "TATACONSUM", "SBILIFE", "HDFCLIFE", "HEROMOTOCO",
    "INDUSINDBK", "HINDALCO", "BAJAJ-AUTO", "LTIM", "SHRIRAMFIN",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY — Retry wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _retry(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Execute `func` with exponential-backoff retries."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            wait = RETRY_DELAY_BASE ** attempt
            logger.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %ds",
                attempt, max_retries, func.__name__, exc, wait,
            )
            time.sleep(wait)
    logger.error("All %d retries exhausted for %s: %s", max_retries, func.__name__, last_error)
    raise last_error  # type: ignore[misc]


def _nse_ticker(symbol: str) -> str:
    """Ensure symbol has .NS suffix for yfinance."""
    symbol = symbol.strip().upper()
    if not symbol.endswith(NSE_SUFFIX):
        symbol += NSE_SUFFIX
    return symbol


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. OHLCV HISTORY (yfinance)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data for an NSE stock.

    Args:
        symbol:   NSE ticker (e.g. "RELIANCE" or "RELIANCE.NS")
        period:   yfinance period string ("1y", "2y", "5y", "max")
        interval: candle interval ("1d", "1wk", "1mo")

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex (timezone-naive, IST dates).
        Empty DataFrame if fetch fails after retries.
    """
    ticker_str = _nse_ticker(symbol)

    cached = _ohlcv_cache.get(ticker_str)
    if cached:
        age = time.time() - float(cached.get("timestamp", 0.0))
        cached_df = cached.get("df")
        if age < CACHE_TTL and cached_df is not None:
            logger.info("Cache hit for %s OHLCV (age=%.1fs)", ticker_str, age)
            return cached_df.copy()

    def _fetch():
        t = yf.Ticker(ticker_str)
        df = t.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {ticker_str} (period={period})")
        return df

    try:
        df = _retry(_fetch)
    except Exception as exc:
        logger.error("Failed to fetch OHLCV for %s: %s", ticker_str, exc)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Clean up — drop dividends/stock splits columns, localize index
    for col in ["Dividends", "Stock Splits", "Capital Gains"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ist = pytz.timezone("Asia/Kolkata")
    today_ist = dt.datetime.now(ist).date()
    last_candle_date = df.index[-1].date() if not df.empty else None
    days_stale = (today_ist - last_candle_date).days if last_candle_date else 999

    df.attrs["days_stale"] = int(days_stale)
    df.attrs["last_trading_date"] = str(last_candle_date) if last_candle_date else "unknown"
    df.attrs["is_stale"] = bool(days_stale > 1)

    df.index.name = "Date"
    _ohlcv_cache[ticker_str] = {"df": df.copy(), "timestamp": time.time()}

    logger.info(
        "Fetched %d candles for %s (%s → %s)",
        len(df), ticker_str,
        df.index[0].strftime("%Y-%m-%d"),
        df.index[-1].strftime("%Y-%m-%d"),
    )
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. COMPANY INFO / FUNDAMENTALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_company_info(symbol: str) -> dict:
    """
    Fetch fundamental data for an NSE stock — sector, market cap,
    P/E, 52-week range, etc.

    Returns:
        Dict of key fundamentals. Empty dict on failure.
    """
    ticker_str = _nse_ticker(symbol)

    def _fetch():
        t = yf.Ticker(ticker_str)
        info = t.info
        if not info or "regularMarketPrice" not in info:
            raise ValueError(f"No info data for {ticker_str}")
        return info

    try:
        raw = _retry(_fetch)
    except Exception as exc:
        logger.error("Failed to fetch info for %s: %s", ticker_str, exc)
        return {}

    # Extract the fields that matter for our agents
    keys_of_interest = [
        "shortName", "longName", "sector", "industry",
        "marketCap", "enterpriseValue",
        "trailingPE", "forwardPE", "priceToBook",
        "regularMarketPrice", "regularMarketVolume",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "fiftyDayAverage", "twoHundredDayAverage",
        "dividendYield", "beta",
        "totalRevenue", "revenueGrowth",
        "grossMargins", "operatingMargins", "profitMargins",
        "returnOnEquity", "debtToEquity",
        "freeCashflow", "earningsGrowth",
    ]
    result = {k: raw.get(k) for k in keys_of_interest}
    result["symbol"] = symbol.upper().replace(NSE_SUFFIX, "")
    result["fetch_time"] = datetime.now().isoformat()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. QUARTERLY EARNINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_earnings(symbol: str) -> dict:
    """
    Fetch quarterly earnings data — revenue and earnings actuals.

    Returns dict with:
        - quarterly_earnings: DataFrame (Revenue, Earnings per quarter)
        - earnings_dates: DataFrame of upcoming/past earnings dates
        - analyst_estimates: revenue/earnings estimates if available
    """
    ticker_str = _nse_ticker(symbol)

    def _fetch():
        t = yf.Ticker(ticker_str)
        return t

    try:
        t = _retry(_fetch)
    except Exception as exc:
        logger.error("Failed to create Ticker for %s: %s", ticker_str, exc)
        return {"quarterly_earnings": pd.DataFrame(), "earnings_dates": pd.DataFrame()}

    result = {}

    # Quarterly income statement (replaces deprecated quarterly_earnings)
    try:
        qis = t.quarterly_income_stmt
        if qis is not None and not qis.empty:
            result["quarterly_earnings"] = qis
        else:
            result["quarterly_earnings"] = pd.DataFrame()
    except Exception as exc:
        logger.warning("No quarterly income stmt for %s: %s", ticker_str, exc)
        result["quarterly_earnings"] = pd.DataFrame()

    # Earnings dates (upcoming and recent)
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            result["earnings_dates"] = ed
        else:
            result["earnings_dates"] = pd.DataFrame()
    except Exception as exc:
        logger.warning("No earnings dates for %s: %s", ticker_str, exc)
        result["earnings_dates"] = pd.DataFrame()

    # Compute beat/miss if we have estimate vs actual
    if not result["earnings_dates"].empty:
        df_ed = result["earnings_dates"].copy()
        if "EPS Estimate" in df_ed.columns and "Reported EPS" in df_ed.columns:
            df_ed["beat"] = df_ed["Reported EPS"] > df_ed["EPS Estimate"]
            df_ed["beat_pct"] = np.where(
                df_ed["EPS Estimate"] != 0,
                ((df_ed["Reported EPS"] - df_ed["EPS Estimate"]) / df_ed["EPS Estimate"].abs()) * 100,
                0.0,
            )
            result["earnings_dates"] = df_ed
            # Latest quarter beat/miss summary
            latest = df_ed.dropna(subset=["Reported EPS"]).head(1)
            if not latest.empty:
                row = latest.iloc[0]
                result["latest_beat"] = bool(row.get("beat", False))
                result["latest_beat_pct"] = float(row.get("beat_pct", 0.0))
                logger.info(
                    "%s latest earnings: %s (%.1f%%)",
                    ticker_str,
                    "BEAT" if result["latest_beat"] else "MISS",
                    result["latest_beat_pct"],
                )

    result["symbol"] = symbol.upper().replace(NSE_SUFFIX, "")
    result["fetch_time"] = datetime.now().isoformat()
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. NSE SESSION — Cookie-based session for NSE website APIs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NSESession:
    """
    Manages a requests.Session with the NSE India website.
    NSE requires a valid cookie obtained by first visiting the homepage.
    This class handles cookie acquisition and session reuse.
    """

    BASE_URL = "https://www.nseindia.com"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(NSE_HEADERS)
        self._cookie_time: Optional[datetime] = None
        self._cookie_ttl = timedelta(minutes=4)  # NSE cookies expire ~5 min

    def _refresh_cookies(self) -> None:
        """Hit NSE homepage to get fresh session cookies."""
        try:
            resp = self._session.get(
                self.BASE_URL,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            self._cookie_time = datetime.now()
            logger.debug("NSE cookies refreshed")
        except Exception as exc:
            logger.error("Failed to refresh NSE cookies: %s", exc)
            raise

    def _ensure_cookies(self) -> None:
        """Refresh cookies if expired or missing."""
        if (
            self._cookie_time is None
            or datetime.now() - self._cookie_time > self._cookie_ttl
        ):
            self._refresh_cookies()

    def get(self, url: str) -> dict:
        """GET request to NSE API endpoint with cookie management."""
        self._ensure_cookies()

        def _do_get():
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        return _retry(_do_get)


# Singleton session instance
_nse_session: Optional[NSESession] = None


def _get_nse_session() -> NSESession:
    """Get or create the shared NSE session."""
    global _nse_session
    if _nse_session is None:
        _nse_session = NSESession()
    return _nse_session


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BULK DEALS & BLOCK DEALS (NSE public API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_bulk_deals() -> pd.DataFrame:
    """
    Fetch today's bulk deals from NSE.

    Bulk deals: Trades where total quantity > 0.5% of listed shares.
    These are PUBLIC signals — large institutional activity.

    Returns:
        DataFrame with columns: symbol, client_name, deal_type (Buy/Sell),
        quantity, price, trade_date
    """
    url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.error("Failed to fetch bulk deals from NSE: %s", exc)
        return pd.DataFrame(columns=[
            "symbol", "client_name", "deal_type", "quantity", "price", "trade_date"
        ])

    # NSE returns {"BULK": [...], "BLOCK": [...]}
    bulk_records = data.get("BULK", [])
    if not bulk_records:
        logger.info("No bulk deals found today")
        return pd.DataFrame(columns=[
            "symbol", "client_name", "deal_type", "quantity", "price", "trade_date"
        ])

    rows = []
    for rec in bulk_records:
        rows.append({
            "symbol": rec.get("symbol", "").strip(),
            "client_name": rec.get("clientName", "").strip(),
            "deal_type": rec.get("buySell", "").strip(),
            "quantity": _safe_float(rec.get("quantity", 0)),
            "price": _safe_float(rec.get("wAvgPrice", 0)),
            "trade_date": rec.get("dealDate", ""),
        })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d bulk deals", len(df))
    return df


def fetch_block_deals() -> pd.DataFrame:
    """
    Fetch today's block deals from NSE.

    Block deals: Trades of 5 lakh+ shares OR ₹10 crore+ value,
    executed in a single transaction on the block deal window.

    Returns:
        DataFrame with same schema as bulk deals.
    """
    url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.error("Failed to fetch block deals from NSE: %s", exc)
        return pd.DataFrame(columns=[
            "symbol", "client_name", "deal_type", "quantity", "price", "trade_date"
        ])

    block_records = data.get("BLOCK", [])
    if not block_records:
        logger.info("No block deals found today")
        return pd.DataFrame(columns=[
            "symbol", "client_name", "deal_type", "quantity", "price", "trade_date"
        ])

    rows = []
    for rec in block_records:
        rows.append({
            "symbol": rec.get("symbol", "").strip(),
            "client_name": rec.get("clientName", "").strip(),
            "deal_type": rec.get("buySell", "").strip(),
            "quantity": _safe_float(rec.get("quantity", 0)),
            "price": _safe_float(rec.get("wAvgPrice", 0)),
            "trade_date": rec.get("dealDate", ""),
        })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d block deals", len(df))
    return df


def fetch_all_deals() -> pd.DataFrame:
    """
    Fetch BOTH bulk and block deals in one call, tagged by deal_class.
    More efficient — single API hit covers both.

    Returns:
        DataFrame with extra column 'deal_class' = 'BULK' | 'BLOCK'
    """
    url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.error("Failed to fetch deals from NSE: %s", exc)
        return pd.DataFrame()

    rows = []
    for deal_class in ("BULK", "BLOCK"):
        for rec in data.get(deal_class, []):
            rows.append({
                "symbol": rec.get("symbol", "").strip(),
                "client_name": rec.get("clientName", "").strip(),
                "deal_type": rec.get("buySell", "").strip(),
                "quantity": _safe_float(rec.get("quantity", 0)),
                "price": _safe_float(rec.get("wAvgPrice", 0)),
                "trade_date": rec.get("dealDate", ""),
                "deal_class": deal_class,
            })

    df = pd.DataFrame(rows)
    logger.info("Fetched %d total deals (bulk + block)", len(df))
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. LIVE QUOTE (NSE API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_live_quote(symbol: str) -> dict:
    """
    Fetch real-time quote for a single NSE stock.

    Returns dict with: ltp, change, pChange, open, high, low,
    previousClose, totalTradedVolume, totalTradedValue, etc.
    """
    clean = symbol.strip().upper().replace(NSE_SUFFIX, "")
    url = f"https://www.nseindia.com/api/quote-equity?symbol={clean}"

    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.error("Failed to fetch live quote for %s: %s", clean, exc)
        return {}

    price_info = data.get("priceInfo", {})
    info = data.get("info", {})

    result = {
        "symbol": clean,
        "company_name": info.get("companyName", ""),
        "industry": info.get("industry", ""),
        "ltp": _safe_float(price_info.get("lastPrice")),
        "change": _safe_float(price_info.get("change")),
        "pchange": _safe_float(price_info.get("pChange")),
        "open": _safe_float(price_info.get("open")),
        "high": _safe_float(price_info.get("intraDayHighLow", {}).get("max")),
        "low": _safe_float(price_info.get("intraDayHighLow", {}).get("min")),
        "prev_close": _safe_float(price_info.get("previousClose")),
        "volume": _safe_float(data.get("securityWiseDP", {}).get("quantityTraded")),
        "52w_high": _safe_float(price_info.get("weekHighLow", {}).get("max")),
        "52w_low": _safe_float(price_info.get("weekHighLow", {}).get("min")),
        "upper_band": _safe_float(price_info.get("upperCP")),
        "lower_band": _safe_float(price_info.get("lowerCP")),
        "fetch_time": datetime.now().isoformat(),
    }
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. INDEX CONSTITUENTS (NSE API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_index_constituents(index_name: str = "NIFTY 50") -> list[str]:
    """
    Fetch constituent symbols of an NSE index.

    Args:
        index_name: "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100",
                    "NIFTY 200", "NIFTY 500", etc.

    Returns:
        List of ticker symbols (without .NS suffix).
        Falls back to hardcoded NIFTY50 on failure.
    """
    encoded = index_name.replace(" ", "%20")
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}"

    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.warning(
            "Failed to fetch %s constituents: %s — using fallback list",
            index_name, exc,
        )
        return list(NIFTY50_TICKERS)

    stocks = data.get("data", [])
    symbols = []
    for s in stocks:
        sym = s.get("symbol", "").strip()
        if sym and sym != index_name:
            symbols.append(sym)

    if not symbols:
        logger.warning("Empty constituent list for %s — using fallback", index_name)
        return list(NIFTY50_TICKERS)

    logger.info("Fetched %d constituents for %s", len(symbols), index_name)
    return symbols


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. INSIDER TRADES (SEBI PIT Disclosures via NSE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_insider_trades(symbol: str) -> pd.DataFrame:
    """
    Fetch insider/promoter trading disclosures for a given stock
    from NSE's corporate governance endpoint.

    These map to SEBI PIT (Prohibition of Insider Trading) disclosures.
    Promoter BUYING is the strongest signal in our convergence model.

    Returns:
        DataFrame with columns: acquirer_name, category (Promoter/Director/etc),
        securities_type, transaction_type (Buy/Sell/Pledge), quantity,
        price, date_of_allotment
    """
    clean = symbol.strip().upper().replace(NSE_SUFFIX, "")
    url = (
        f"https://www.nseindia.com/api/corporates-pit?"
        f"index=equities&from_date=01-01-2024&to_date=20-03-2026&symbol={clean}"
    )

    try:
        session = _get_nse_session()
        data = session.get(url)
    except Exception as exc:
        logger.error("Failed to fetch insider trades for %s: %s", clean, exc)
        return pd.DataFrame()

    records = data if isinstance(data, list) else data.get("data", [])
    if not records:
        logger.info("No insider trades found for %s", clean)
        return pd.DataFrame()

    rows = []
    for rec in records:
        rows.append({
            "symbol": clean,
            "acquirer_name": rec.get("acqName", "").strip(),
            "category": rec.get("personCategory", "").strip(),
            "securities_type": rec.get("secType", "").strip(),
            "transaction_type": rec.get("tdpTransactionType", "").strip(),
            "quantity": _safe_float(rec.get("secAcq", 0)),
            "value_lakhs": _safe_float(rec.get("secVal", 0)),
            "date_of_trade": rec.get("acqfromDt", ""),
            "date_of_filing": rec.get("intimDt", ""),
            "pre_holding_pct": _safe_float(rec.get("befAcqSharesPerc", 0)),
            "post_holding_pct": _safe_float(rec.get("aftAcqSharesPerc", 0)),
        })

    df = pd.DataFrame(rows)
    # Filter to recent transactions (last 90 days)
    try:
        df["date_of_trade"] = pd.to_datetime(df["date_of_trade"], dayfirst=True, errors="coerce")
        cutoff = datetime.now() - timedelta(days=90)
        df = df[df["date_of_trade"] >= cutoff].sort_values("date_of_trade", ascending=False)
    except Exception:
        pass  # If date parsing fails, return all records

    logger.info("Fetched %d insider trade records for %s", len(df), clean)
    return df


def detect_promoter_buying(symbol: str) -> dict:
    """
    High-level signal: check if promoters have been NET BUYING recently.

    Returns:
        {
            "symbol": str,
            "promoter_buying": bool,
            "net_buy_quantity": float,
            "net_buy_value_lakhs": float,
            "recent_trades": list[dict],
            "signal_strength": float  # 0.0 to 1.0
        }
    """
    df = fetch_insider_trades(symbol)
    result = {
        "symbol": symbol.upper().replace(NSE_SUFFIX, ""),
        "promoter_buying": False,
        "net_buy_quantity": 0.0,
        "net_buy_value_lakhs": 0.0,
        "recent_trades": [],
        "signal_strength": 0.0,
    }

    if df.empty:
        return result

    # Filter to promoter/promoter group only
    promoter_mask = df["category"].str.contains(
        "Promoter|Promoter Group", case=False, na=False
    )
    promoter_df = df[promoter_mask].copy()

    if promoter_df.empty:
        return result

    # Compute net buying
    buys = promoter_df[
        promoter_df["transaction_type"].str.contains("Buy|Acquisition", case=False, na=False)
    ]
    sells = promoter_df[
        promoter_df["transaction_type"].str.contains("Sell|Disposal", case=False, na=False)
    ]

    buy_qty = buys["quantity"].sum()
    sell_qty = sells["quantity"].sum()
    buy_val = buys["value_lakhs"].sum()
    sell_val = sells["value_lakhs"].sum()

    net_qty = buy_qty - sell_qty
    net_val = buy_val - sell_val

    result["net_buy_quantity"] = float(net_qty)
    result["net_buy_value_lakhs"] = float(net_val)
    result["promoter_buying"] = net_qty > 0

    # Signal strength based on value (normalized to 0-1)
    # ₹10 crore+ = max signal, proportional below
    if net_val > 0:
        result["signal_strength"] = min(float(net_val) / 1000.0, 1.0)  # 1000 lakhs = 10 crore

    # Recent trades for citation
    result["recent_trades"] = promoter_df.head(5).to_dict(orient="records")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. UNIVERSE SCANNER — Batch fetch for the full stock universe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scan_universe(
    symbols: Optional[list[str]] = None,
    index_name: str = "NIFTY 500",
    delay_between: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """
    Batch-fetch OHLCV data for every stock in the universe.

    Args:
        symbols:       Explicit list of tickers. If None, fetches index constituents.
        index_name:    Index to use if symbols is None.
        delay_between: Seconds between requests (rate limiting).

    Returns:
        Dict mapping symbol → OHLCV DataFrame.
        Failed fetches are logged and omitted from results.
    """
    if symbols is None:
        symbols = fetch_index_constituents(index_name)

    logger.info("Scanning universe: %d stocks", len(symbols))
    results = {}
    failed = []

    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_ohlcv(sym)
            if not df.empty:
                results[sym] = df
            else:
                failed.append(sym)
        except Exception as exc:
            logger.warning("Skipping %s: %s", sym, exc)
            failed.append(sym)

        if i % 50 == 0:
            logger.info("Progress: %d/%d stocks fetched", i, len(symbols))

        time.sleep(delay_between)

    logger.info(
        "Universe scan complete: %d succeeded, %d failed",
        len(results), len(failed),
    )
    if failed:
        logger.warning("Failed tickers: %s", ", ".join(failed[:20]))

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_float(val) -> float:
    """Convert any value to float safely, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        # Handle strings with commas (Indian number format)
        if isinstance(val, str):
            val = val.replace(",", "").replace(" ", "").strip()
            if val in ("", "-", "NA", "N/A"):
                return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QUICK SMOKE TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("NSE FETCHER — Smoke Test")
    print("=" * 60)

    # 1. OHLCV
    print("\n[1] Fetching OHLCV for RELIANCE.NS (2y)...")
    df = fetch_ohlcv("RELIANCE")
    print(f"    → {len(df)} candles")
    if not df.empty:
        print(f"    → Latest: {df.index[-1].strftime('%Y-%m-%d')} "
              f"Close=₹{df['Close'].iloc[-1]:.2f}")

    # 2. Company Info
    print("\n[2] Fetching company info for RELIANCE...")
    info = fetch_company_info("RELIANCE")
    print(f"    → Name: {info.get('longName', 'N/A')}")
    print(f"    → Sector: {info.get('sector', 'N/A')}")
    print(f"    → Market Cap: ₹{info.get('marketCap', 0):,.0f}")

    # 3. Earnings
    print("\n[3] Fetching earnings for RELIANCE...")
    earnings = fetch_earnings("RELIANCE")
    if "latest_beat" in earnings:
        print(f"    → Latest quarter: {'BEAT' if earnings['latest_beat'] else 'MISS'} "
              f"({earnings['latest_beat_pct']:.1f}%)")
    else:
        print("    → No beat/miss data available")

    # 4. Insider Trades
    print("\n[4] Checking promoter buying for TATAMOTORS...")
    insider = detect_promoter_buying("TATAMOTORS")
    print(f"    → Promoter buying: {insider['promoter_buying']}")
    print(f"    → Net buy qty: {insider['net_buy_quantity']:,.0f}")
    print(f"    → Signal strength: {insider['signal_strength']:.2f}")

    # 5. Deals
    print("\n[5] Fetching today's bulk + block deals...")
    deals = fetch_all_deals()
    print(f"    → {len(deals)} deals found")
    if not deals.empty:
        print(deals.head(3).to_string(index=False))

    # 6. Index constituents
    print("\n[6] Fetching NIFTY 50 constituents...")
    constituents = fetch_index_constituents("NIFTY 50")
    print(f"    → {len(constituents)} stocks: {', '.join(constituents[:5])}...")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)
