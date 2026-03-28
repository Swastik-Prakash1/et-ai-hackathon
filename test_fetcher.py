"""Quick verify radar_agent output."""
import sys
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.WARNING)
from agents.radar_agent import scan_stock

for ticker in ["RELIANCE", "SBIN"]:
    p = scan_stock(ticker)
    print(f"\n{ticker}: {p.total_signal_count} signals ({p.bullish_signals}B {p.bearish_signals}R) composite={p.composite_score:.3f}")
    print(f"  insider_buying={p.has_insider_buying} earnings_beat={p.has_earnings_beat} deals={p.has_deal_activity}")
    for s in p.signals:
        d = "B" if s.direction == "bullish" else "R"
        print(f"  [{d}] {s.signal_type}: {s.headline}")
print("\nPASS")
