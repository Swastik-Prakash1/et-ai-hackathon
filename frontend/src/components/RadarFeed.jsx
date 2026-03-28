export default function RadarFeed({ opportunities = [], statusMessage = "" }) {
  return (
    <section className="rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[#e7eeff]">Opportunity Radar</h2>
        <span className="font-mono text-xs tracking-[0.2em] text-[#7f92ba]">LIVE FEED</span>
      </div>

      {statusMessage ? (
        <p className="mb-3 rounded-xl border border-[#2f3f61] bg-[#0a0e1a] p-3 text-xs text-[#b9c8e8]">
          {statusMessage}
        </p>
      ) : null}

      <div className="max-h-[320px] space-y-2 overflow-y-auto pr-1">
        {opportunities.length === 0 ? (
          <p className="rounded-xl bg-[#0a0e1a] p-3 text-sm text-[#8ea0c6]">
            {statusMessage || "No radar opportunities found."}
          </p>
        ) : (
          opportunities.map((item, idx) => (
            <div key={`${item.ticker || item.symbol}-${idx}`} className="rounded-xl border border-[#1f2e50] bg-[#0a0e1a] p-3">
              <div className="flex items-center justify-between">
                <p className="font-mono text-sm text-[#00d4ff]">{item.ticker || item.symbol || "UNKNOWN"}</p>
                <p className="font-mono text-xs text-[#ffd700]">Score {Number(item.composite_score || 0).toFixed(2)}</p>
              </div>
              <p className="mt-2 text-xs text-[#b3c3e8]">
                Insider: {item.has_insider_buying ? "Yes" : "No"} | Earnings: {item.earnings_signal || (item.has_earnings_beat ? "Beat" : "No beat data")}
              </p>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
