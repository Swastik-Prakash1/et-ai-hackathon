function CircularGauge({ score }) {
  const safeScore = Math.max(0, Math.min(1, Number(score) || 0));
  const percent = Math.round(safeScore * 100);
  const radius = 44;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 0.5;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  return (
    <div className="relative h-28 w-28">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 88 88">
        <circle
          cx="44"
          cy="44"
          r={normalizedRadius}
          stroke="#1b253a"
          strokeWidth={stroke}
          fill="transparent"
        />
        <circle
          cx="44"
          cy="44"
          r={normalizedRadius}
          stroke="#ffd700"
          strokeWidth={stroke}
          strokeLinecap="round"
          fill="transparent"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: "stroke-dashoffset 600ms ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <p className="font-mono text-2xl font-bold text-[#ffd700]">{percent}</p>
        <p className="font-mono text-[10px] tracking-[0.2em] text-[#8ea0c6]">SCORE</p>
      </div>
    </div>
  );
}

export default function SignalCard({ signal, portfolioMap = {}, selected, onSelect }) {
  const normalizeTicker = (rawTicker) => {
    return String(rawTicker || "")
      .trim()
      .toUpperCase()
      .replace(/\.NS$/i, "");
  };

  const convergenceScore = Number(signal?.convergence_score || 0);
  const label = String(signal?.convergence_label || "WEAK");
  const isBuy = label.includes("STRONG BUY") || signal?.action === "BUY";
  const winRateRaw = Number(signal?.win_rate || 0);
  const winRatePercent = winRateRaw <= 1 ? Math.round(winRateRaw * 100) : Math.round(winRateRaw);
  const ticker = normalizeTicker(signal?.ticker || signal?.symbol);
  const quantity = Number(portfolioMap?.[ticker] || 0);
  const inPortfolio = Number.isFinite(quantity) && quantity > 0;

  // Calculate potential gain if target is available
  const actionPlan = signal?.reasoning_data?.action_plan || signal?.action_plan || null;
  const currentPrice = actionPlan?.entry_price_range
    ? (Number(actionPlan.entry_price_range.min || 0) + Number(actionPlan.entry_price_range.max || 0)) / 2
    : 0;
  const targetPrice = Number(actionPlan?.target_price?.price || 0);
  const potentialGainPerShare = targetPrice > 0 && currentPrice > 0 ? targetPrice - currentPrice : 0;
  const potentialGainTotal = inPortfolio ? potentialGainPerShare * quantity : 0;

  const formatINR = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num === 0) return null;
    const sign = num >= 0 ? "+" : "";
    return `${sign}₹${Math.abs(num).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
  };

  return (
    <button
      type="button"
      onClick={() => onSelect?.(signal)}
      className={`w-full rounded-2xl border p-4 text-left transition-all duration-300 ${selected
        ? "border-[#00d4ff] bg-[#101933] shadow-[0_0_32px_rgba(0,212,255,0.22)]"
        : "border-[#1a2640] bg-[#0f172e] hover:border-[#2d3f67]"
        }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-mono text-xs tracking-[0.3em] text-[#7f92ba]">TICKER</p>
          <h3 className="mt-1 text-xl font-semibold text-[#e7eeff]">{signal?.ticker}</h3>
          <span
            className={`mt-2 inline-block rounded-full px-3 py-1 text-xs font-semibold ${isBuy
              ? "bg-[#00ff8830] text-[#00ff88]"
              : "bg-[#ff444430] text-[#ff4444]"
              }`}
          >
            {isBuy ? "BUY" : "SELL / AVOID"}
          </span>
          {inPortfolio ? (
            <span className="ml-2 mt-2 inline-block rounded-full bg-[#ffd70025] px-3 py-1 text-xs font-bold text-[#ffd700]">
              IN YOUR PORTFOLIO
            </span>
          ) : null}
        </div>
        <CircularGauge score={convergenceScore} />
      </div>

      {/* Portfolio details */}
      {inPortfolio ? (
        <div className="mt-3 rounded-xl border border-[#ffd70030] bg-[#1a1a0d] p-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-[#ffd700]">
                You hold: <span className="font-mono font-semibold">{quantity} shares</span>
              </p>
              {potentialGainTotal !== 0 ? (
                <p className="mt-1 text-xs">
                  <span className="text-[#8ea0c6]">Potential gain if target hit: </span>
                  <span
                    className="font-mono font-semibold"
                    style={{ color: potentialGainTotal >= 0 ? "#00ff88" : "#ff5c5c" }}
                  >
                    {formatINR(potentialGainTotal)}
                  </span>
                </p>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl bg-[#0a0e1a] p-3">
          <p className="font-mono text-[10px] tracking-[0.2em] text-[#7f92ba]">CONVERGENCE</p>
          <p className="mt-1 font-mono text-lg text-[#ffd700]">{(convergenceScore * 100).toFixed(1)} / 100</p>
        </div>
        <div className="rounded-xl bg-[#0a0e1a] p-3">
          <p className="font-mono text-[10px] tracking-[0.2em] text-[#7f92ba]">WIN RATE</p>
          <p className="mt-1 font-mono text-lg text-[#00d4ff]">{winRatePercent}%</p>
        </div>
      </div>

      <p className="mt-3 text-xs uppercase tracking-[0.2em] text-[#93a8d4]">{label}</p>
    </button>
  );
}
