import { useEffect, useMemo, useState } from "react";

import ChartViewer from "./components/ChartViewer";
import LiveAlerts from "./components/LiveAlerts";
import QueryBox from "./components/QueryBox";
import RadarFeed from "./components/RadarFeed";
import SignalCard from "./components/SignalCard";
import {
  fetchChart,
  fetchRadar,
  fetchSignals,
  fetchTopSignals,
  queryIntelligence,
  queryVoice,
} from "./services/api";

const PORTFOLIO_STORAGE_KEY = "financial_intel_portfolio";

const DEMO_SIGNALS = [
  {
    ticker: "RELIANCE",
    convergence_score: 0.84,
    convergence_label: "STRONG BUY SIGNAL",
    win_rate: 0.71,
    action: "BUY",
  },
  {
    ticker: "TATAMOTORS",
    convergence_score: 0.78,
    convergence_label: "STRONG BUY SIGNAL",
    win_rate: 0.68,
    action: "BUY",
  },
  {
    ticker: "HDFCBANK",
    convergence_score: 0.63,
    convergence_label: "WATCH",
    win_rate: 0.59,
    action: "WATCH",
  },
];

const DEMO_RADAR = [
  { ticker: "RELIANCE", composite_score: 0.81, has_insider_buying: true, has_earnings_beat: true },
  { ticker: "TATAMOTORS", composite_score: 0.74, has_insider_buying: true, has_earnings_beat: false },
  { ticker: "HDFCBANK", composite_score: 0.66, has_insider_buying: false, has_earnings_beat: true },
];

export default function Dashboard() {
  const [signals, setSignals] = useState(DEMO_SIGNALS);
  const [portfolioInput, setPortfolioInput] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return window.localStorage.getItem(PORTFOLIO_STORAGE_KEY) || "";
  });
  const [radarOpportunities, setRadarOpportunities] = useState(DEMO_RADAR);
  const [radarStatusMessage, setRadarStatusMessage] = useState("");
  const [selectedSignal, setSelectedSignal] = useState(DEMO_SIGNALS[0]);
  const [chartData, setChartData] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [liveLoading, setLiveLoading] = useState(false);
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryOutput, setQueryOutput] = useState(null);
  const [marketStatus, setMarketStatus] = useState(null);
  const [error, setError] = useState("");

  const selectedTicker = selectedSignal?.ticker || "";

  const normalizeTicker = (rawTicker) => {
    return String(rawTicker || "")
      .trim()
      .toUpperCase()
      .replace(/\.NS$/i, "");
  };

  const parsePortfolioInput = (rawText) => {
    const parsed = {};
    const chunks = String(rawText || "")
      .split(/[;,\n]+/)
      .map((part) => part.trim())
      .filter(Boolean);

    chunks.forEach((chunk) => {
      const [rawTicker, rawQty] = chunk.split(":");
      const ticker = normalizeTicker(rawTicker);
      if (!ticker) {
        return;
      }

      const qty = Number(rawQty || 0);
      if (Number.isFinite(qty) && qty > 0) {
        parsed[ticker] = qty;
      }
    });

    return parsed;
  };

  const portfolioMap = useMemo(() => parsePortfolioInput(portfolioInput), [portfolioInput]);

  const handlePortfolioChange = (event) => {
    const value = event.target.value;
    setPortfolioInput(value);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(PORTFOLIO_STORAGE_KEY, value);
    }
  };

  const withTimeout = async (promise, timeoutMs, timeoutMessage) => {
    let timerId;
    try {
      const timeoutPromise = new Promise((_, reject) => {
        timerId = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      });
      return await Promise.race([promise, timeoutPromise]);
    } finally {
      clearTimeout(timerId);
    }
  };

  const buildEarningsOnlyRadar = (items) => {
    return items.map((item) => {
      const radarData = item?.radar_data || {};
      const radarSignals = Array.isArray(radarData?.signals) ? radarData.signals : [];
      const earningsSignal = radarSignals.find((sig) => sig?.signal_category === "earnings");

      let earningsLabel = "No earnings signal";
      if (earningsSignal?.signal_type === "earnings_beat") {
        earningsLabel = "Beat";
      } else if (earningsSignal?.signal_type === "earnings_miss") {
        earningsLabel = "Miss";
      } else if (radarData?.has_earnings_beat) {
        earningsLabel = "Beat";
      }

      return {
        ticker: item?.ticker || item?.symbol || "UNKNOWN",
        symbol: item?.ticker || item?.symbol || "UNKNOWN",
        composite_score: Number(item?.convergence_score || radarData?.composite_score || 0),
        has_insider_buying: false,
        has_earnings_beat: earningsLabel === "Beat",
        earnings_signal: earningsLabel,
      };
    });
  };

  const handleLoadLiveSignals = async () => {
    setLiveLoading(true);
    setError("");
    setRadarStatusMessage("");
    try {
      const topResp = await withTimeout(
        fetchTopSignals(3, 3),
        120000,
        "LIVE SIGNALS timed out after 120 seconds."
      );
      let baseItems = topResp.items || [];

      const isProcessing = (items) =>
        items.length > 0 && items.every((item) => String(item?.convergence_label || "") === "PROCESSING");

      if (isProcessing(baseItems)) {
        const pollStart = Date.now();
        while (Date.now() - pollStart < 120000) {
          await new Promise((resolve) => setTimeout(resolve, 5000));
          const polled = await fetchTopSignals(3, 3);
          const polledItems = polled.items || [];
          if (!isProcessing(polledItems) && polledItems.length > 0) {
            baseItems = polledItems;
            break;
          }
        }
      }

      const enriched = await Promise.all(
        baseItems.map(async (item) => {
          try {
            const full = await withTimeout(
              fetchSignals({
                ticker: item.ticker,
                document_type: "earnings_call",
                use_vlm: false,
                chart_days: 120,
                holding_days: 20,
              }),
              120000,
              `Signal build timed out for ${item.ticker}`
            );

            const patterns = full?.chart_data?.patterns || [];
            const bestWinRate = patterns.reduce((max, p) => {
              const current = Number(p.win_rate || 0);
              return current > max ? current : max;
            }, 0);

            return {
              ...item,
              ...full,
              win_rate: bestWinRate,
              action: full?.reasoning_data?.action || "WATCH",
            };
          } catch {
            return {
              ...item,
              win_rate: 0,
              action: item.convergence_label?.includes("STRONG BUY") ? "BUY" : "AVOID",
            };
          }
        })
      );

      let radarItems = [];
      try {
        const radarResp = await withTimeout(
          fetchRadar(3),
          30000,
          "Radar fetch timed out after 30 seconds."
        );
        radarItems = radarResp.opportunities || [];
      } catch {
        radarItems = buildEarningsOnlyRadar(enriched);
        setRadarStatusMessage("Market closed — showing earnings signals only");
      }

      if (radarItems.length === 0) {
        radarItems = buildEarningsOnlyRadar(enriched);
      }

      setSignals(enriched.length > 0 ? enriched : DEMO_SIGNALS);
      setRadarOpportunities(radarItems);
      setSelectedSignal((enriched.length > 0 ? enriched : DEMO_SIGNALS)[0]);
    } catch (loadError) {
      setError(loadError.message || "Failed to load live dashboard data.");
    } finally {
      setLiveLoading(false);
    }
  };

  useEffect(() => {
    fetch("http://localhost:8000/api/market-status")
      .then((r) => r.json())
      .then((data) => setMarketStatus(data))
      .catch(() => { });
  }, []);

  useEffect(() => {
    let active = true;

    async function loadChart() {
      if (!selectedTicker) {
        setChartData(null);
        return;
      }

      setChartLoading(true);
      setError("");
      try {
        const chartResp = await fetchChart(selectedTicker, true);
        if (!active) {
          return;
        }
        setChartData(chartResp);
      } catch (chartError) {
        if (!active) {
          return;
        }
        setError(chartError.message || "Failed to load chart data.");
      } finally {
        if (active) {
          setChartLoading(false);
        }
      }
    }

    loadChart();

    return () => {
      active = false;
    };
  }, [selectedTicker]);

  const handleSubmitQuery = async (queryText) => {
    setQueryBusy(true);
    setError("");
    try {
      const response = await queryIntelligence({
        query: queryText,
        ticker: selectedTicker || undefined,
        top_k: 5,
        portfolio: Object.keys(portfolioMap).length > 0 ? portfolioMap : undefined,
      });
      setQueryOutput(response);
    } catch (queryError) {
      setError(queryError.message || "Query failed.");
    } finally {
      setQueryBusy(false);
    }
  };

  const handleSubmitVoice = async (audioFile) => {
    setQueryBusy(true);
    setError("");
    try {
      const response = await queryVoice(audioFile, selectedTicker || "");
      setQueryOutput(response.query_response);
    } catch (voiceError) {
      setError(voiceError.message || "Voice query failed.");
    } finally {
      setQueryBusy(false);
    }
  };

  const activeReasoning = useMemo(() => {
    if (queryOutput?.reasoning_data) {
      return queryOutput.reasoning_data;
    }
    if (selectedSignal?.reasoning_data) {
      return selectedSignal.reasoning_data;
    }
    return null;
  }, [queryOutput, selectedSignal]);

  const reasoningPanel = useMemo(() => {
    if (!activeReasoning?.explanation) {
      return "Run a text or voice query to see reasoning output.";
    }
    return activeReasoning.explanation;
  }, [activeReasoning]);

  const actionPlan = useMemo(() => {
    return activeReasoning?.action_plan || null;
  }, [activeReasoning]);

  const portfolioSummary = useMemo(() => {
    return Array.isArray(activeReasoning?.portfolio_summary) ? activeReasoning.portfolio_summary : [];
  }, [activeReasoning]);

  const formatMoney = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return "--";
    }
    return `₹${num.toLocaleString("en-IN", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  };

  const riskColor = (rating) => {
    const r = String(rating).toLowerCase();
    if (r === "low") return { text: "#00ff88", bg: "#0d3b1f", border: "#00ff8840" };
    if (r === "medium") return { text: "#ffd700", bg: "#2e2a0d", border: "#ffd70040" };
    return { text: "#ff5c5c", bg: "#2c0e18", border: "#ff5c5c40" };
  };

  const rrColor = (ratio) => {
    if (ratio >= 2.0) return "#00ff88";
    if (ratio >= 1.0) return "#ffd700";
    return "#ff5c5c";
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_20%_0%,#112145_0%,#0a0e1a_50%,#070a13_100%)] px-4 py-6 text-[#e7eeff] md:px-6 lg:px-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-mono text-xs tracking-[0.3em] text-[#7f92ba]">ET AI HACKATHON 2026</p>
          <h1 className="text-2xl font-bold text-[#ffd700] md:text-3xl">FINANCIAL_INTEL Terminal</h1>
        </div>
        <button
          type="button"
          onClick={handleLoadLiveSignals}
          disabled={liveLoading}
          className="inline-flex items-center gap-2 rounded-xl border border-[#2a3f68] bg-[#0f172e] px-4 py-2 font-mono text-xs tracking-[0.2em] text-[#00d4ff] transition hover:border-[#00d4ff] disabled:opacity-60"
        >
          {liveLoading ? (
            <>
              <span className="h-3 w-3 animate-spin rounded-full border border-[#00d4ff] border-t-transparent" />
              LOADING LIVE...
            </>
          ) : (
            "LIVE SIGNALS"
          )}
        </button>
      </header>

      <section className="mb-4 rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4">
        <p className="mb-2 text-sm font-semibold text-[#e7eeff]">Your Portfolio</p>
        <input
          value={portfolioInput}
          onChange={handlePortfolioChange}
          placeholder="RELIANCE:100, TATAMOTORS:50, INFY:200"
          className="w-full rounded-xl border border-[#26406e] bg-[#0a0e1a] px-4 py-3 text-sm text-[#e6ecff] outline-none placeholder:text-[#6f82ad] focus:border-[#00d4ff]"
        />
        <p className="mt-2 text-xs text-[#8ea0c6]">
          Format: TICKER:QTY separated by commas. Saved automatically in your browser.
        </p>
      </section>

      {error ? (
        <div className="mb-4 rounded-xl border border-[#ff444480] bg-[#2c0e18] px-4 py-3 text-sm text-[#ffb6bf]">{error}</div>
      ) : null}

      {marketStatus && (
        <div
          style={{
            background: marketStatus.is_live ? "#0d2b1a" : "#1a1a0d",
            border: `1px solid ${marketStatus.is_live ? "#00ff88" : "#ffd700"}`,
            borderRadius: "6px",
            padding: "10px 16px",
            marginBottom: "12px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "10px",
            flexWrap: "wrap",
          }}
        >
          <span
            style={{
              color: marketStatus.is_live ? "#00ff88" : "#ffd700",
              fontFamily: "monospace",
              fontSize: "13px",
            }}
          >
            {marketStatus.is_live ? "● MARKET LIVE" : `○ ${String(marketStatus.status || "").toUpperCase()}`}
          </span>
          <span style={{ color: "#94a3b8", fontSize: "12px" }}>{marketStatus.message}</span>
          <span style={{ color: "#64748b", fontSize: "11px" }}>{marketStatus.current_time_ist}</span>
        </div>
      )}

      <LiveAlerts />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <aside className="xl:col-span-4">
          <div className="space-y-3">
            {signals.map((signal) => (
              <SignalCard
                key={signal.ticker}
                signal={signal}
                portfolioMap={portfolioMap}
                selected={selectedTicker === signal.ticker}
                onSelect={setSelectedSignal}
              />
            ))}
          </div>
        </aside>

        <main className="space-y-4 xl:col-span-8">
          <ChartViewer ticker={selectedTicker} chartData={chartData} loading={chartLoading} />
          <RadarFeed opportunities={radarOpportunities} statusMessage={radarStatusMessage} />
          <QueryBox onSubmitQuery={handleSubmitQuery} onSubmitVoice={handleSubmitVoice} busy={queryBusy} />
          <section className="rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4">
            <h2 className="mb-2 text-lg font-semibold text-[#e7eeff]">Reasoning Output</h2>
            <p className="text-sm leading-7 text-[#bbcaea]">{reasoningPanel}</p>

            {portfolioSummary.length > 0 ? (
              <div className="mt-3 rounded-xl border border-[#ffd70030] bg-[#1a1a0d] p-4">
                <h3 className="mb-2 text-sm font-semibold tracking-[0.12em] text-[#ffd700]">PORTFOLIO HEALTH REPORT</h3>
                <div className="space-y-1">
                  {portfolioSummary.map((line, idx) => (
                    <p key={`ps-${idx}`} className="font-mono text-xs text-[#c8d6f4]">{line}</p>
                  ))}
                </div>
              </div>
            ) : null}

            {actionPlan ? (() => {
              const rc = riskColor(actionPlan.risk_rating);
              const rr = Number(actionPlan.risk_reward_ratio || 0);
              return (
                <div className="mt-4 rounded-xl border border-[#27426f] bg-[#0a1226] p-5">
                  {/* Header row */}
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-sm font-semibold tracking-[0.15em] text-[#00d4ff]">ACTION PLAN</h3>
                    <div className="flex items-center gap-3">
                      {/* Risk rating badge */}
                      <span
                        className="rounded-md px-2.5 py-1 text-[10px] font-bold tracking-wider"
                        style={{ color: rc.text, background: rc.bg, border: `1px solid ${rc.border}` }}
                      >
                        {String(actionPlan.risk_rating || "--").toUpperCase()} RISK
                      </span>
                      {/* Risk/Reward ratio */}
                      <div className="text-right">
                        <p className="text-[10px] uppercase tracking-wider text-[#5a6d8e]">R:R Ratio</p>
                        <p className="font-mono text-lg font-bold" style={{ color: rrColor(rr) }}>
                          {rr > 0 ? `${rr.toFixed(1)}x` : "--"}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Price grid */}
                  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                    {/* Entry */}
                    <div className="rounded-lg border border-[#1e3050] bg-[#0d1b30] p-3">
                      <p className="mb-1 text-[10px] uppercase tracking-wider text-[#5a6d8e]">Entry Range</p>
                      <p className="font-mono text-sm font-semibold text-[#00ff88]">
                        {formatMoney(actionPlan.entry_price_range?.min)}
                      </p>
                      <p className="font-mono text-xs text-[#00ff8880]">
                        to {formatMoney(actionPlan.entry_price_range?.max)}
                      </p>
                    </div>
                    {/* Stop Loss */}
                    <div className="rounded-lg border border-[#1e3050] bg-[#0d1b30] p-3">
                      <p className="mb-1 text-[10px] uppercase tracking-wider text-[#5a6d8e]">Stop Loss</p>
                      <p className="font-mono text-sm font-semibold text-[#ff5c5c]">
                        {formatMoney(actionPlan.stop_loss?.price)}
                      </p>
                      <p className="text-[10px] text-[#6f82ad]">{actionPlan.stop_loss?.basis || ""}</p>
                    </div>
                    {/* Target */}
                    <div className="rounded-lg border border-[#1e3050] bg-[#0d1b30] p-3">
                      <p className="mb-1 text-[10px] uppercase tracking-wider text-[#5a6d8e]">Target Price</p>
                      <p className="font-mono text-sm font-semibold text-[#ffd700]">
                        {formatMoney(actionPlan.target_price?.price)}
                      </p>
                      <p className="text-[10px] text-[#6f82ad]">{actionPlan.target_price?.basis || ""}</p>
                    </div>
                    {/* Time Horizon */}
                    <div className="rounded-lg border border-[#1e3050] bg-[#0d1b30] p-3">
                      <p className="mb-1 text-[10px] uppercase tracking-wider text-[#5a6d8e]">Time Horizon</p>
                      <p className="font-mono text-sm font-semibold text-[#c8d6f4]">
                        {actionPlan.time_horizon_days || "--"} days
                      </p>
                    </div>
                  </div>

                  {/* Rationale */}
                  {Array.isArray(actionPlan.rationale) && actionPlan.rationale.length > 0 ? (
                    <div className="mt-3 rounded-lg border border-[#1a2640] bg-[#080f1e] p-3">
                      <p className="mb-1.5 text-[10px] uppercase tracking-wider text-[#5a6d8e]">Rationale</p>
                      <div className="space-y-1 text-xs text-[#94a3b8]">
                        {actionPlan.rationale.map((line, idx) => (
                          <p key={`${line}-${idx}`}>• {line}</p>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })() : null}
          </section>
        </main>
      </div>
    </div>
  );
}
