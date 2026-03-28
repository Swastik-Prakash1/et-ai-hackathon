import { useEffect, useState } from "react";
import { fetchLatestAlerts } from "../services/api";

const REFRESH_INTERVAL_MS = 30_000;

function actionColor(action) {
    const a = String(action).toUpperCase();
    if (a === "BUY" || a === "STRONG BUY") return "#00ff88";
    if (a === "WATCH") return "#ffd700";
    return "#ff5c5c";
}

function scoreBadge(score) {
    const pct = Math.round((score || 0) * 100);
    let bg = "#1a2640";
    if (pct >= 75) bg = "#0d3b1f";
    else if (pct >= 50) bg = "#2e2a0d";
    return { pct, bg };
}

export default function LiveAlerts() {
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [lastRefresh, setLastRefresh] = useState(null);

    const loadAlerts = async () => {
        try {
            const resp = await fetchLatestAlerts(10);
            setAlerts(resp.alerts || []);
            setLastRefresh(new Date().toLocaleTimeString());
        } catch {
            // Silently ignore — pipeline may not have produced alerts yet
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadAlerts();
        const timer = setInterval(loadAlerts, REFRESH_INTERVAL_MS);
        return () => clearInterval(timer);
    }, []);

    if (loading && alerts.length === 0) {
        return (
            <section
                id="live-alerts-section"
                className="mb-4 rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4"
            >
                <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#00ff88]" />
                    <h2 className="text-sm font-semibold tracking-[0.15em] text-[#00ff88]">LIVE ALERTS</h2>
                </div>
                <p className="mt-2 text-xs text-[#6f82ad]">Loading autonomous pipeline alerts...</p>
            </section>
        );
    }

    return (
        <section
            id="live-alerts-section"
            className="mb-4 rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4"
        >
            {/* Header */}
            <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#00ff88]" />
                    <h2 className="text-sm font-semibold tracking-[0.15em] text-[#00ff88]">LIVE ALERTS</h2>
                    <span className="rounded bg-[#0d3b1f] px-2 py-0.5 text-[10px] font-bold tracking-wider text-[#00ff88]">
                        AUTONOMOUS
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    {lastRefresh && (
                        <span className="text-[10px] text-[#4a5d82]">Updated {lastRefresh}</span>
                    )}
                    <button
                        type="button"
                        onClick={loadAlerts}
                        className="rounded border border-[#263654] px-2 py-0.5 text-[10px] text-[#7f92ba] transition hover:border-[#00d4ff] hover:text-[#00d4ff]"
                    >
                        REFRESH
                    </button>
                </div>
            </div>

            {alerts.length === 0 ? (
                <p className="text-xs text-[#6f82ad]">
                    No alerts yet — autonomous pipeline will generate alerts when convergence scores exceed 50%.
                </p>
            ) : (
                <div className="space-y-2">
                    {alerts.map((alert, idx) => {
                        const { pct, bg } = scoreBadge(alert.convergence_score);
                        const ac = actionColor(alert.action);
                        return (
                            <div
                                key={`${alert.ticker}-${alert.timestamp}-${idx}`}
                                className="rounded-xl border border-[#1e3050] p-3 transition hover:border-[#2a4a7a]"
                                style={{ background: bg }}
                            >
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                    {/* Ticker + Action */}
                                    <div className="flex items-center gap-2">
                                        <span className="font-mono text-sm font-bold text-[#e7eeff]">
                                            {alert.ticker}
                                        </span>
                                        <span
                                            className="rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wider"
                                            style={{ color: ac, border: `1px solid ${ac}40` }}
                                        >
                                            {String(alert.action).toUpperCase()}
                                        </span>
                                        <span className="font-mono text-xs text-[#00d4ff]">{pct}%</span>
                                    </div>

                                    {/* Price targets */}
                                    <div className="flex items-center gap-3 text-[11px]">
                                        {alert.entry_price != null && (
                                            <span className="text-[#8ea0c6]">
                                                Entry <span className="text-[#c8d6f4]">₹{Number(alert.entry_price).toLocaleString()}</span>
                                            </span>
                                        )}
                                        {alert.stop_loss != null && (
                                            <span className="text-[#8ea0c6]">
                                                SL <span className="text-[#ff5c5c]">₹{Number(alert.stop_loss).toLocaleString()}</span>
                                            </span>
                                        )}
                                        {alert.target_price != null && (
                                            <span className="text-[#8ea0c6]">
                                                Target <span className="text-[#00ff88]">₹{Number(alert.target_price).toLocaleString()}</span>
                                            </span>
                                        )}
                                        {alert.time_horizon != null && (
                                            <span className="text-[#8ea0c6]">
                                                {alert.time_horizon}d
                                            </span>
                                        )}
                                    </div>
                                </div>

                                {/* Evidence chain */}
                                <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px] text-[#5a6d8e]">
                                    <span>{alert.timestamp}</span>
                                    <span>•</span>
                                    <span>Cycle #{alert.cycle}</span>
                                    {alert.rag_context_count > 0 && (
                                        <>
                                            <span>•</span>
                                            <span>{alert.rag_context_count} RAG docs</span>
                                        </>
                                    )}
                                    {alert.patterns?.length > 0 && alert.patterns[0]?.name && (
                                        <>
                                            <span>•</span>
                                            <span>{alert.patterns[0].name}</span>
                                        </>
                                    )}
                                </div>

                                {/* Explanation preview */}
                                {alert.explanation && (
                                    <p className="mt-1.5 line-clamp-2 text-[11px] leading-4 text-[#94a3b8]">
                                        {alert.explanation}
                                    </p>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </section>
    );
}
