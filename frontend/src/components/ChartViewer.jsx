export default function ChartViewer({ ticker, chartData, loading }) {
  return (
    <section className="rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[#e7eeff]">Chart Viewer</h2>
        <span className="font-mono text-xs tracking-[0.2em] text-[#7f92ba]">{ticker || "NO TICKER"}</span>
      </div>

      {loading ? (
        <div className="flex h-[380px] items-center justify-center rounded-xl bg-[#0a0e1a] text-[#8ea0c6]">
          Loading chart...
        </div>
      ) : chartData?.chart_image_base64 ? (
        <div className="space-y-3">
          <img
            src={`data:image/png;base64,${chartData.chart_image_base64}`}
            alt={`Chart for ${ticker}`}
            className="h-[380px] w-full rounded-xl border border-[#203157] object-contain bg-[#0a0e1a]"
          />
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="rounded-xl bg-[#0a0e1a] p-3">
              <p className="font-mono text-[10px] tracking-[0.2em] text-[#7f92ba]">CONFIDENCE</p>
              <p className="mt-1 font-mono text-[#ffd700]">{Number(chartData.chart_confidence || 0).toFixed(2)}</p>
            </div>
            <div className="rounded-xl bg-[#0a0e1a] p-3">
              <p className="font-mono text-[10px] tracking-[0.2em] text-[#7f92ba]">BIAS</p>
              <p className="mt-1 font-mono text-[#00d4ff]">{chartData.overall_bias || "neutral"}</p>
            </div>
            <div className="rounded-xl bg-[#0a0e1a] p-3">
              <p className="font-mono text-[10px] tracking-[0.2em] text-[#7f92ba]">VLM</p>
              <p className={`mt-1 font-mono ${chartData.vlm_confirmed ? "text-[#00ff88]" : "text-[#ff4444]"}`}>
                {chartData.vlm_confirmed ? "confirmed" : "not confirmed"}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex h-[380px] items-center justify-center rounded-xl bg-[#0a0e1a] text-[#8ea0c6]">
          Select a signal card to load chart.
        </div>
      )}
    </section>
  );
}
