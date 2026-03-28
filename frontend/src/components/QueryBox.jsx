import { useRef, useState } from "react";

export default function QueryBox({ onSubmitQuery, onSubmitVoice, busy }) {
  const [query, setQuery] = useState("");
  const fileInputRef = useRef(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim() || busy) {
      return;
    }
    await onSubmitQuery(query.trim());
  };

  const handleVoiceClick = () => {
    if (busy) {
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file || busy) {
      return;
    }
    await onSubmitVoice(file);
    event.target.value = "";
  };

  return (
    <section className="rounded-2xl border border-[#1a2640] bg-[#0f172e] p-4">
      <h2 className="mb-3 text-lg font-semibold text-[#e7eeff]">Ask Financial Intel</h2>
      <form className="flex flex-col gap-3 md:flex-row" onSubmit={handleSubmit}>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask: show me insider buying on INFY"
          className="flex-1 rounded-xl border border-[#26406e] bg-[#0a0e1a] px-4 py-3 text-sm text-[#e6ecff] outline-none placeholder:text-[#6f82ad] focus:border-[#00d4ff]"
        />
        <button
          type="submit"
          className="rounded-xl bg-[#00d4ff] px-5 py-3 text-sm font-semibold text-[#04131f] transition hover:bg-[#53e4ff]"
          disabled={busy}
        >
          {busy ? "Running..." : "Run Query"}
        </button>
        <button
          type="button"
          onClick={handleVoiceClick}
          className="rounded-xl border border-[#ffd700] px-5 py-3 text-sm font-semibold text-[#ffd700] transition hover:bg-[#ffd7001c]"
          disabled={busy}
        >
          Mic Input
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </form>
    </section>
  );
}
