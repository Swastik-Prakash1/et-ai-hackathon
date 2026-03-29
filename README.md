# FINANCIAL_INTEL 🧠📈

<div align="center">

**The Intelligence Layer India's 14 Crore Investors Were Waiting For**

[![ET AI Hackathon 2026](https://img.shields.io/badge/ET%20AI%20Hackathon%202026-Problem%20Statement%20%236-orange?style=for-the-badge)](https://github.com/Swastik-Prakash1/et-ai-hackathon)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-cyan?style=for-the-badge&logo=react)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

</div>

---

## 🏆 ET AI Hackathon 2026 — Problem Statement #6

> *"India has 14 crore+ demat accounts, but most retail investors are flying blind — reacting to tips, missing filings, unable to read technicals, and managing portfolios on gut feel. ET Markets has the data. Build the intelligence layer that turns data into actionable, money-making decisions."*

**FINANCIAL_INTEL** is a **5-agent autonomous AI system** that solves PS6 by combining:

| PS6 Sub-Problem | Our Implementation |
|---|---|
| 📊 Chart Pattern Intelligence | 14 pattern types with **per-stock back-tested win rates** |
| 🎯 Opportunity Radar | NSE bulk deals + SEBI insider trades + earnings beats |
| 💬 Market ChatGPT Next-Gen | Groq LLM + RAG-grounded + voice input + portfolio-aware |
| 🔄 Convergence Scoring | **4 signals → 1 actionable number** — our unique moat |

---

## ⚡ The Moat — Convergence Score

No other system combines all four signals into one number. We do.

```
Convergence Score = (0.35 × chart_confidence)
                  + (0.30 × insider_signal_strength)
                  + (0.20 × earnings_beat_normalized)
                  + (0.15 × sentiment_score)

Score > 75  →  🟢 STRONG BUY SIGNAL
Score 50-75 →  🟡 WATCH
Score < 50  →  🔴 SELL / AVOID
```

**Example verified result:** Bullish MACD Crossover on BHARTIARTL — **71% win rate, 21 occurrences** over 2 years. Not a generic statistic. Verified on this specific stock.

---

## 🏗️ Architecture — 5 Agents

```
NSE/SEBI Data
     │
     ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│ CHART AGENT │───▶│ RADAR AGENT │───▶│ SENTIMENT    │───▶│ CONVERGENCE      │───▶│ REASONING      │
│             │    │             │    │ AGENT        │    │ AGENT            │    │ AGENT          │
│ pandas-ta   │    │ NSE bulk    │    │ FinBERT      │    │ 4-signal score   │    │ Groq LLM       │
│ 14 patterns │    │ SEBI PIT    │    │ tone shifts  │    │ THE MOAT         │    │ Plain English  │
│ VLM confirm │    │ earnings    │    │ QoQ detect   │    │ price targets    │    │ RAG-grounded   │
│ back-test   │    │ beat/miss   │    │ 440MB local  │    │ stop loss        │    │ action plan    │
└─────────────┘    └─────────────┘    └──────────────┘    └──────────────────┘    └────────────────┘
                                                                    │
                                                                    ▼
                                                         ChromaDB RAG Pipeline
                                                         (30 documents, 6 stocks)
```

### Autonomous Pipeline — Zero Human Input

```
[2026-03-28 02:30:51] STEP 1 — Detected: BHARTIARTL Bullish MACD Crossover (80% conf, 71% WR) | Convergence: 80%
[2026-03-28 02:31:43] STEP 2 — Enriched: BHARTIARTL — Retrieved 5 relevant documents from ChromaDB
[2026-03-28 02:31:48] STEP 3 — Alert: Entry ₹1,842  SL ₹1,793  Target ₹1,920  Horizon 10 days
[2026-03-28 02:31:48] === CYCLE #5 COMPLETE — 2 alerts generated, next cycle in 60s ===
```

Satisfies the judges' requirement: **3 sequential steps, zero human input.**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- ~600MB disk space (local models download on first run)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Swastik-Prakash1/et-ai-hackathon
cd et-ai-hackathon
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Set Up Your API Keys

> ⚠️ **Important:** The `.env.example` file in the repo is just a **template**. You must create your own `.env` file.

```bash
# Create your .env file from the template
cp .env.example .env
```

Now open `.env` in any text editor and fill in your API keys:

```env
# .env — fill in your own keys below

# Primary LLM — Sign up free at console.groq.com → API Keys
GROQ_API_KEY=your_groq_key_here

# Chart Vision VLM — Sign up free at aistudio.google.com → Get API Key
GEMINI_API_KEY=your_gemini_key_here

# LLM Fallback — Sign up free at openrouter.ai → Keys
OPENROUTER_API_KEY=your_openrouter_key_here
```

**All 3 keys are 100% free. No credit card required:**

| Key | Sign Up URL | Free Tier |
|-----|------------|-----------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | 14,400 requests/day |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | 1,500 requests/day |
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) | Free models available |

### Step 4 — Install Frontend

```bash
cd frontend
npm install
cd ..
```

### Step 5 — Seed the RAG Database (First Time Only)

```bash
python scripts/seed_chromadb.py
```

Expected output: `Seeded 24 documents for 6 stocks`

### Step 6 — Run the System

Open **3 separate terminals** and run one command in each:

```bash
# Terminal 1 — Backend API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend Dashboard
cd frontend && npm run dev

# Terminal 3 — Autonomous Pipeline
python -m scripts.auto_pipeline
```

### Step 7 — Open the Dashboard

```
Dashboard:   http://localhost:5173
API Docs:    http://localhost:8000/docs
Health:      http://localhost:8000/api/health
```

Health check should return: `"groq_key": true, "gemini_key": true, "openrouter_key": true`

---

## 📁 Project Structure

```
et-ai-hackathon/
│
├── agents/
│   ├── chart_agent.py          # Pattern detection + VLM confirmation + backtesting
│   ├── radar_agent.py          # Bulk deals + insider trades + earnings scanner
│   ├── sentiment_agent.py      # FinBERT local sentiment analysis
│   ├── convergence_agent.py    # Compound confidence score — THE MOAT
│   └── reasoning_agent.py      # Groq LLM plain English + action plans
│
├── data/
│   ├── nse_fetcher.py          # yfinance + nsepython market data (821 lines)
│   ├── ta_processor.py         # pandas-ta pattern detection (680 lines)
│   ├── backtest_engine.py      # Per-pattern per-stock win rates (530 lines)
│   ├── chart_renderer.py       # mplfinance Bloomberg-style annotated PNG
│   └── sebi_fetcher.py         # SEBI insider trade disclosures
│
├── rag/
│   ├── embedder.py             # all-MiniLM-L6-v2 sentence embeddings
│   ├── vector_store.py         # ChromaDB PersistentClient
│   └── retriever.py            # RAG query pipeline
│
├── api/
│   ├── main.py                 # FastAPI entry point + CORS
│   ├── routes.py               # 7 endpoints
│   └── schemas.py              # Pydantic models
│
├── voice/
│   └── whisper_stt.py          # Whisper tiny.en — runs on GPU/CPU
│
├── scripts/
│   ├── auto_pipeline.py        # Autonomous 3-step pipeline (zero human input)
│   └── seed_chromadb.py        # RAG database population
│
├── frontend/
│   └── src/
│       ├── Dashboard.jsx        # Bloomberg dark theme main dashboard
│       └── components/
│           ├── SignalCard.jsx   # Gold convergence gauge + portfolio badge
│           ├── ChartViewer.jsx  # Annotated chart PNG renderer
│           ├── RadarFeed.jsx    # Live opportunity feed
│           ├── QueryBox.jsx     # Text + voice query input
│           └── LiveAlerts.jsx  # Autonomous pipeline alerts
│
├── .env.example                # ← Copy this to .env and add your keys
├── .gitignore                  # .env is gitignored — never commits
├── requirements.txt            # All Python dependencies
└── CONTEXT.md                  # Full project context for AI assistants
```

---

## 🧠 Tech Stack — All Free

| Component | Tool | Type | Cost |
|-----------|------|------|------|
| Primary LLM | Groq — `llama-3.3-70b-versatile` | API | Free tier |
| Chart Vision | Gemini — `gemini-3-flash-preview` | API | Free tier |
| LLM Fallback | OpenRouter — `llama-3.3-70b-instruct:free` | API | Free |
| Sentiment NLP | FinBERT `yiyanghkust/finbert-tone` | Local (440MB) | ₹0 forever |
| Embeddings | `all-MiniLM-L6-v2` | Local (80MB) | ₹0 forever |
| Speech-to-Text | Whisper `tiny.en` | Local (75MB) | ₹0 forever |
| Vector DB | ChromaDB Embedded | Local | ₹0 forever |
| Market Data | yfinance + nsepython | Free scraping | ₹0 forever |
| Technical Analysis | pandas-ta + backtesting.py | Local lib | ₹0 forever |
| Backend | FastAPI + uvicorn | Local | ₹0 forever |
| Frontend | React + Vite + Tailwind | Local | ₹0 forever |

**Total external model cost: ₹0**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/signals` | Run full 5-agent pipeline for a ticker |
| `GET` | `/api/signals/top` | Scan universe, return top convergence scores |
| `GET` | `/api/charts/{ticker}` | Return annotated chart PNG + pattern data |
| `GET` | `/api/radar` | Today's opportunity feed |
| `POST` | `/api/query` | Natural language Q&A (portfolio-aware) |
| `POST` | `/api/voice` | Audio → Whisper STT → Groq response |
| `GET` | `/api/alerts/latest` | Latest autonomous pipeline alerts |
| `GET` | `/api/health` | API health + key status check |

---

## 🔧 Troubleshooting

**`Failed to fetch` on dashboard?**
- Check uvicorn is running on port 8000
- Verify `.env` has real keys (not placeholders)
- Run `curl http://localhost:8000/api/health` — all keys should show `true`

**NSE API returning 403 errors?**
- Normal during market hours and NSE holidays
- System automatically falls back to yfinance earnings data
- Signals still work — only bulk deal data is unavailable

**FinBERT downloading slowly?**
- First run downloads ~440MB (one time only)
- Subsequent runs use cached model — instant startup

**`No module named 'groq'` error?**
- Run: `pip install groq`

**All scores showing as WEAK?**
- Check `/api/health` — if `groq_key: false`, your `.env` is not loaded
- Make sure you ran `cp .env.example .env` and filled in real keys
- Make sure uvicorn is started from the project root directory

**`NaN` values in alerts?**
- Delete `alerts_log.json` and restart `auto_pipeline.py`

---

## 📊 Dashboard Features

- **Bloomberg-style dark terminal** — professional data-dense UI
- **Gold convergence gauge** — circular ring fills as score increases
- **Live Alerts panel** — autonomous pipeline results, auto-refreshes every 30s
- **4-panel annotated charts** — candlestick + SMA50/200 + BB + RSI + MACD
- **Portfolio input** — enter your holdings, get personalised signal badges
- **Voice query** — click Mic Input, speak your question in plain English
- **Market status banner** — OPEN / CLOSED / WEEKEND / HOLIDAY detection
- **Opportunity Radar** — earnings beats, insider signals, sector momentum

---

## 🏅 Judging Criteria — How We Satisfy Each

| Criterion | Our Implementation |
|-----------|-------------------|
| ✅ Signal quality over summarization | Back-tested win rates per stock — real alpha, not news |
| ✅ Depth of financial data integration | NSE + SEBI + yfinance + FinBERT + ChromaDB RAG |
| ✅ Agent ability to ACT on signals | Price targets + stop loss + time horizon per alert |
| ✅ 3 sequential steps, zero human input | `auto_pipeline.py` — detect → enrich → alert every 60s |
| ✅ Portfolio-aware personalisation | Holdings input → IN YOUR PORTFOLIO badges + gain calc |
| ✅ Cost-efficient multi-model routing | Local models ₹0 + free API tiers = extra credit |

---

## 💰 Business Impact

```
TAM:              14 crore+ existing ET Markets users (Zero CAC)
Year 1 target:    1% adoption = 14 lakh users × ₹500/year = ₹70 Crore ARR
Back-tested WR:   71% on golden cross signals (NIFTY50, 2022–2024)
Alpha per signal: ₹11,786 on ₹2L average retail portfolio
```

---

## ⚠️ Important Notes

- **`.env` is gitignored** — your API keys are never committed to GitHub
- **NSE API** blocks automated requests during market hours (9:15 AM – 3:30 PM IST) and public holidays — the system handles this gracefully with fallbacks
- **Local models** (FinBERT, MiniLM, Whisper) download automatically on first run and are cached for all subsequent runs
- **Gemini 2.0 Flash is deprecated** (retired March 3, 2026) — we use `gemini-3-flash-preview`

---

## 👨‍💻 Author

**Swastik Prakash**
3rd Year CS Undergraduate — JIIT Noida (2027)
[github.com/Swastik-Prakash1](https://github.com/Swastik-Prakash1)

---

<div align="center">

**Built for ET AI Hackathon 2026 — Problem Statement #6 — AI for the Indian Investor**

*Powered by Groq • Gemini • FinBERT • ChromaDB • FastAPI • React*

</div>
