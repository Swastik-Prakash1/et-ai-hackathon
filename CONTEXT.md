# ET AI Hackathon 2026 — Full Project Context for GitHub Copilot
# Keep this file open in VS Code while building any file in this project.
# Copilot reads open tabs as context — this file tells it everything.

---

## WHAT THIS PROJECT IS

Name: FINANCIAL_INTEL — AI Signal Intelligence for Indian Retail Investors
Hackathon: ET AI Hackathon 2026 (Economic Times + Avataar.ai + Unstop)
Problem: PS6 — AI for the Indian Investor

India has 14 crore+ demat accounts. Most retail investors react to WhatsApp 
tips, miss corporate filings, and manage portfolios on gut feel. This system 
is the intelligence layer that turns raw NSE/SEBI data into actionable, 
back-tested, plain-English signals.

---

## THE UNIQUE DIFFERENTIATOR — CONVERGENCE SCORING

When a chart pattern fires on the SAME stock where:
  + Insider buying just happened (SEBI PIT disclosure) AND
  + Earnings beat estimates AND
  + Management sentiment is bullish (FinBERT)

→ System produces a COMPOUND CONFIDENCE SCORE combining all 4 signals.

Formula (NEVER change these weights):
convergence_score = (
    0.35 * chart_confidence +      # from chart_agent.py
    0.30 * insider_signal_strength + # from radar_agent.py
    0.20 * earnings_beat_normalized + # from radar_agent.py
    0.15 * sentiment_score_normalized # from sentiment_agent.py
)

Score > 0.75 → STRONG BUY SIGNAL
Score 0.50–0.75 → WATCH
Score < 0.50 → WEAK / IGNORE

---

## PROJECT FILE STRUCTURE

et-ai-hackathon/
├── data/
│   ├── nse_fetcher.py        ✅ BUILT — market data pipeline
│   ├── ta_processor.py       ✅ BUILT — technical pattern detection
│   ├── backtest_engine.py    ✅ BUILT — per-pattern win rates
│   ├── chart_renderer.py     ✅ BUILT — mplfinance annotated PNGs
│   └── sebi_fetcher.py       ❌ NOT BUILT YET
├── agents/
│   ├── chart_agent.py        ✅ BUILT — full chart pipeline
│   ├── radar_agent.py        ✅ BUILT — bulk/insider/earnings scanner
│   ├── sentiment_agent.py    🔄 BUILDING — FinBERT sentiment
│   ├── convergence_agent.py  ❌ NOT BUILT YET
│   └── reasoning_agent.py    ❌ NOT BUILT YET
├── rag/
│   ├── embedder.py           ❌ NOT BUILT YET
│   ├── vector_store.py       ❌ NOT BUILT YET
│   └── retriever.py          ❌ NOT BUILT YET
├── api/
│   ├── main.py               ❌ NOT BUILT YET
│   ├── routes.py             ❌ NOT BUILT YET
│   └── schemas.py            ❌ NOT BUILT YET
├── frontend/
│   └── src/
│       ├── App.jsx           ❌ NOT BUILT YET
│       ├── Dashboard.jsx     ❌ NOT BUILT YET
│       └── components/
│           ├── SignalCard.jsx
│           ├── ChartViewer.jsx
│           ├── RadarFeed.jsx
│           └── QueryBox.jsx
├── voice/
│   └── whisper_stt.py        ❌ NOT BUILT YET
├── scripts/
│   ├── seed_chromadb.py      ❌ NOT BUILT YET
│   └── demo_run.py           ❌ NOT BUILT YET
├── .env                      ✅ EXISTS — never commit
├── .env.example              ✅ EXISTS
├── requirements.txt          ✅ EXISTS
└── CONTEXT.md                ← THIS FILE

---

## ENVIRONMENT VARIABLES — ALWAYS LOAD THIS WAY

Every Python file that uses an API key MUST start with:
```python
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```

NEVER hardcode any key. NEVER use os.environ["KEY"] without a default.
Always use os.getenv("KEY") which returns None instead of crashing.

---

## API CLIENTS — EXACT SETUP CODE

### Groq (Primary LLM)
```python
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # EXACT string, do not change
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    max_tokens=1000
)
result = response.choices[0].message.content
```

### Gemini (VLM for chart vision)
```python
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")  # EXACT string

# For vision (chart image analysis):
img = PIL.Image.open("chart.png")
response = model.generate_content([img, "your prompt here"])
result = response.text

# Fallback if quota exceeded:
# Skip VLM, use pandas-ta results only
# Mark signal as "TA-only, no VLM confirmation"
```

### OpenRouter (LLM Fallback)
```python
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

or_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

response = or_client.chat.completions.create(
    model="meta-llama/llama-3.3-70b-instruct:free",  # EXACT string
    messages=[{"role": "user", "content": prompt}]
)
```

---

## LOCAL MODELS — EXACT SETUP CODE

### FinBERT (Financial Sentiment) — agents/sentiment_agent.py
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# Load once at module level (440MB, cached after first download)
finbert_model = BertForSequenceClassification.from_pretrained(
    'yiyanghkust/finbert-tone', 
    num_labels=3
)
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=finbert_model, 
    tokenizer=finbert_tokenizer,
    device=0 if torch.cuda.is_available() else -1  # auto GPU/CPU
)

# Usage:
results = sentiment_pipeline(["Revenue guidance raised for Q4"])
# Returns: [{'label': 'Positive', 'score': 0.97}]
# Labels: 'Positive', 'Negative', 'Neutral'
```

### Sentence Embeddings (RAG) — rag/embedder.py
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level (80MB)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Usage:
embedding = embedder.encode("Promoter bought 2L shares at ₹450")
# Returns: numpy array shape (384,)
```

### Whisper STT — voice/whisper_stt.py
```python
import whisper
import torch

# Load once (75MB, runs on RTX 3050)
whisper_model = whisper.load_model(
    "tiny.en",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Usage:
result = whisper_model.transcribe("voice_query.wav")
text = result["text"]
```

---

## DATA SOURCES — EXACT USAGE

### yfinance (OHLCV + Fundamentals)
```python
import yfinance as yf

# ALWAYS append .NS for NSE stocks
ticker = yf.Ticker("RELIANCE.NS")
df = ticker.history(period="2y")     # 2 years daily OHLCV
info = ticker.info                    # fundamentals dict
# Use quarterly_income_stmt NOT quarterly_earnings (deprecated in yfinance 1.2.0)
earnings = ticker.quarterly_income_stmt
```

### nsepython (NSE Live Data)
```python
from nsepython import nse_eq

bulk_deals = nse_eq("bulk_deals")    # today's bulk deals
block_deals = nse_eq("block_deals")  # today's block deals
quote = nse_eq("RELIANCE")           # live quote
```

### Technical Analysis (pandas-ta)
```python
import pandas_ta as ta

df['RSI'] = ta.rsi(df['Close'], length=14)
macd = ta.macd(df['Close'])
df['MACD'] = macd['MACD_12_26_9']
df['MACD_Signal'] = macd['MACDs_12_26_9']
df['MACD_Hist'] = macd['MACDh_12_26_9']
bb = ta.bbands(df['Close'], length=20)
df['BB_upper'] = bb['BBU_20_2.0']
df['BB_lower'] = bb['BBL_20_2.0']
df['SMA50'] = ta.sma(df['Close'], length=50)
df['SMA200'] = ta.sma(df['Close'], length=200)
df['Golden_Cross'] = (
    (df['SMA50'] > df['SMA200']) & 
    (df['SMA50'].shift(1) <= df['SMA200'].shift(1))
)
```

### ChromaDB (Vector Database)
```python
import chromadb

# Always use persistent client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("market_intelligence")

# Add documents
collection.add(
    documents=["Promoter acquired 2L shares at ₹450"],
    embeddings=[embedding.tolist()],
    ids=["filing_001"],
    metadatas=[{
        "stock": "RELIANCE", 
        "type": "insider_trade", 
        "date": "2026-03-15"
    }]
)

# Query
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)
```

---

## AGENT OUTPUT SCHEMAS — MUST MATCH EXACTLY

Every agent returns a dataclass or dict. These field names are 
FIXED — convergence_agent.py depends on them exactly as written.

### chart_agent.py output:
```python
{
    "ticker": "RELIANCE",
    "chart_confidence": 0.80,        # 0-1, feeds convergence formula
    "patterns": [
        {
            "name": "Bullish Engulfing",
            "type": "bullish",
            "confidence": 80,         # 0-100
            "win_rate": 0.53,         # 0-1 historical win rate
            "occurrences": 15,        # backtested sample size
            "explanation": "...",     # plain English
            "action": "buy"
        }
    ],
    "chart_path": "charts/RELIANCE_20260320.png",
    "overall_bias": "bullish",
    "vlm_confirmed": True             # False if Gemini quota exceeded
}
```

### radar_agent.py output:
```python
{
    "ticker": "RELIANCE",
    "composite_score": 0.72,          # 0-1
    "has_insider_buying": True,
    "insider_signal_strength": 0.85,  # 0-1, feeds convergence formula
    "has_earnings_beat": True,
    "earnings_beat_pct": 0.204,       # 0.204 = 20.4% beat
    "signals": [
        {
            "type": "insider_trade",
            "headline": "Promoter bought ₹2.3Cr worth of shares",
            "detail": "...",
            "strength": 0.85,
            "date": "2026-03-15"
        }
    ]
}
```

### sentiment_agent.py output:
```python
{
    "ticker": "RELIANCE",
    "sentiment_score": 0.72,          # -1 to +1, feeds convergence formula
    "sentiment_label": "Positive",    # Positive / Negative / Neutral
    "tone_shift": True,               # True if shifted from prev quarter
    "tone_shift_direction": "positive", # positive / negative
    "key_sentences": [
        {
            "text": "Revenue guidance raised for FY27",
            "label": "Positive",
            "score": 0.97
        }
    ],
    "document_type": "earnings_call"
}
```

### convergence_agent.py output:
```python
{
    "ticker": "RELIANCE",
    "convergence_score": 0.84,        # THE NUMBER — 0-1
    "convergence_label": "STRONG BUY SIGNAL",  # or WATCH or WEAK
    "signal_breakdown": {
        "chart_contribution": 0.28,   # 0.35 * chart_confidence
        "insider_contribution": 0.255, # 0.30 * insider_signal_strength
        "earnings_contribution": 0.16, # 0.20 * earnings_beat_normalized
        "sentiment_contribution": 0.108 # 0.15 * sentiment_normalized
    },
    "signals_present": ["chart_pattern", "insider_buying", 
                         "earnings_beat", "positive_sentiment"],
    "chart_data": { ...chart_agent output... },
    "radar_data": { ...radar_agent output... },
    "sentiment_data": { ...sentiment_agent output... },
    "timestamp": "2026-03-20T22:45:00"
}
```

### reasoning_agent.py output:
```python
{
    "ticker": "RELIANCE",
    "explanation": "Reliance just showed something we rarely see...",
    "action": "BUY",                  # BUY / WATCH / AVOID
    "confidence_plain": "Very High",  # Very High / High / Medium / Low
    "key_points": [
        "Golden cross detected — 71% win rate historically on this stock",
        "Promoter bought ₹2.3Cr worth last week",
        "Q3 earnings beat by 20.4%"
    ],
    "sources": [
        "NSE bulk deal data, March 15 2026",
        "SEBI PIT disclosure, March 14 2026",
        "yfinance quarterly earnings data"
    ],
    "rag_context_used": True
}
```

---

## FASTAPI ENDPOINTS — WHAT routes.py MUST EXPOSE
```
POST /api/signals          → run full pipeline for a ticker
GET  /api/signals/top      → top 10 convergence scores today  
GET  /api/charts/{ticker}  → chart PNG + pattern data
GET  /api/radar            → today's opportunity feed
POST /api/query            → natural language Q&A via reasoning agent
POST /api/voice            → audio file → STT → query → response
GET  /api/health           → health check
```

---

## CODING RULES — COPILOT MUST FOLLOW THESE

1. Every file starts with all imports, then load_dotenv(), then constants
2. Every API call wrapped in try/except — never let quota errors crash the app
3. All API keys via os.getenv() only — never hardcoded
4. All agent outputs match the schemas above EXACTLY — field names are fixed
5. All NSE tickers use .NS suffix for yfinance
6. Use quarterly_income_stmt not quarterly_earnings (yfinance 1.2.0+)
7. ChromaDB always uses PersistentClient with path="./chroma_db"
8. FinBERT and embedder loaded once at module level, not per-function call
9. Gemini quota errors → fallback gracefully, mark vlm_confirmed=False
10. Groq quota errors → fallback to OpenRouter same prompt same model tier

---

## THE 3-MINUTE DEMO FLOW (build everything toward this)

0:00 — Dashboard loads, radar feed shows 3 stocks with convergence scores
0:20 — Click TATAMOTORS (score 84/100) — chart loads with golden cross marked
0:40 — Reasoning agent explains in plain English with win rate + sources cited
1:20 — Show insider buying + earnings beat data that fed the convergence score
1:40 — VOICE: speak "show me insider buying on Infosys" → Whisper transcribes
2:00 — System queries RAG + returns sourced answer
2:20 — Backtesting panel: golden cross on TATAMOTORS, 71% win rate, 14 occurrences
2:40 — "This runs on 500+ NSE stocks every morning. Zero hallucinations."

---

## WINNING DIFFERENTIATORS — NEVER COMPROMISE

1. Convergence Score — 4 signals → 1 number. Build this perfectly.
2. Per-stock back-tested win rates — not generic, specific to THAT stock
3. Zero hallucinations — every claim cited to a real data source
4. Plain English — Class 10 reading level, no jargon
5. Real-time NSE universe — architecture supports 500+ stocks

---