"""schemas.py
=================================
Pydantic schemas for FINANCIAL_INTEL FastAPI endpoints.
"""

from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_TOP_K = 5
DEFAULT_TOP_N = 10


class SignalsRequest(BaseModel):
    """Request payload to run full signal pipeline for one ticker."""

    ticker: str = Field(..., description="NSE ticker without .NS suffix")
    document_type: str = Field(default="earnings_call", description="Sentiment source type")
    use_vlm: bool = Field(default=True, description="Enable Gemini chart confirmation")
    chart_days: int = Field(default=120, ge=30, le=500)
    holding_days: int = Field(default=20, ge=5, le=120)


class SignalBreakdownModel(BaseModel):
    """Convergence score contribution breakdown."""

    chart_contribution: float
    insider_contribution: float
    earnings_contribution: float
    sentiment_contribution: float


class SignalsResponse(BaseModel):
    """Combined response for /api/signals endpoint."""

    ticker: str
    convergence_score: float
    convergence_label: str
    signal_breakdown: SignalBreakdownModel
    signals_present: list[str]
    chart_data: dict[str, Any]
    radar_data: dict[str, Any]
    sentiment_data: dict[str, Any]
    reasoning_data: dict[str, Any]
    timestamp: str
    pipeline_failed: Optional[bool] = False


class TopSignalsItem(BaseModel):
    """One ranked ticker result for /api/signals/top."""

    ticker: str
    convergence_score: float
    convergence_label: str
    radar_composite_score: float
    insider_signal_strength: float
    earnings_beat_pct: float


class TopSignalsResponse(BaseModel):
    """Response for /api/signals/top."""

    generated_at: str
    index_name: str
    scanned_count: int
    returned_count: int
    market_status: dict[str, Any]
    items: list[TopSignalsItem]


class ChartResponse(BaseModel):
    """Response for /api/charts/{ticker}."""

    ticker: str
    chart_path: str
    chart_image_base64: str
    patterns: list[dict[str, Any]]
    chart_confidence: float
    overall_bias: str
    vlm_confirmed: bool


class RadarResponse(BaseModel):
    """Response for /api/radar."""

    generated_at: str
    index_name: str
    scanned_count: int
    returned_count: int
    opportunities: list[dict[str, Any]]


class QueryRequest(BaseModel):
    """Request payload for /api/query."""

    query: str
    ticker: Optional[str] = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=15)
    portfolio: Optional[dict[str, int]] = Field(default=None, description="User portfolio: ticker -> quantity")


class QueryResponse(BaseModel):
    """Response payload for /api/query."""

    query: str
    ticker: str
    convergence_data: dict[str, Any]
    rag_context: list[dict[str, Any]]
    reasoning_data: dict[str, Any]


class VoiceResponse(BaseModel):
    """Response payload for /api/voice."""

    transcript: str
    query_response: QueryResponse


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str
    timestamp: str
    version: str
    components: dict[str, bool]


class AlertItem(BaseModel):
    """One autonomous pipeline alert with full evidence chain."""

    ticker: str
    timestamp: str
    cycle: int = 0
    convergence_score: float = 0.0
    convergence_label: str = ""
    action: str = "WATCH"
    confidence_plain: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    time_horizon: Optional[float] = None
    patterns: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    top_signal_headline: str = ""
    rag_context_count: int = 0
    pipeline_steps: list[str] = Field(default_factory=list)
    autonomous: bool = True


class AlertsLatestResponse(BaseModel):
    """Response for GET /api/alerts/latest."""

    count: int
    alerts: list[AlertItem]


class ErrorResponse(BaseModel):
    """Error response model for route failures."""

    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
