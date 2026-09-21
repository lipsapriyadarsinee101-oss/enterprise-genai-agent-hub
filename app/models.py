from pydantic import BaseModel, Field
from typing import Literal


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


class Citation(BaseModel):
    source: str
    excerpt: str
    score: float


class AgentStep(BaseModel):
    agent: str
    action: str
    detail: str


class QualityReport(BaseModel):
    citation_present: bool
    citations_valid: bool
    invalid_citations: list[str]
    support_status: Literal["exact_extract", "unverified", "no_evidence", "abstained"]
    unmatched_blocks: list[str] = Field(default_factory=list)
    limitations: str = "Exact extraction checks provenance only; truth, question relevance, completeness, and cross-source contradictions are not verified."


class Usage(BaseModel):
    provider_called: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_status: str = "unavailable: provider does not report billed cost"


class QueryResponse(BaseModel):
    answer: str
    route: str
    citations: list[Citation]
    trace: list[AgentStep]
    confidence: float | None = Field(default=None, deprecated=True, description="Unavailable; no calibrated probability is computed.")
    grounded: bool | None = Field(default=None, deprecated=True, description="Unknown; citation/extraction checks do not establish factual grounding.")
    quality: QualityReport
    abstained: bool
    usage: Usage
    latency_ms: int
    mode: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    mode: str
