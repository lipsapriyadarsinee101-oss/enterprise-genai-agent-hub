from pydantic import BaseModel, Field


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


class QueryResponse(BaseModel):
    answer: str
    route: str
    citations: list[Citation]
    trace: list[AgentStep]
    confidence: float = Field(ge=0, le=1)
    grounded: bool
    latency_ms: int
    mode: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    mode: str
