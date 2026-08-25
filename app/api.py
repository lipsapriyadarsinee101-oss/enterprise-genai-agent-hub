from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.service import AgentHub


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.hub = AgentHub(get_settings())
    yield


app = FastAPI(
    title="Enterprise GenAI Agent Hub",
    description="Explainable multi-agent RAG API for enterprise knowledge.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    hub: AgentHub = app.state.hub
    return HealthResponse(status="healthy", documents=len(hub.kb.documents), mode="demo" if hub.settings.demo_mode else "openai")


@app.get("/v1/documents", response_model=list[str])
def documents() -> list[str]:
    return app.state.hub.kb.documents


@app.post("/v1/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return app.state.hub.query(payload.question, payload.top_k)
