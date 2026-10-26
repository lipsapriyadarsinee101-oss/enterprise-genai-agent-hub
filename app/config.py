from typing import Literal
from functools import lru_cache
import os

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # Allows dependency-light evaluation environments.
    pass


class Settings(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    demo_mode: bool = Field(default_factory=lambda: os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes"})
    knowledge_dir: str = Field(default_factory=lambda: os.getenv("KNOWLEDGE_DIR", "data/knowledge"))
    retrieval_backend: Literal['tfidf', 'faiss', 'opensearch'] = Field(default_factory=lambda: os.getenv('RETRIEVAL_BACKEND', 'tfidf'))
    embedding_model: str = Field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'))
    embedding_dimensions: int = Field(default=1536, ge=1)
    opensearch_url: str = Field(default_factory=lambda: os.getenv('OPENSEARCH_URL', 'http://localhost:9200'))
    opensearch_index: str = Field(default_factory=lambda: os.getenv('OPENSEARCH_INDEX', 'agent-hub-v1'))
    langfuse_enabled: bool = Field(default_factory=lambda: os.getenv('LANGFUSE_ENABLED', 'false').lower() == 'true')
    max_context_chars: int = Field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "8000")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
