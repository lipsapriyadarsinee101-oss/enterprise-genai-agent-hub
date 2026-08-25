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
    max_context_chars: int = Field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "8000")))


@lru_cache
def get_settings() -> Settings:
    return Settings()
