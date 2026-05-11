from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class CacheStatus(str, Enum):
    HIT_EXACT = "hit_exact"       # Redis exact match (Component 2)
    HIT_SEMANTIC = "hit_semantic"  # pgvector semantic match (Component 4)
    MISS = "miss"                  # Fell through to LLM


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to the LLM")
    model: str = Field(default="mock-llm", description="LLM model identifier")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    session_id: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    query: str
    response: str
    model: str
    cache_status: CacheStatus
    latency_ms: float
    similarity_score: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    components: dict