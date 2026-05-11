from fastapi import FastAPI, HTTPException, Request
import httpx
from app.models import ChatRequest, ChatResponse
from app.proxy import LLMProxy
from app.config import get_settings
from app.embedding_service import EmbeddingService

from contextlib import asynccontextmanager
from redis.asyncio import ConnectionPool, Redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    settings = get_settings()
    pool = ConnectionPool.from_url(
        settings.redis_url,
        max_connections=10,
        decode_responses=True
    )
    app.state.redis = Redis(connection_pool=pool)

    # Embedding service
    app.state.embedding_service = EmbeddingService()
    yield
    # shutdown
    await app.state.redis.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat")
async def chat(request: ChatRequest, req: Request):
    settings = get_settings()
    proxy = LLMProxy(settings, req.app.state.redis, req.app.state.embedding_service)
    try:
        return await proxy.resolve(request)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timed out")
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="LLM API returned an error")
    
@app.get("/health")
async def health(request: Request):
    redis_status = "not_connected"
    try:
        await request.app.state.redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "not_connected"

    return {
        "status": "ok",
        "components": {
            "proxy": "ok",
            "redis": redis_status,
            "pgvector": "not_connected",
            "kafka": "not_connected",
        }
    }