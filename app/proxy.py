import time
import asyncio
import httpx
import hashlib

import logging
logger = logging.getLogger(__name__)



from app.models import CacheStatus, ChatResponse, ChatRequest

class LLMProxy:
    
    def __init__(self, settings, redis):
        self.settings = settings
        self.redis = redis

    async def resolve(self, request):
        start = time.perf_counter()
        # 1. Check exact cache (Redis)
        result = await self._check_exact_cache(request)
        if result:
            response_text, cache_status = result
            return self._build_response(request, response_text, cache_status, start)

        # 2. Check semantic cache (pgvector)
        result = await self._check_semantic_cache(request)
        if result:
            response_text, cache_status, similarity_score = result
            response = self._build_response(request, response_text, cache_status, start)
            response.similarity_score = similarity_score
            return response
        
        # 3. Fall back to LLM
        result = await self._call_llm(request)
        await self._store_exact_cache(request, result)
        await self._store_semantic_cache(request, result)
        return self._build_response(request, result, CacheStatus.MISS, start)
    
    def _build_response(self, request, response_text:str, cache_status, start:float):
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            query=request.query,
            response=response_text,
            model=request.model,
            cache_status=cache_status,
            latency_ms=round(latency_ms, 2)
        )
    
    async def _call_llm(self, request):
        if self.settings.llm_api_url == "mock":
            return await self._mock_llm(request)
        return await self._real_llm(request)
    
    async def _mock_llm(self, request):
        await asyncio.sleep(0.05)
        return f"[MOCK] You asked: '{request.query}'"
    
    async def _real_llm(self, request: ChatRequest) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": request.model,
                "messages": [{"role": "user", "content": request.query}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
            resp = await client.post(
                f"{self.settings.llm_api_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # ── Cache stubs (filled by later components) ─────────────────────────────

    async def _check_exact_cache(self, request: ChatRequest):
        if self.redis is None:
            return None
        
        try:
            key = self._make_cache_key(request.query)
            cached = await self.redis.get(key)
            if cached:
                return cached, CacheStatus.HIT_EXACT
            return None
        except Exception as e:
            logger.warning(f"Redis check failed: {e}")
            return None
        
    def _make_cache_key(self, query: str) -> str:
        normalised = query.lower().strip()
        return hashlib.md5(normalised.encode()).hexdigest()

    async def _check_semantic_cache(self, request: ChatRequest):

        return None

    async def _store_exact_cache(self, request: ChatRequest, response: str):
        if self.redis is None:
            return
        try:

            key = self._make_cache_key(request.query)
            await self.redis.setex(key, self.settings.redis_ttl_seconds, response)
        
        except Exception as e:
            logger.warning(f"Redis store failed: {e}")



    async def _store_semantic_cache(self, request: ChatRequest, response: str):
        pass


