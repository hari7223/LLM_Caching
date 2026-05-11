import asyncio
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    async def get_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self._compute_embedding,
            text
        )
        return embedding

    def _compute_embedding(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()