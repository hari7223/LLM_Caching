import asyncio
from app.embedding_service import EmbeddingService
import numpy as np

async def test():
    service = EmbeddingService()
    
    e1 = await service.get_embedding("Tell me a joke")
    e2 = await service.get_embedding("Tell me 2 jokes")

    
    
    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"Joke queries: {cosine_similarity(e1, e2):.4f}")

asyncio.run(test())