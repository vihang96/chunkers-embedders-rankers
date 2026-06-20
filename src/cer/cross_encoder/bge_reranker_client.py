import asyncio

from sentence_transformers import CrossEncoder

from .client import CrossEncoderClient


class BGERerankerClient(CrossEncoderClient):
    def __init__(self) -> None:
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        input_pairs = [[query, passage] for passage in passages]

        # Run the synchronous predict method in an executor
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self.model.predict, input_pairs)

        ranked_passages = sorted(
            [
                (passage, float(score))
                for passage, score in zip(passages, scores, strict=False)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked_passages
