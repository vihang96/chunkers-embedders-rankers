from typing import Awaitable, Callable, Generic, List, TypeVar

from pydantic import BaseModel

# Define a TypeVar for Pydantic models
PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# Define the callable type that EmbeddingRetriever expects for its reranker_func
RerankerCallable = Callable[[str, List[PydanticModel]], Awaitable[List[PydanticModel]]]


class ToolWithEmbedding(BaseModel, Generic[PydanticModel]):
    tool_schema: PydanticModel
    embedding: List[float]
