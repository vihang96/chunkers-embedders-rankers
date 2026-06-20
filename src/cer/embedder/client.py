import asyncio
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from typing import Any, List, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field

EMBEDDING_DIM = 1024
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 20))

T = TypeVar("T")


class EmbedderConfig(BaseModel):
    embedding_model: str
    embedding_dim: int = Field(default=EMBEDDING_DIM, frozen=True)

# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(
    *coroutines: Coroutine[None, None, T],
    max_coroutines: int = SEMAPHORE_LIMIT,
) -> List[Union[T, BaseException]]:
    """Execute multiple coroutines with a maximum concurrency limit.

    All coroutines are launched immediately but the semaphore ensures at most
    ``max_coroutines`` run at the same time.  When any coroutine finishes, the
    semaphore releases and the next waiting coroutine starts — a fast page is
    never blocked behind a slow sibling in the same batch.

    Args:
        *coroutines: Variable number of coroutines to execute
        max_coroutines: Maximum number of coroutines to run concurrently (default: 20)

    Returns:
        List of results or exceptions from the coroutines (order preserved).
        Exceptions are returned as values rather than raised.
    """
    semaphore = asyncio.Semaphore(max_coroutines)

    async def _wrap(coro: Coroutine[None, None, T]) -> T:
        async with semaphore:
            return await coro

    return list(
        await asyncio.gather(*[_wrap(c) for c in coroutines], return_exceptions=True)
    )


def normalize_embedding(
    embedding: Union[List[float], np.ndarray],  # type: ignore
) -> Union[List[float], np.ndarray]:  # type: ignore
    """L2 normalize an embedding vector or a batch of vectors."""
    x = np.array(embedding, dtype=np.float32)  # Ensure it's a numpy array

    if x.ndim == 1:
        # Handle single vector
        norm = np.linalg.norm(x)
        if norm == 0:
            warnings.warn("Attempted to normalize a zero vector (1D).")
            # Return original zero vector (or array)
            return embedding if isinstance(embedding, list) else x
        normalized_x = x / norm
        # Return in original format (list or ndarray)
        return normalized_x.tolist() if isinstance(embedding, list) else normalized_x
    elif x.ndim == 2:
        # Handle batch of vectors
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        # Use np.where to avoid division by zero
        normalized_x = np.where(norm == 0, x, x / norm)
        if isinstance(embedding, list) and all(isinstance(i, list) for i in embedding):
            return normalized_x.tolist()  # type: ignore
        else:
            return normalized_x
    else:
        raise ValueError("Input must be a 1D or 2D array/list of lists.")


ConfigType = TypeVar('ConfigType', bound=EmbedderConfig)


class AbstractEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def get_embedding(self, input_data: Union[str, List[str]]) -> List[float]:
        """
        Converts a single text or a batch of texts into a list of embeddings.

        This method typically returns raw embeddings from the underlying model/API.
        Normalization should be handled by specific methods or by the caller if needed,
        especially if dimensionality changes are performed post-generation.

        Args:
            texts: A single input text (str) or a list of input texts (List[str]) to embed.

        Returns:
            An embedding (List[float]). If the input was a list of strings,
            the outer list will contain a single averaged embedding.
        """
        pass

    @abstractmethod
    async def get_batch_embedding(
        self,
        input_data_list: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Converts a single text or a batch of texts into a list of embeddings.

        This method typically returns raw embeddings from the underlying model/API.
        Normalization should be handled by specific methods or by the caller if needed,
        especially if dimensionality changes are performed post-generation.

        Args:
            texts: A single input text (str) or a list of input texts (List[str]) to embed.

        Returns:
            A list of embeddings (List[List[float]]). If the input was a single string,
            the outer list will contain a single embedding.
        """
        pass
