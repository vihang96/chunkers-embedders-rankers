from collections.abc import Iterable
from typing import List, Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types import EmbeddingModel

from .client import AbstractEmbeddingModel, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


class OpenAIEmbedderConfig(EmbedderConfig):
    api_key: str | None = None
    base_url: str | None = None
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL


class OpenAIEmbeddingModel(AbstractEmbeddingModel):
    """Embedding model using OpenAI. Defaults to 1024 dimensions if not specified."""

    # Default dimension set here
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        config: Optional[OpenAIEmbedderConfig] = None,
        client: Union[AsyncOpenAI, AsyncAzureOpenAI, None] = None,
    ):
        """
        Initializes the embedding model.

        Args:
            config: The configuration for the embedding model.
            client: The client to use for the embedding model (optional). Needed for openai models.
        """
        if config is None:
            config = OpenAIEmbedderConfig()
        self.config = config

        if client is not None:
            self.client = client
        else:
            if config.api_key is None:
                raise ValueError("API key is required for OpenAI embedding model.")
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    async def get_embedding(
        self,
        input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> List[float]:
        """Generates embeddings for a single text or a batch of texts using the OpenAI API. Returns raw embeddings."""
        input_list = [input_data] if isinstance(input_data, str) else input_data
        if not input_list:  # Handle empty list case
            return []
        try:
            response = await self.client.embeddings.create(
                input=input_data,
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dim,
            )
            # TODO: return the average if the input was a list of strings
            return response.data[0].embedding[: self.config.embedding_dim]
        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")  # Generic error message
            raise

    async def get_batch_embedding(
        self,
        input_data_list: str | list[str] | Iterable[int] | Iterable[Iterable[int]],
    ) -> List[List[float]]:
        input_list = (
            [input_data_list] if isinstance(input_data_list, str) else input_data_list
        )
        if not input_list:  # Handle empty list case
            return []
        try:
            response = await self.client.embeddings.create(
                input=input_list,
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dim,
            )
            return [
                item.embedding[: self.config.embedding_dim] for item in response.data
            ]
        except Exception as e:
            print(f"Error getting OpenAI embeddings: {e}")  # Generic error message
            raise
