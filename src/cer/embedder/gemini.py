from typing import Any, Dict, List, Literal, Optional, Union, cast

from google.oauth2.service_account import Credentials

from .client import AbstractEmbeddingModel, EmbedderConfig, semaphore_gather

DEFAULT_EMBEDDING_MODEL = "text-embedding-large-exp-03-07"


class GeminiEmbedderConfig(EmbedderConfig):
    api_key: str | None = None
    project: str | None = None
    location: str | None = None
    service_account_info: Dict[str, Any] | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class GeminiEmbeddingModel(AbstractEmbeddingModel):
    # Default dimension set here
    DEFAULT_DIMENSION = 1024

    def __init__(self, config: Optional[GeminiEmbedderConfig] = None):
        """
        Initializes the embedding model.

        Args:
            config: The configuration for the embedding model.
        """
        try:
            from google import genai
            from google.genai import types

            self.genai = genai
            self.genai_types = types
        except ImportError:
            raise ImportError("Google GenAI client not found")

        if config is None:
            config = GeminiEmbedderConfig()
        self.config = config

        if config.service_account_info is None:
            raise ValueError(
                "Service account file is required for Gemini embedding model."
            )

        # Configure the Gemini API
        self.client = self.genai.Client(
            vertexai=True if config.project else False,
            api_key=config.api_key,
            project=config.project,
            location=config.location,
            credentials=(
                Credentials.from_service_account_info(  # type: ignore
                    config.service_account_info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if config.service_account_info
                else None
            ),
        )

    async def get_embedding(
        self,
        input_data: Union[str, list[str]],  # Simplified type hint
        task_type: Literal[
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "CODE_RETRIEVAL_QUERY",
        ] = "RETRIEVAL_QUERY",
    ) -> List[float]:
        input_list = [input_data] if isinstance(input_data, str) else input_data
        if not input_list:  # Handle empty list case
            return []
        # Ensure all items in input_list are strings, as embed_content expects List[str]
        if not all(isinstance(item, str) for item in input_list):
            raise TypeError(
                "All items in input_data must be strings for Gemini embedding."
            )
        try:
            response = await self.client.aio.models.embed_content(
                model=self.config.embedding_model or DEFAULT_EMBEDDING_MODEL,
                contents=input_list,
                config=self.genai_types.EmbedContentConfig(
                    output_dimensionality=self.config.embedding_dim,
                    task_type=task_type,
                ),
            )
            # TODO: return the average if the input was a list of strings
            return cast(List[float], response.embeddings[0].values)
        except Exception as e:
            print(f"Error getting Gemini embeddings: {e}")  # Generic error message
            raise

    async def get_batch_embedding(
        self,
        input_data_list: Union[str, list[str]],  # Simplified type hint
        task_type: Literal[
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "CODE_RETRIEVAL_QUERY",
        ] = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        input_list = (
            [input_data_list] if isinstance(input_data_list, str) else input_data_list
        )
        if not input_list:  # Handle empty list case
            return []
        # Ensure all items in input_list are strings
        if not all(isinstance(item, str) for item in input_list):
            raise TypeError(
                "All items in input_data_list must be strings for Gemini batch embedding."
            )
        try:
            responses = await semaphore_gather(
                *[
                    self.get_embedding(
                        [input_data_item], task_type
                    )  # Pass as list of str
                    for input_data_item in input_list
                ]
            )
            return [response for response in responses]
        except Exception as e:
            print(
                f"Error getting Gemini Batch embeddings: {e}"
            )  # Generic error message
            raise
