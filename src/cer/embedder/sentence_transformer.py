from typing import List, Optional, Union, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from .client import AbstractEmbeddingModel, EmbedderConfig, normalize_embedding

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"


class SentenceTransformerEmbedderConfig(EmbedderConfig):
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


class SentenceTransformerEmbeddingModel(AbstractEmbeddingModel):
    """Embedding model using SentenceTransformers.

    Generates embeddings with the model's native dimension (raw).
    Provides a separate method to resize AND normalize single or batch embeddings post-generation.
    """

    def __init__(
        self,
        config: Optional[SentenceTransformerEmbedderConfig] = None,
    ):
        """
        Initializes the embedding model.

        Args:
            config: The configuration for the embedding model.
        """
        if config is None:
            config = SentenceTransformerEmbedderConfig()
        self.config = config

        # TODO: Utilize a proper tokenizer function for the sentence transformer class
        self.tokenizer = self

        try:
            self.model = SentenceTransformer(config.embedding_model)
            self._native_dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(
                f"Error loading SentenceTransformer model '{config.embedding_model}': {e}"
            )
            raise

    @property
    def native_dimension(self) -> int:
        """Returns the native embedding dimension of the loaded model."""
        return self._native_dimension or 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))  # Simple byte list as tokens

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    async def get_embedding(self, input_data: Union[str, List[str]]) -> List[float]:
        """Generates raw embeddings for a single text or batch of texts using the SentenceTransformer model's native dimension."""
        input_list = [input_data] if isinstance(input_data, str) else input_data
        if not input_list:  # Handle empty list case
            return []
        try:
            embeddings_np = self.model.encode(input_list, normalize_embeddings=True)
            embeddings_normalized = self.resize_and_normalize_embedding(
                embeddings_np[0].tolist(), self.config.embedding_dim
            )
            # TODO: return the average if the input was a list of strings
            return cast(List[float], embeddings_normalized)
        except Exception as e:
            print(f"Error getting SentenceTransformer embeddings: {e}")
            raise

    async def get_batch_embedding(
        self,
        input_data_list: Union[str, List[str]],
    ) -> List[List[float]]:
        input_list = (
            [input_data_list] if isinstance(input_data_list, str) else input_data_list
        )
        if not input_list:
            return []
        try:
            embeddings_np = self.model.encode(input_list, normalize_embeddings=True)
            embeddings_normalized = self.resize_and_normalize_embedding(
                embeddings_np.tolist(), self.config.embedding_dim
            )
            return cast(List[List[float]], embeddings_normalized)
        except Exception as e:
            print(f"Error getting SentenceTransformer batch embeddings: {e}")
            raise

    async def get_query_embedding(
        self, task_description: str, queries: Union[str, List[str]]
    ) -> List[float]:
        """Generates embeddings for a single text or a batch of texts using the SentenceTransformer model's native dimension."""
        input_list = (
            [self.get_detailed_instruct(task_description, query) for query in queries]
            if isinstance(queries, list)
            else [self.get_detailed_instruct(task_description, queries)]
        )
        if not input_list:
            return []
        return await self.get_embedding(input_list)

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def resize_and_normalize_embedding(
        self, embeddings: Union[List[float], List[List[float]]], target_dimension: int
    ) -> Union[List[float], List[List[float]]]:
        """Resizes embedding(s) and L2 normalizes ONLY IF resizing occurred.

        If an input embedding already has the target_dimension, it is returned unchanged.
        Resizing uses truncation or zero-padding.

        Args:
            embeddings: The raw single embedding or batch to resize.
            target_dimension: The desired dimension size.

        Returns:
            The potentially resized and potentially normalized embedding(s) in the original input format.
        """
        is_batch = (
            isinstance(embeddings, list)
            and len(embeddings) > 0
            and isinstance(embeddings[0], list)
        )
        input_was_single = not is_batch

        if input_was_single:
            # Handle single embedding case
            if len(embeddings) == target_dimension:
                return embeddings  # Return original if dimension matches
            else:
                # Resize and normalize since dimension differs
                np_embedding = np.array(embeddings, dtype=np.float32)
                if target_dimension < len(embeddings):
                    resized_emb = np_embedding[:target_dimension]
                else:
                    resized_emb = np.zeros(target_dimension, dtype=np.float32)
                    resized_emb[: len(embeddings)] = np_embedding
                # Normalize the single resized embedding
                normalized_result = normalize_embedding(resized_emb)
                return (
                    normalized_result.tolist()
                    if isinstance(normalized_result, np.ndarray)
                    else normalized_result
                )
        else:
            # Handle batch case
            embeddings_batch = cast(List[List[float]], embeddings)
            if not embeddings_batch:
                return []

            resized_batch = []
            needs_normalization = False  # Flag to track if any resizing occurred
            for emb in embeddings_batch:
                current_dimension = len(emb)
                np_embedding = np.array(emb, dtype=np.float32)

                if current_dimension == target_dimension:
                    # Dimension matches, add original (as np array) without resizing
                    resized_emb = np_embedding
                else:
                    # Dimension differs, resize and flag for normalization
                    needs_normalization = True
                    if target_dimension < current_dimension:
                        resized_emb = np_embedding[:target_dimension]
                    else:
                        resized_emb = np.zeros(target_dimension, dtype=np.float32)
                        resized_emb[:current_dimension] = np_embedding
                resized_batch.append(resized_emb)

            # Convert list of numpy arrays to a 2D numpy array
            final_batch_np = np.array(resized_batch)

            # Normalize the entire batch ONLY if at least one embedding was resized
            if needs_normalization:
                final_batch_np = normalize_embedding(final_batch_np)  # type: ignore

            # Convert back to list of lists and return
            return cast(
                List[List[float]],
                (
                    final_batch_np.tolist()
                    if isinstance(final_batch_np, np.ndarray)
                    else final_batch_np
                ),
            )
