import logging
import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import tiktoken

from ..utils.schemas import PydanticModel
from ..utils.yaml_helpers import convert_pydantic_to_yaml, convert_pydantic_to_yaml_selective
from .client import AbstractEmbeddingModel

# Optional backends: only available when the matching extra is installed. We keep
# references (possibly None) so the isinstance-based dispatch below can be guarded.
try:
    from .gemini import GeminiEmbeddingModel
except ImportError:
    GeminiEmbeddingModel = None  # type: ignore[assignment, misc]

try:
    from .openai import OpenAIEmbeddingModel
except ImportError:
    OpenAIEmbeddingModel = None  # type: ignore[assignment, misc]

try:
    from .sentence_transformer import SentenceTransformerEmbeddingModel
except ImportError:
    SentenceTransformerEmbeddingModel = None  # type: ignore[assignment, misc]

# --- Tokenizer and Chunking Helpers ---
logger = logging.getLogger(__name__)


def _isinstance(model: object, cls: object) -> bool:
    """isinstance check that is safe when an optional backend isn't installed."""
    return cls is not None and isinstance(model, cls)  # type: ignore[arg-type]


def _get_model_tokenizer_info(
    model: AbstractEmbeddingModel,
) -> Tuple[object, int, Callable[[str], int]]:
    """Gets tokenizer, max tokens, and token counting function for a model."""
    if _isinstance(model, SentenceTransformerEmbeddingModel):
        st_tokenizer = model.tokenizer
        # Use model_max_length, default to 512 if unavailable
        max_tokens = getattr(st_tokenizer, "model_max_length", 512)

        def count_func(text: str) -> int:
            return len(st_tokenizer.encode(text, add_special_tokens=False))

        return st_tokenizer, max_tokens, count_func
    elif _isinstance(model, OpenAIEmbeddingModel):
        try:
            # Assuming text-embedding-3 models use cl100k_base
            openai_tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
        except Exception:
            warnings.warn("tiktoken model not found, falling back to cl100k_base.")
            openai_tokenizer = tiktoken.get_encoding("cl100k_base")
        max_tokens = 8191  # Standard for text-embedding-3

        def count_func(text: str) -> int:
            return len(openai_tokenizer.encode(text))

        return openai_tokenizer, max_tokens, count_func
    elif _isinstance(model, GeminiEmbeddingModel):
        # Gemini uses its own count_tokens method
        gemini_tokenizer = model.client
        max_tokens = 8191  # Known limit for gemini-embedding models
        # Ensure model name format for count_tokens is correct (without 'models/')
        base_model_name = model.config.embedding_model.replace("models/", "")

        def count_func(text: str) -> int:
            try:
                # TODO: Fix this error. 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Should provide instances for text model prediction.', 'status': 'INVALID_ARGUMENT'}}.
                return int(
                    gemini_tokenizer.models.count_tokens(
                        model=base_model_name, contents=text
                    ).total_tokens
                )
            except Exception as e:
                warnings.warn(
                    f"Gemini count_tokens failed: {e}. Returning rough estimate."
                )
                return round(len(text.split()) * 1.33)  # Fallback estimate

        return gemini_tokenizer, max_tokens, count_func
    elif hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        max_tokens = getattr(model, "model_max_length", 512)

        def count_func(text: str) -> int:
            return len(tokenizer.encode(text))

        return tokenizer, max_tokens, count_func
    else:
        raise TypeError(f"Unsupported model type for tokenizer info: {type(model)}")


def _chunk_text(
    text: str, tokenizer: object, max_tokens: int, overlap: int = 50
) -> List[str]:
    # genai is only needed for the Gemini tokenizer branch; treat it as optional.
    try:
        from google import genai
    except ImportError:
        genai = None  # type: ignore[assignment]

    """Chunks text into overlapping segments based on token limits."""
    if isinstance(tokenizer, tiktoken.Encoding) or (
        hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode")
    ):
        # Tiktoken or HuggingFace Tokenizer
        tokens = tokenizer.encode(text)
        decode_func = tokenizer.decode
    elif genai is not None and tokenizer is genai:
        # Cannot directly tokenize with genai count_tokens, use basic split
        warnings.warn(
            "Cannot perform accurate token-based chunking for Gemini, using word split."
        )
        words = text.split()
        # Use max_tokens as approximate word count, adjust overlap
        max_len_words = max_tokens // 2  # Rough estimate
        overlap_words = overlap // 2
        chunks = []
        for i in range(0, len(words), max_len_words - overlap_words):
            chunk = words[i : i + max_len_words]
            if chunk:
                chunks.append(" ".join(chunk))
        return chunks
    else:
        raise TypeError("Unsupported tokenizer type for chunking.")

    if not tokens:
        return []

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_token_ids = tokens[start_idx:end_idx]
        # Decode carefully, handling potential errors
        try:
            chunk_text = decode_func(chunk_token_ids)
            chunks.append(chunk_text)
        except Exception as e:
            warnings.warn(f"Error decoding token chunk: {e}. Skipping chunk.")

        if end_idx == len(tokens):
            break  # Reached the end

        # Move start index for the next chunk, considering overlap
        start_idx += max_tokens - overlap
        # Ensure start_idx doesn't go backwards or stay static if overlap is large
        start_idx = max(start_idx, end_idx - max_tokens + 1)  # Prevent getting stuck

    return chunks


# --- Main Embedding Generation Function ---


async def generate_embeddings(
    model: AbstractEmbeddingModel,
    texts: Union[str, List[str]],
    input_type: Literal["query", "document", "tool"],
    task_description: Optional[str] = "Represent the document for retrieval:",
) -> List[List[float]]:
    """Generates embeddings for query/document(s), handling token limits and model specifics.

    Args:
        model: An initialized instance of AbstractEmbeddingModel.
        texts: A single text string or a list of text strings.
        input_type: Either "query" or "document".
        task_description: Optional task description used for instruction-following models
                          like SentenceTransformer E5-instruct (especially for queries).
                          Defaults to a retrieval document task description.

    Returns:
        A list of embeddings (List[List[float]]), corresponding to the input texts.
    """
    input_list: list[str] = [texts] if isinstance(texts, str) else texts

    if not input_list:
        return []

    tokenizer, max_tokens, count_func = _get_model_tokenizer_info(model)
    final_embeddings: List[List[float] | None] = [None] * len(input_list)
    batch_texts_to_process = []
    batch_indices = []
    gemini_task_type = None  # Store task type for Gemini batch call

    for idx, text in enumerate(input_list):
        processed_text: str = text
        current_gemini_task = None

        # Apply prefixes / specific handling
        if (
            _isinstance(model, SentenceTransformerEmbeddingModel)
            and "e5" in model.config.embedding_model
        ):
            if input_type == "query":
                if hasattr(model, "get_detailed_instruct"):
                    processed_text = model.get_detailed_instruct(
                        task_description or "", text
                    )
                else:
                    processed_text = f"query: {text}"
            elif input_type == "document":
                processed_text = f"passage: {text}"
            elif input_type == "tool":
                processed_text = f"tool: {text}"
        elif _isinstance(model, GeminiEmbeddingModel):
            # Determine task type for Gemini
            current_gemini_task = (
                "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
            )
            if gemini_task_type is None:
                gemini_task_type = current_gemini_task  # Set for batch
            elif gemini_task_type != current_gemini_task:
                # This scenario (mixing query/doc in one call) isn't handled by the current batch logic
                # Requires separate batch calls or processing one-by-one. Raising error for now.
                raise ValueError(
                    "Mixing query and document types in a single call is not supported for Gemini model via this function."
                )

        # Check token count
        token_count = count_func(processed_text)

        # Handle chunking for documents exceeding limits
        if input_type in ["document", "tool"] and token_count > max_tokens:
            print(
                f"Document at index {idx} (len {token_count} tokens) exceeds limit {max_tokens}, chunking..."
            )
            chunks = _chunk_text(processed_text, tokenizer, max_tokens, overlap=50)
            if not chunks:
                warnings.warn(
                    f"Document at index {idx} could not be chunked, placing empty embedding."
                )
                final_embeddings[idx] = []  # Or handle as error
                continue

            logger.info(
                f"Sending {len(chunks)} chunks to model.get_batch_embedding for document at index {idx}."
            )

            # Use the model's get_batch_embedding for chunks.
            chunk_embeddings = await model.get_batch_embedding(chunks)

            if not chunk_embeddings:
                warnings.warn(
                    f"Got empty embeddings for chunks of document at index {idx}."
                )
                final_embeddings[idx] = []
                continue

            # Average chunk embeddings
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            final_embeddings[idx] = avg_embedding
        else:
            # Add to batch for direct processing
            batch_texts_to_process.append(processed_text)
            batch_indices.append(idx)

    # Process the batch of texts that didn't need chunking
    if batch_texts_to_process:
        # The batch_kwargs related to task_type are removed as current model methods don't accept them.
        # The Gemini model is expected to handle task_type internally.
        # The gemini_task_type variable is determined and validated for consistency within this function.

        logger.info(
            f"Sending final batch of {len(batch_texts_to_process)} texts to model.get_batch_embedding."
        )

        # Call the model's get_batch_embedding for the batch
        direct_embeddings = await model.get_batch_embedding(batch_texts_to_process)

        # Place the results into the final list using original indices
        for i, original_idx in enumerate(batch_indices):
            if i < len(direct_embeddings):
                final_embeddings[original_idx] = direct_embeddings[i]
            else:
                warnings.warn(
                    f"Mismatch between batch indices and embeddings returned for index {original_idx}."
                )
                final_embeddings[original_idx] = []  # Handle error case

    # Always return List[List[float]] for consistency
    # Filter out potential None placeholders if errors occurred
    return [emb if emb is not None else [] for emb in final_embeddings]


async def generate_object_embeddings(
    model: AbstractEmbeddingModel,
    objects: List[PydanticModel],
    input_type: Literal["query", "document", "tool"] = "tool",
    fields: Optional[List[str]] = None,
) -> List[List[float]]:
    """Generates embeddings for a list of objects. The objects are converted to yaml string representation and then embedded."""
    return await generate_embeddings(
        model,
        [
            (
                convert_pydantic_to_yaml_selective(obj, fields)
                if fields
                else convert_pydantic_to_yaml(obj)
            )
            for obj in objects
        ],
        input_type,
    )
