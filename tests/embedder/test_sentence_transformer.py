import numpy as np
import pytest

from cer.embedder import (
    SentenceTransformerEmbedderConfig,
    SentenceTransformerEmbeddingModel,
    normalize_embedding,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


def test_get_detailed_instruct() -> None:
    task = "Classify the sentiment."
    query = "This is a great movie!"
    expected = f"Instruct: {task}\nQuery: {query}"
    st_embedding_model = SentenceTransformerEmbeddingModel()
    assert st_embedding_model.get_detailed_instruct(task, query) == expected


# --- Tests for resize_and_normalize_embedding ---
@pytest.fixture
def st_config() -> SentenceTransformerEmbedderConfig:
    # Using default model name, but it won't be loaded for these specific method tests
    # The embedding_dim in config is used by the class methods, but resize_and_normalize takes target_dimension directly
    return SentenceTransformerEmbedderConfig(embedding_dim=128)


@pytest.fixture
def st_model_for_resize(
    st_config: SentenceTransformerEmbedderConfig,
) -> SentenceTransformerEmbeddingModel:
    # We need an instance to call the method, but don't need to load the actual heavy model
    # We can mock the parts of __init__ that load the model if SentenceTransformer is not available
    class MockSTModel(SentenceTransformerEmbeddingModel):
        def __init__(self, config: SentenceTransformerEmbedderConfig) -> None:
            self.config = config
            # Mock properties that would be set by SentenceTransformer
            self._native_dimension = (
                256  # Assume a native dimension for testing resize logic
            )

    return MockSTModel(config=st_config)


def test_resize_single_embedding_truncate(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    embedding = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]  # Native dim for mock is 256, but this is input
    target_dim = 3
    resized = st_model_for_resize.resize_and_normalize_embedding(embedding, target_dim)
    assert len(resized) == target_dim
    assert np.isclose(
        np.linalg.norm(resized), 1.0
    )  # Should be normalized because resized
    assert np.allclose(resized, normalize_embedding([1.0, 2.0, 3.0]))


def test_resize_single_embedding_pad(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    embedding = [1.0, 2.0, 3.0]
    target_dim = 5
    resized = st_model_for_resize.resize_and_normalize_embedding(embedding, target_dim)
    assert len(resized) == target_dim
    assert np.isclose(
        np.linalg.norm(resized), 1.0
    )  # Should be normalized because resized
    expected_padded = [1.0, 2.0, 3.0, 0.0, 0.0]
    assert np.allclose(resized, normalize_embedding(expected_padded))


def test_resize_single_embedding_no_change(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    embedding = [1.0, 2.0, 3.0]
    target_dim = 3
    resized = st_model_for_resize.resize_and_normalize_embedding(embedding, target_dim)
    assert len(resized) == target_dim
    # If an input embedding already has the target_dimension, it is returned unchanged (NO normalization)
    assert np.allclose(resized, embedding)
    assert not np.isclose(np.linalg.norm(resized), 1.0)


def test_resize_batch_embeddings(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    batch_embeddings = [
        [1.0, 2.0, 3.0, 4.0],  # Truncate to 3, then normalize
        [1.0, 2.0],  # Pad to 3, then normalize
        [1.0, 2.0, 3.0],  # No change in size
    ]
    target_dim = 3
    resized_batch = st_model_for_resize.resize_and_normalize_embedding(
        batch_embeddings, target_dim
    )

    assert isinstance(resized_batch, list)
    assert len(resized_batch) == 3

    assert (
        isinstance(resized_batch[0], list)
        and isinstance(resized_batch[1], list)
        and isinstance(resized_batch[2], list)
    )
    assert len(resized_batch[0]) == target_dim
    assert len(resized_batch[1]) == target_dim
    assert len(resized_batch[2]) == target_dim

    # First two were resized, so the whole batch should be normalized.
    # The third one was not resized itself, but because others were, it gets normalized as part of the batch.
    # This is because needs_normalization flag is set if ANY embedding is resized.
    for res_emb in resized_batch:
        assert np.isclose(
            np.linalg.norm(res_emb), 1.0
        ), f"Embedding {res_emb} not normalized"

    expected_0_norm = normalize_embedding([1.0, 2.0, 3.0])
    expected_1_norm = normalize_embedding([1.0, 2.0, 0.0])
    expected_2_norm = normalize_embedding(
        [1.0, 2.0, 3.0]
    )  # Original, but now normalized due to batch behavior

    assert np.allclose(resized_batch[0], expected_0_norm)
    assert np.allclose(resized_batch[1], expected_1_norm)
    assert np.allclose(resized_batch[2], expected_2_norm)


def test_resize_batch_embeddings_no_resize_needed(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    batch_embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    target_dim = 3
    resized_batch = st_model_for_resize.resize_and_normalize_embedding(
        batch_embeddings, target_dim
    )
    # No resizing occurred for any embedding, so batch should be returned as is (no normalization)
    assert np.allclose(resized_batch, batch_embeddings)
    assert not np.isclose(np.linalg.norm(resized_batch[0]), 1.0)


def test_resize_empty_batch(
    st_model_for_resize: SentenceTransformerEmbeddingModel,
) -> None:
    # Returns resized 0 vector for empty batch
    assert st_model_for_resize.resize_and_normalize_embedding([], 128) == [0.0] * 128


# This requires SentenceTransformer to be a test dependency.
@pytest.mark.skipif(
    SentenceTransformer is None, reason="sentence-transformers library not installed"
)
@pytest.mark.asyncio
async def test_st_get_batch_embedding_real_model(
    st_config: SentenceTransformerEmbedderConfig,
) -> None:
    # This test uses the actual SentenceTransformer model.
    texts = ["hello world", "another text"]

    # Instantiate our class; it will load the real model based on st_config.
    # st_config.embedding_dim is 128 (from fixture).
    # The default model (e.g., "sentence-transformers/all-MiniLM-L6-v2") has a native dimension (384).
    # So, resizing (truncation from 384 to 128) and subsequent normalization will occur.
    service_model = SentenceTransformerEmbeddingModel(config=st_config)

    results = await service_model.get_batch_embedding(texts)

    assert len(results) == len(texts)
    # Ensure all embeddings have the target dimension
    assert all(len(res_emb) == st_config.embedding_dim for res_emb in results)

    # Check if normalization happened.
    # Resizing occurred (native_dim -> st_config.embedding_dim), which triggers normalization.
    # Also, SentenceTransformerModel's encode method itself normalizes if config.normalize is True (default).
    for res_emb in results:
        embedding_array = np.array(
            res_emb
        )  # Ensure it's a NumPy array for norm calculation
        norm = np.linalg.norm(embedding_array)
        assert np.isclose(
            norm, 1.0
        ), f"Embedding norm is {norm}, not 1.0. Embedding: {res_emb}"
