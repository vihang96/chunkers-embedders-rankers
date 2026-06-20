from typing import Any, List, Union

import numpy as np
import pytest

# Adjust the import path based on your project structure
from cer.embedder import AbstractEmbeddingModel, EmbedderConfig, normalize_embedding

# --- Tests for normalize_embedding ---


def test_normalize_embedding_single_vector_list() -> None:
    vec = [1.0, 2.0, 3.0]
    norm_vec = normalize_embedding(vec)
    assert isinstance(norm_vec, list)
    assert len(norm_vec) == len(vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)
    # Check original vector is not modified if it's a list
    assert vec == [1.0, 2.0, 3.0]


def test_normalize_embedding_single_vector_numpy() -> None:
    vec = np.array([1.0, 2.0, 3.0])
    norm_vec = normalize_embedding(vec)
    assert isinstance(norm_vec, np.ndarray)
    assert len(norm_vec) == len(vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0)


def test_normalize_embedding_batch_vector_list_of_lists() -> None:
    batch_vec = [[1.0, 2.0], [3.0, 4.0]]
    norm_batch = normalize_embedding(batch_vec)
    assert isinstance(norm_batch, list)
    assert all(isinstance(v, list) for v in norm_batch)
    assert len(norm_batch) == len(batch_vec)
    for v_norm in norm_batch:
        assert np.isclose(np.linalg.norm(v_norm), 1.0)


def test_normalize_embedding_batch_vector_numpy() -> None:
    batch_vec = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_batch = normalize_embedding(batch_vec)
    assert isinstance(norm_batch, np.ndarray)
    assert norm_batch.shape == batch_vec.shape
    for v_norm in norm_batch:
        assert np.isclose(np.linalg.norm(v_norm), 1.0)


def test_normalize_embedding_zero_vector_single_list() -> None:
    vec = [0.0, 0.0, 0.0]
    with pytest.warns(UserWarning, match="Attempted to normalize a zero vector."):
        norm_vec = normalize_embedding(vec)
    assert norm_vec == vec  # Should return original zero vector


def test_normalize_embedding_zero_vector_single_numpy() -> None:
    vec = np.array([0.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="Attempted to normalize a zero vector."):
        norm_vec = normalize_embedding(vec)
    assert np.array_equal(norm_vec, vec)


def test_normalize_embedding_batch_with_zero_vector_numpy() -> None:
    batch_vec = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
    # No warning expected here from the function itself, as np.where handles it
    norm_batch = normalize_embedding(batch_vec)
    assert isinstance(norm_batch, np.ndarray)
    assert np.isclose(np.linalg.norm(norm_batch[0]), 1.0)
    assert np.array_equal(
        norm_batch[1], np.array([0.0, 0.0])
    )  # Zero vector remains zero
    assert np.isclose(np.linalg.norm(norm_batch[2]), 1.0)


def test_normalize_embedding_invalid_input_shape() -> None:
    with pytest.raises(
        ValueError, match="Input must be a 1D or 2D array/list of lists."
    ):
        normalize_embedding(np.array([[[1.0]]]))  # 3D array

    with pytest.raises(
        ValueError, match="Input must be a 1D or 2D array/list of lists."
    ):
        normalize_embedding(np.array([[[1.0]]]))  # list of list of list


# --- Basic test for EmbedderConfig (Pydantic model) ---
def test_embedder_config() -> None:
    config = EmbedderConfig(embedding_model="test_model")
    assert config.embedding_dim == 1024  # Default value
    config_custom = EmbedderConfig(embedding_dim=512, embedding_model="test_model")
    assert config_custom.embedding_dim == 512


# --- Basic test for AbstractEmbeddingModel (Abstract class) ---
# We can't instantiate it directly, but we can define a minimal concrete class for testing purposes.
class MinimalConcreteModel(AbstractEmbeddingModel):
    def __init__(self, config: EmbedderConfig, client: Any = None) -> None:
        self.config = config
        self.client = client

    async def get_embedding(self, input_data: Union[str, List[str]]) -> List[float]:
        return [1.0, 2.0]

    async def get_batch_embedding(
        self, input_data_list: Union[str, List[str]]
    ) -> List[List[float]]:
        return [[1.0, 2.0]]


def test_abstract_embedding_model_instantiation() -> None:
    config = EmbedderConfig(embedding_model="test_model")
    # Test that a concrete implementation can be instantiated
    model = MinimalConcreteModel(config=config)
    assert model.config == config

    # Example with a dummy client object
    dummy_client = object()
    model_with_client = MinimalConcreteModel(config=config, client=dummy_client)
    assert model_with_client.client == dummy_client
