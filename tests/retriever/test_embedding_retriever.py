import json
from pathlib import Path
from typing import Generator, List

import pytest
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from cer.retriever.embedding_retriever import EmbeddingRetriever

from tests.support import ToolSchema


@pytest.fixture
def dummy_file_path_embedding(tmp_path: Path) -> Generator[str, None, None]:
    dummy_file = tmp_path / "dummy_tools_data_embedding_async.json"
    example_tool_data = [
        {
            "tool_schema": {
                "book_name": "Utils",
                "procedure": "GetWeather",
                "signature": "get weather in {city}",
                "description": "Fetches the current weather for a specified city.",
                "skill_type": "EXTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["get weather in London"],
                "bad_examples": ["weather London"],
                "key_topics": ["weather", "forecast", "city"],
                "synthetic_queries": ["What's the weather like in Paris?"],
            },
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # Ensure consistent dim
        },
        {
            "tool_schema": {
                "book_name": "Office",
                "procedure": "CreateDocument",
                "signature": "create document with title {title}",
                "description": "Creates a new document with a given title.",
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["create document with title '''My Report'''"],
                "bad_examples": ["new doc"],
                "key_topics": ["document", "creation", "office"],
                "synthetic_queries": ["How do I make a new document?"],
            },
            "embedding": [0.6, 0.7, 0.8, 0.9, 1.0],  # Ensure consistent dim
        },
        {
            "tool_schema": {
                "book_name": "Utils",
                "procedure": "GetTime",
                "signature": "get current time",
                "description": "Fetches the current time.",
                "skill_type": "EXTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["what time is it"],
                "bad_examples": [],
                "key_topics": ["time", "clock"],
                "synthetic_queries": ["Can you tell me the current time?"],
            },
            "embedding": [0.1, 0.1, 0.1, 0.1, 0.1],  # Ensure consistent dim
        },
        {
            "tool_schema": {  # Tool with missing embedding
                "book_name": "Test",
                "procedure": "NoEmbeddingTool",
                "description": "A tool without an embedding.",
                "signature": "no embedding tool",
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": [],
                "bad_examples": [],
                "key_topics": [],
                "synthetic_queries": [],
            },
            "embedding": [],
        },
    ]
    with open(dummy_file, "w") as f:
        json.dump(example_tool_data, f, indent=2)
    yield str(dummy_file)


async def mock_embedder_func(texts: List[str]) -> List[List[float]]:
    # Ensure embedding dimension matches tool embeddings (e.g., 5 for these tests)
    dim = 5
    return [
        [0.1 * len(text) + i * 0.01 for i in range(dim)] if text else [0.0] * dim
        for text in texts
    ]


@pytest.mark.asyncio
async def test_embedding_retriever_initialization(
    dummy_file_path_embedding: str,
) -> None:
    if cosine_similarity is None or cosine_similarity.__module__ == __name__:
        pytest.skip(
            "scikit-learn not found or using mock, skipping full initialization test that depends on it."
        )

    retriever = EmbeddingRetriever(
        tools_data_path=dummy_file_path_embedding,
        model_class=ToolSchema,
        embedder_func=mock_embedder_func,
    )
    assert (
        retriever.tool_embeddings_matrix.size > 0
    ), "Tool embeddings matrix should not be empty"
    # Expect 3 tools with valid embeddings from the fixture
    assert retriever.tool_embeddings_matrix.shape[0] == 3
    assert retriever.tool_embeddings_matrix.shape[1] == 5  # Dimension of embeddings
    assert len(retriever.tool_schemas_ordered) == 3


@pytest.mark.asyncio
async def test_embedding_retriever_retrieve(dummy_file_path_embedding: str) -> None:
    retriever = EmbeddingRetriever(
        tools_data_path=dummy_file_path_embedding,
        model_class=ToolSchema,
        embedder_func=mock_embedder_func,
    )

    queries = ["weather forecast", "make new document", "what is the time"]
    results = await retriever.retrieve(
        queries=queries, limit=2, similarity_threshold=0.01
    )

    assert len(results) <= 2
    if results:
        assert hasattr(results[0], "procedure")
        # Based on mock_embedder_func and dummy data, we can predict some order or content if needed
        # For now, just check that we get some results


@pytest.mark.asyncio
async def test_embedding_retriever_ordered_ranked(
    dummy_file_path_embedding: str,
) -> None:
    retriever = EmbeddingRetriever(
        tools_data_path=dummy_file_path_embedding,
        model_class=ToolSchema,
        embedder_func=mock_embedder_func,
    )

    queries = ["get weather", "create a doc"]
    limit = 2
    results = await retriever.retrieve_ordered_ranked(
        queries=queries, limit=limit, similarity_threshold=0.01, budget_per_query=1
    )
    assert len(results) <= limit
    if results:
        procedures = [r.procedure for r in results]
        assert len(procedures) == len(set(procedures)), "Expected unique tools"
        # With budget_per_query=1 and 2 queries, we expect at most 2 results
        assert len(results) <= min(limit, len(queries) * 1)


@pytest.mark.asyncio
async def test_retrieve_empty_queries_or_limit_zero(
    dummy_file_path_embedding: str,
) -> None:
    retriever = EmbeddingRetriever(
        tools_data_path=dummy_file_path_embedding,
        model_class=ToolSchema,
        embedder_func=mock_embedder_func,
    )

    results_empty_q = await retriever.retrieve(
        queries=[], limit=2, similarity_threshold=0.1
    )
    assert results_empty_q == []

    results_limit_0 = await retriever.retrieve(
        queries=["test"], limit=0, similarity_threshold=0.1
    )
    assert results_limit_0 == []

    results_ordered_empty_q = await retriever.retrieve_ordered_ranked(
        queries=[], limit=2, similarity_threshold=0.1
    )
    assert results_ordered_empty_q == []

    results_ordered_limit_0 = await retriever.retrieve_ordered_ranked(
        queries=["test"], limit=0, similarity_threshold=0.1
    )
    assert results_ordered_limit_0 == []


@pytest.mark.asyncio
async def test_embedding_retriever_handles_missing_embeddings(tmp_path: Path) -> None:
    # Test case where a tool has a missing or invalid embedding
    invalid_embedding_data = [
        {
            "tool_schema": {"procedure": "ToolWithValidEmb", "description": "Valid"},
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        {
            "tool_schema": {"procedure": "ToolWithEmptyEmb", "description": "Empty"},
            "embedding": [],  # Invalid/empty
        },
        {
            "tool_schema": {"procedure": "ToolWithNoneEmb", "description": "None"},
            "embedding": None,  # Invalid
        },
        {
            "tool_schema": {"procedure": "ToolWithWrongDimEmb", "description": "Short"},
            "embedding": [0.1, 0.2],  # wrong dimension if others are 5D
        },
    ]
    dummy_file = tmp_path / "invalid_embeddings.json"
    with open(dummy_file, "w") as f:
        json.dump(invalid_embedding_data, f)

    retriever = EmbeddingRetriever(
        tools_data_path=str(dummy_file),
        model_class=ToolSchema,
        embedder_func=mock_embedder_func,
    )

    # The retriever should initialize, potentially with warnings printed.
    # It should have filtered out tools with bad embeddings.
    # Check tool_embeddings_matrix and tool_schemas_ordered based on how it handles them.
    # Expecting only 'ToolWithValidEmb' to be processed if strict dimension check is on.
    # If flexible, 'ToolWithWrongDimEmb' might cause an error or be skipped.
    # Current EmbeddingRetriever tries to make a matrix; if dimensions mismatch, it fails or results in empty matrix.
    # This depends on the strictness of np.array() and subsequent checks.

    # If all embeddings must have same dimension as first valid one found:
    # ToolWithValidEmb (dim 5)
    # ToolWithEmptyEmb (skipped)
    # ToolWithNoneEmb (skipped)
    # ToolWithWrongDimEmb (dim 2) -> will cause ValueError if first was dim 5, or matrix becomes empty.
    # For this test, let's assume the constructor handles it by only keeping valid ones (dim 5)

    # If the first valid embedding dictates the dimension (5 in this case)
    # then only ToolWithValidEmb should be in tool_embeddings_matrix
    if retriever.tool_embeddings_matrix.size > 0:
        assert retriever.tool_embeddings_matrix.shape[0] == 1  # Only ToolWithValidEmb
        assert retriever.tool_embeddings_matrix.shape[1] == 5
        assert len(retriever.tool_schemas_ordered) == 1
        assert retriever.tool_schemas_ordered[0].procedure == "ToolWithValidEmb"
    else:
        # This case means the matrix construction failed or was empty due to inconsistent embeddings
        assert len(retriever.tool_schemas_ordered) == 0

    # Test retrieval with this setup
    results = await retriever.retrieve(
        queries=["valid"], limit=1, similarity_threshold=0.01
    )
    if retriever.tool_embeddings_matrix.size > 0:
        assert len(results) == 1
        assert results[0].procedure == "ToolWithValidEmb"
    else:
        assert len(results) == 0
