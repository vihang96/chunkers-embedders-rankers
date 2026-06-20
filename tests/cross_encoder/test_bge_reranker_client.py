from typing import List

import pytest

from cer.cross_encoder.bge_reranker_client import BGERerankerClient


@pytest.fixture
def client() -> BGERerankerClient:
    return BGERerankerClient()


@pytest.mark.asyncio
async def test_bge_reranker_rank_empty_passages(client: BGERerankerClient) -> None:
    """Test the rank method with an empty list of passages."""
    query = "test query"
    passages: List[str] = []
    result = await client.rank(query, passages)
    assert result == []


@pytest.mark.asyncio
async def test_bge_reranker_rank_with_passages(client: BGERerankerClient) -> None:
    """Test the rank method with a list of passages."""
    query = "test query"
    passages = ["passage1", "passage2", "passage3"]

    expected_ranked_passages = [
        ("passage3", 0.00014),
        ("passage1", 0.00013),
        ("passage2", 0.00012),
    ]

    result = await client.rank(query, passages)
    assert [res[0] for res in result] == [res[0] for res in expected_ranked_passages]


@pytest.mark.asyncio
async def test_rank_basic_functionality(client: BGERerankerClient) -> None:
    query = "What is the capital of France?"
    passages = [
        "Paris is the capital and most populous city of France.",
        "London is the capital city of England and the United Kingdom.",
        "Berlin is the capital and largest city of Germany.",
    ]

    ranked_passages = await client.rank(query, passages)

    # Check if the output is a list of tuples
    assert isinstance(ranked_passages, list)
    assert all(isinstance(item, tuple) for item in ranked_passages)

    # Check if the output has the correct length
    assert len(ranked_passages) == len(passages)

    # Check if the scores are floats and passages are strings
    for passage, score in ranked_passages:
        assert isinstance(passage, str)
        assert isinstance(score, float)

    # Check if the results are sorted in descending order
    scores = [score for _, score in ranked_passages]
    assert scores == sorted(scores, reverse=True)
