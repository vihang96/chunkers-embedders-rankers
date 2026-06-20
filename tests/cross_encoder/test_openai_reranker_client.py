import os

import pytest

from cer.cross_encoder.openai_reranker_client import (
    LLMConfig,
    OpenAIRerankerClient,
    RateLimitError,
)

from tests.support import get_openai_key


@pytest.mark.asyncio
async def test_openai_reranker_client_integration() -> None:
    """
    Integration test for OpenAIRerankerClient.
    This test makes actual calls to the OpenAI API.
    Ensure OPENAI_API_KEY is set in the environment.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    api_key = await get_openai_key()
    if not api_key:
        pytest.skip(
            "OPENAI_API_KEY not found in environment. Skipping integration test."
        )

    config = LLMConfig(api_key=api_key)  # Uses default model and other settings
    client = OpenAIRerankerClient(config=config)

    query = "What are the primary benefits of using renewable energy sources?"

    passage_relevant = "Renewable energy sources like solar and wind power help reduce greenhouse gas emissions, diversify energy supply, and reduce dependence on fossil fuels, leading to a more sustainable environment."
    passage_somewhat_relevant = "Geothermal energy is a type of renewable energy that harnesses heat from within the Earth."
    passage_irrelevant = "The stock market experienced significant volatility last week due to new economic forecasts."

    passages = [
        passage_irrelevant,  # Intentionally putting irrelevant first
        passage_relevant,
        passage_somewhat_relevant,
    ]

    try:
        ranked_results = await client.rank(query, passages)
    except RateLimitError:
        pytest.skip("OpenAI API rate limit exceeded. Skipping test.")
        return
    except Exception as e:
        pytest.fail(f"OpenAIRerankerClient().rank() raised an exception: {e}")
        return

    assert len(ranked_results) == len(passages), (
        "Number of ranked results should match input passages"
    )

    found_relevant = False
    found_irrelevant = False
    score_relevant = -1.0
    score_irrelevant = -1.0

    for i, (passage_text, score) in enumerate(ranked_results):
        assert isinstance(passage_text, str), (
            f"Passage text at index {i} is not a string."
        )
        assert isinstance(score, float), f"Score at index {i} is not a float."
        assert 0.0 <= score <= 1.0, (
            f"Score at index {i} ({score}) is not between 0 and 1."
        )

        if passage_text == passage_relevant:
            found_relevant = True
            score_relevant = score
        elif passage_text == passage_irrelevant:
            found_irrelevant = True
            score_irrelevant = score

        # Check that the original passage objects are returned (not modified)
        assert passage_text in passages, (
            f"Ranked passage '{passage_text}' not in original passages."
        )

    assert found_relevant, (
        f"The highly relevant passage was not found in the results: '{passage_relevant}'"
    )
    assert found_irrelevant, (
        f"The clearly irrelevant passage was not found in the results: '{passage_irrelevant}'"
    )

    # The core assertion: relevant passage should have a higher score than irrelevant one.
    # We also expect the relevant passage to be ranked higher (appear earlier in the sorted list).
    # And the somewhat_relevant passage to be somewhere in between or lower than relevant.

    # Get original indices for comparison
    ranked_texts_only = [res[0] for res in ranked_results]

    try:
        index_relevant = ranked_texts_only.index(passage_relevant)
    except ValueError:
        pytest.fail(f"Relevant passage not found in ranked results: {passage_relevant}")

    try:
        index_irrelevant = ranked_texts_only.index(passage_irrelevant)
    except ValueError:
        pytest.fail(
            f"Irrelevant passage not found in ranked results: {passage_irrelevant}"
        )

    assert score_relevant > score_irrelevant, (
        f"Relevant passage score ({score_relevant}) should be greater than irrelevant passage score ({score_irrelevant})."
    )

    assert index_relevant < index_irrelevant, (
        f"Relevant passage (ranked at {index_relevant}) should appear before irrelevant passage (ranked at {index_irrelevant})."
    )

    # Check if somewhat_relevant passage is also present
    found_somewhat_relevant = any(
        p_text == passage_somewhat_relevant for p_text, _ in ranked_results
    )
    assert found_somewhat_relevant, (
        f"The somewhat relevant passage was not found in the results: '{passage_somewhat_relevant}'"
    )

    # Optional: Check if somewhat_relevant is ranked between relevant and irrelevant, or at least below relevant
    # This can be flaky depending on the model's interpretation.
    # score_somewhat_relevant = next(s for p, s in ranked_results if p == passage_somewhat_relevant)
    # index_somewhat_relevant = ranked_texts_only.index(passage_somewhat_relevant)
    # assert score_relevant >= score_somewhat_relevant, \
    #    f"Relevant passage score ({score_relevant}) should be >= somewhat_relevant score ({score_somewhat_relevant})."
    # assert index_relevant <= index_somewhat_relevant, \
    #    f"Relevant passage (idx {index_relevant}) should be ranked higher or same as somewhat_relevant (idx {index_somewhat_relevant})."

    print(f"Query: {query}")
    print("Ranked results:")
    for passage, score in ranked_results:
        print(f"  Score: {score:.4f} - Passage: {passage[:100]}...")
