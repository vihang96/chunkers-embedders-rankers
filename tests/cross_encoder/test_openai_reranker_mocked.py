from collections.abc import Coroutine
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import openai
import pytest

from cer.cross_encoder.openai_reranker_client import (
    DEFAULT_MODEL,
    SEMAPHORE_LIMIT,
    LLMConfig,
    OpenAIRerankerClient,
    RateLimitError,
)


# Mock structures that mimic OpenAI API responses
class MockLogprobContent:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


class MockLogprobs:
    def __init__(self, top_logprobs_data: list[tuple[str, float]]):
        self.content = [MagicMock()]  # Mock the first element of content list
        self.content[0].top_logprobs = [
            MockLogprobContent(t, lp) for t, lp in top_logprobs_data
        ]


class MockChoice:
    def __init__(self, top_logprobs_data: list[tuple[str, float]]):
        self.logprobs = MockLogprobs(top_logprobs_data) if top_logprobs_data else None


class MockCompletionResponse:
    def __init__(self, top_logprobs_data: list[tuple[str, float]]):
        self.choices = [MockChoice(top_logprobs_data)]


@pytest.fixture
def mock_async_openai_client() -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(api_key="test_key", model="test_model")


@pytest.mark.asyncio
async def test_openai_reranker_client_init_with_config(llm_config: LLMConfig) -> None:
    """Test client initialization with LLMConfig."""
    with patch(
        "cer.cross_encoder.openai_reranker_client.AsyncOpenAI"
    ) as mock_async_openai_constructor:
        mock_instance = AsyncMock()
        mock_async_openai_constructor.return_value = mock_instance

        client = OpenAIRerankerClient(config=llm_config)

        mock_async_openai_constructor.assert_called_once_with(
            api_key=llm_config.api_key, base_url=llm_config.base_url
        )
        assert client.client == mock_instance
        assert client.config == llm_config


@pytest.mark.asyncio
async def test_openai_reranker_client_init_with_existing_client(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test client initialization with an existing client instance."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    assert client.client == mock_async_openai_client
    assert client.config == llm_config
    mock_async_openai_client.chat.completions.create.assert_not_called()  # Ensure no API calls on init


@pytest.mark.asyncio
async def test_rank_empty_passages(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test rank method with empty passages list."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    query = "test query"
    passages: list[str] = []
    result = await client.rank(query, passages)
    assert result == []
    mock_async_openai_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_rank_success(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test successful ranking of passages."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    query = "test query"
    passages = ["passage true", "passage false", "passage also true"]

    # Mock responses from OpenAI API
    # Higher logprob for 'True' means more relevant
    # Passage 1: True (score: exp(-0.1))
    # Passage 2: False (score: exp(-0.2)) approx 0.818
    # Passage 3: True (score: exp(-0.05))
    # logit_bias={'6432': 1, '7983': 1} -> True, False
    # Let's assume 'True' is token '6432' and 'False' is '7983' for mock purposes
    # Mocking based on how scores are calculated:
    # if top_logprobs[0].token == 'True' (or any truthy value): score = np.exp(top_logprobs[0].logprob)
    # else: score = 1 - np.exp(top_logprobs[0].logprob)
    # So, for a high score (relevant), we want token='True' and high logprob (closer to 0)
    # For a low score (irrelevant), we want token='False' and high logprob (closer to 0), leading to 1 - high_val = low_score

    mock_responses = [
        MockCompletionResponse(
            [("True", -0.1)]
        ),  # High relevance, score = exp(-0.1) approx 0.904
        MockCompletionResponse(
            [("False", -0.2)]
        ),  # Low relevance, score = exp(-0.2) approx 0.818
        MockCompletionResponse(
            [("True", -0.05)]
        ),  # Highest relevance, score = exp(-0.05) approx 0.951
    ]
    mock_async_openai_client.chat.completions.create.side_effect = mock_responses

    expected_ranking = [
        ("passage also true", np.exp(-0.05)),
        ("passage true", np.exp(-0.1)),
        ("passage false", 1 - np.exp(-0.2)),
    ]
    # Sort expected by score to match the implementation's sort
    expected_ranking.sort(key=lambda x: x[1], reverse=True)

    # Patch semaphore_gather to run serially for simpler testing of create calls
    with patch(
        "cer.cross_encoder.openai_reranker_client.semaphore_gather",
        new_callable=AsyncMock,
    ) as mock_sem_gather:
        # This makes semaphore_gather immediately await and return results of the coroutines
        async def serial_gather(  # type: ignore
            *coros: Coroutine, max_coroutines: int = SEMAPHORE_LIMIT  # type: ignore
        ):
            results = []
            for coro in coros:
                results.append(await coro)
            return results

        mock_sem_gather.side_effect = serial_gather

        actual_ranking = await client.rank(query, passages)

    assert mock_async_openai_client.chat.completions.create.call_count == len(passages)
    # Check one of the calls to ensure correct parameters
    first_call_args = mock_async_openai_client.chat.completions.create.call_args_list[
        0
    ][1]
    assert first_call_args["model"] == DEFAULT_MODEL
    assert first_call_args["temperature"] == 0
    assert first_call_args["max_tokens"] == 1
    assert first_call_args["logit_bias"] == {"6432": 1, "7983": 1}
    assert first_call_args["logprobs"] is True
    assert first_call_args["top_logprobs"] == 2

    assert len(actual_ranking) == len(expected_ranking)
    for actual, expected in zip(actual_ranking, expected_ranking):
        assert actual[0] == expected[0]
        assert actual[1] == pytest.approx(expected[1])


@pytest.mark.asyncio
async def test_rank_rate_limit_error(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test that RateLimitError is raised when OpenAI API signals it."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    query = "test query"
    passages = ["passage1"]
    mock_async_openai_client.chat.completions.create.side_effect = (
        openai.RateLimitError(
            message="Rate limit exceeded", response=MagicMock(), body=None
        )
    )

    with pytest.raises(RateLimitError, match="Rate limit exceeded"):
        await client.rank(query, passages)


@pytest.mark.asyncio
async def test_rank_general_exception(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test that general exceptions from OpenAI API are caught and re-raised."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    query = "test query"
    passages = ["passage1"]
    mock_async_openai_client.chat.completions.create.side_effect = Exception(
        "Some API error"
    )

    with pytest.raises(Exception, match="Some API error"):
        await client.rank(query, passages)


@pytest.mark.asyncio
async def test_rank_response_missing_logprobs(
    mock_async_openai_client: AsyncMock, llm_config: LLMConfig
) -> None:
    """Test handling of responses where logprobs might be missing or incomplete."""
    client = OpenAIRerankerClient(config=llm_config, client=mock_async_openai_client)
    query = "test query"
    passages = ["passage1", "passage2"]

    # Response 1: Logprobs completely missing
    # Response 2: Logprobs.content is None
    mock_response1 = MagicMock()
    mock_response1.choices = [MagicMock()]
    mock_response1.choices[0].logprobs = None  # No logprobs attribute

    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock()]
    mock_response2.choices[0].logprobs = MagicMock()
    mock_response2.choices[0].logprobs.content = None  # Logprobs content is None

    mock_async_openai_client.chat.completions.create.side_effect = [
        mock_response1,
        mock_response2,
    ]

    # Patch semaphore_gather to run serially for simpler testing of create calls
    with patch(
        "cer.cross_encoder.openai_reranker_client.semaphore_gather",
        new_callable=AsyncMock,
    ) as mock_sem_gather:

        async def serial_gather(  # type: ignore
            *coros: Coroutine, max_coroutines: int = SEMAPHORE_LIMIT  # type: ignore
        ):
            results = []
            for coro in coros:
                results.append(await coro)
            return results

        mock_sem_gather.side_effect = serial_gather
        actual_ranking = await client.rank(query, passages)

    # If logprobs are missing, the passage is effectively skipped for scoring
    # The current code continues if len(top_logprobs) == 0, which happens if content[0].top_logprobs is empty
    # For these mocks, responses_top_logprobs will be [[], []]
    # Then scores list will be empty.
    # zip(passages, scores, strict=True) will then raise ValueError if scores is shorter than passages.
    # This needs to be handled: either ensure scores has a default or passages are filtered.
    # Based on current code: `if len(top_logprobs) == 0: continue` means scores list can be shorter.
    # `zip(passages, scores, strict=True)` will then fail if strict=True and lists are different lengths.
    # Let's assume the intention is to filter out passages with no valid score.
    # For this test, if scores is empty, results should be empty.
    assert actual_ranking == []
