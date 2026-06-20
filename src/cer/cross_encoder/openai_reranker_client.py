import asyncio
import logging
import os
from collections.abc import Coroutine
from logging import Logger
from typing import Any, Optional, TypedDict

import numpy as np
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .client import CrossEncoderClient

USE_PARALLEL_RUNTIME = bool(os.getenv("USE_PARALLEL_RUNTIME", False))
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 20))

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0

DEFAULT_MODEL = "gpt-5-nano"


class RateLimitError(Exception):
    """Exception raised when the rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please try again later."):
        self.message = message
        super().__init__(self.message)


class Message(TypedDict):
    role: str
    content: str


class LLMConfig:
    """
    Configuration class for the Language Learning Model (LLM).

    This class encapsulates the necessary parameters to interact with an LLM API,
    such as OpenAI's GPT models. It stores the API key, model name, and base URL
    for making requests to the LLM service.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        small_model: str | None = None,
    ):
        """
        Initialize the LLMConfig with the provided parameters.

        Args:
                api_key (str): The authentication key for accessing the LLM API.
                                                This is required for making authorized requests.

                model (str, optional): The specific LLM model to use for generating responses.
                                                                Defaults to "gpt-5-mini".

                base_url (str, optional): The base URL of the LLM API service.
                                                                        Defaults to "https://api.openai.com", which is OpenAI's standard API endpoint.
                                                                        This can be changed if using a different provider or a custom endpoint.

                small_model (str, optional): The specific LLM model to use for generating responses of simpler prompts.
                                                                Defaults to "gpt-5-nano".
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.small_model = small_model
        self.temperature = temperature
        self.max_tokens = max_tokens


# Use this instead of asyncio.gather() to bound coroutines
async def semaphore_gather(  # type: ignore
    *coroutines: Coroutine,  # type: ignore
    max_coroutines: int = SEMAPHORE_LIMIT,
):
    semaphore = asyncio.Semaphore(max_coroutines)

    async def _wrap(coro: Coroutine) -> Any:  # type: ignore
        async with semaphore:
            return await coro

    results = []
    batch = []
    for coroutine in coroutines:
        batch.append(_wrap(coroutine))
        # once we hit max_coroutines, gather and clear the batch
        if len(batch) >= max_coroutines:
            results.extend(await asyncio.gather(*batch))
            batch.clear()

    # gather any remaining coroutines in the final batch
    if batch:
        results.extend(await asyncio.gather(*batch))

    return results


class OpenAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: LLMConfig | None = None,
        client: AsyncOpenAI | AsyncAzureOpenAI | None = None,
    ):
        """
        Initialize the OpenAIRerankerClient with the provided configuration and client.

        This reranker uses the OpenAI API to run a simple boolean classifier prompt concurrently
        for each passage. Log-probabilities are used to rank the passages.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            client (AsyncOpenAI | AsyncAzureOpenAI | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
        """
        if config is None:
            config = LLMConfig()

        self.config = config
        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

        self.logger = logging.getLogger(__name__)

    async def rank(
        self, query: str, passages: list[str], logger: Optional[Logger] = None
    ) -> list[tuple[str, float]]:
        log = logger or self.logger
        log.info(f"Ranking {len(passages)} passages for query: {query}")
        openai_messages_list: Any = [
            [
                Message(
                    role="system",
                    content="You are an expert tasked with determining whether the passage is relevant to the query",
                ),
                Message(
                    role="user",
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]
        try:
            responses = await semaphore_gather(
                *[
                    self.client.chat.completions.create(
                        model=DEFAULT_MODEL,
                        messages=openai_messages,
                        temperature=0,
                        max_tokens=1,
                        logit_bias={"6432": 1, "7983": 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    for openai_messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                (
                    response.choices[0].logprobs.content[0].top_logprobs
                    if response.choices[0].logprobs is not None
                    and response.choices[0].logprobs.content is not None
                    else []
                )
                for response in responses
            ]
            results_with_scores: list[tuple[str, float]] = []
            for passage, top_logprobs in zip(passages, responses_top_logprobs):
                if len(top_logprobs) == 0:
                    continue
                norm_logprobs = np.exp(top_logprobs[0].logprob)
                score: float
                token_val = top_logprobs[0].token
                is_true_representation = (
                    isinstance(token_val, str) and token_val.strip().lower() == "true"
                ) or (isinstance(token_val, bool) and token_val is True)

                if is_true_representation:
                    # Token represents 'True' (relevance)
                    score = norm_logprobs
                else:
                    # Token represents 'False' (or is not explicitly 'True'), indicating non-relevance
                    # Score is 1 - P(token), which approximates P('True') if P('True') + P('False') ~ 1
                    score = 1 - norm_logprobs
                results_with_scores.append((passage, score))

            results_with_scores.sort(reverse=True, key=lambda x: x[1])
            return results_with_scores
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            log.error(f"Error in generating LLM response: {e}")
            raise
