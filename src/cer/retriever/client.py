import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from logging import Logger
from typing import Any, Awaitable, Callable, Generic, List, Optional, Type

import yaml

from ..utils.schemas import PydanticModel, ToolWithEmbedding

SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", 20))

log = logging.getLogger(__name__)


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


class RetrieverClient(ABC, Generic[PydanticModel]):
    def __init__(
        self,
        tools_data_path: str,  # Path to JSON file containing List[ToolWithEmbedding[PydanticModel]]
        model_class: Type[PydanticModel],
        embedder_func: Optional[
            Callable[[List[str]], Awaitable[List[List[float]]]]
        ] = None,
        reranker_func: Optional[
            Callable[[str, List[PydanticModel]], Awaitable[List[PydanticModel]]]
        ] = None,
        tools_with_embeddings: Optional[List[ToolWithEmbedding[PydanticModel]]] = None,
        logger: Optional[Logger] = None,
    ):
        self.model_class = model_class
        self.embedder_func = embedder_func
        self.logger = logger or log
        self.tools_with_embeddings = (
            self._load_tools_data(tools_data_path)
            if tools_with_embeddings is None
            else tools_with_embeddings
        )
        self.tools: List[PydanticModel] = [
            item.tool_schema for item in self.tools_with_embeddings
        ]
        self.reranker_func = reranker_func

    def _load_tools_data(
        self, file_path: str
    ) -> List[ToolWithEmbedding[PydanticModel]]:
        """Loads tools and their pre-computed embeddings from a JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return [ToolWithEmbedding[self.model_class](**item) for item in data]  # type: ignore
        except FileNotFoundError:
            # Consider raising an error or logging more formally
            self.logger.error(f"Error: The file {file_path} was not found.")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error: Could not decode JSON from {file_path}.")
            return []
        except Exception as e:  # Consider more specific exception handling
            self.logger.error(
                f"An unexpected error occurred while loading {file_path}: {e}"
            )
            return []

    def _get_text_for_bm25(self, tool: PydanticModel, fields: List[str]) -> str:
        """
        Constructs a YAML string representation of the specified fields of a Pydantic model.
        """
        tool_data_for_yaml = {}
        for field in fields:
            if hasattr(tool, field):
                value = getattr(tool, field)
                tool_data_for_yaml[field] = value
            else:
                # Accessing a non-existent field is usually an error.
                # For ToolSchema, 'procedure' might have been assumed to exist.
                # For a generic PydanticModel, we need a robust way to get a name if available.
                tool_name_for_warning = getattr(
                    tool, "name", getattr(tool, "procedure", "Unnamed PydanticModel")
                )
                self.logger.warning(
                    f"Warning: Field '{field}' not found in model '{tool_name_for_warning}' for BM25 YAML generation."
                )

        try:
            yaml_string = yaml.dump(
                tool_data_for_yaml, sort_keys=False, default_flow_style=False
            )
        except Exception as e:
            tool_name_for_error = getattr(
                tool, "name", getattr(tool, "procedure", "Unnamed PydanticModel")
            )
            self.logger.error(
                f"Error during YAML dump for model '{tool_name_for_error}': {e}"
            )
            return ""
        return yaml_string

    @abstractmethod
    async def retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:
        """
        Retrieves semantically similar models based on the user query.
        Each subclass must implement its own retrieval logic.

        Args:
            queries: A single query or list of queries (splits of the same query).
            limit: Number of models to retrieve.
            **kwargs: Additional retriever-specific arguments.

        Returns:
            A list of relevant PydanticModel objects.
        """
        pass

    @abstractmethod
    async def retrieve_ordered_ranked(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:
        """
        Retrieves semantically similar models based on the user query, with ordered processing.
        Each subclass must implement its own retrieval logic.

        Args:
            queries: A single query or list of queries (splits of the same query).
            limit: Number of models to retrieve.
            **kwargs: Additional retriever-specific arguments.

        Returns:
            A list of relevant PydanticModel objects.
        """
        pass

    @abstractmethod
    async def batch_retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[PydanticModel]]:
        pass
