from logging import Logger
from typing import Any, Awaitable, Callable, List, Optional, Type

from ..cross_encoder.rrf_reranker_client import RRFRerankerClient
from ..utils.schemas import PydanticModel, ToolWithEmbedding
from .client import RetrieverClient, semaphore_gather
from .embedding_retriever import EmbeddingRetriever
from .text_retriever import BM25Retriever


class HybridRetriever(RetrieverClient[PydanticModel]):
    def __init__(
        self,
        tools_data_path: str,
        model_class: Type[PydanticModel],
        bm25_fields: List[str],
        embedder_func: Callable[[List[str]], Awaitable[List[List[float]]]],
        final_reranker_func: Optional[
            Callable[[str, List[PydanticModel]], Awaitable[List[PydanticModel]]]
        ] = None,
        tools_with_embeddings: Optional[List[ToolWithEmbedding[PydanticModel]]] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            tools_data_path=tools_data_path,
            model_class=model_class,
            embedder_func=None,
            reranker_func=None,
            tools_with_embeddings=tools_with_embeddings,
            logger=logger,
        )

        self.bm25_retriever = BM25Retriever(
            tools_data_path=tools_data_path,
            model_class=model_class,
            bm25_fields=bm25_fields,
            embedder_func=None,
            reranker_func=None,
        )
        self.embedding_retriever = EmbeddingRetriever(
            tools_data_path=tools_data_path,
            model_class=model_class,
            embedder_func=embedder_func,
            reranker_func=None,
        )

        self.final_reranker_func = final_reranker_func

    async def retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:
        """
        Retrieves models by combining BM25 and Embedding search results.

        Args:
            queries: List of queries to search for
            limit: Maximum number of results to return

        Keyword Args:
            rrf_k_constant (optional): Constant for RRF re-ranking. Uses 1 by default.
            fuse_tools_by_field (optional): Field to fuse tools by. Must be unique for each tool. Uses "procedure" by default.
            initial_candidates_for_rerank (optional): Number of initial candidates for re-ranking. Uses limit by default.

        Returns:
            List of PydanticModel objects
        """
        if not queries:
            return []

        rrf_k_constant = kwargs.get("rrf_k_constant", 1)
        # Assumes procedure is a unique field for each tool
        fuse_tools_by_field = kwargs.get("fuse_tools_by_field", "procedure")
        initial_candidates_for_rerank = kwargs.get(
            "initial_candidates_for_rerank", limit
        )

        bm25_candidate_tools: List[PydanticModel] = await self.bm25_retriever.retrieve(
            queries, limit=limit, **kwargs
        )

        embedding_candidate_tools: List[
            PydanticModel
        ] = await self.embedding_retriever.retrieve(queries, limit=limit, **kwargs)

        rrf_input_lists: List[List[PydanticModel]] = []
        if bm25_candidate_tools:
            rrf_input_lists.append(bm25_candidate_tools)
        if embedding_candidate_tools:
            rrf_input_lists.append(embedding_candidate_tools)

        fused_tools: List[PydanticModel] = []
        if rrf_input_lists:
            fused_tools = RRFRerankerClient.rrf(
                results=rrf_input_lists,
                field_name=fuse_tools_by_field,
                rank_const=rrf_k_constant,
                min_score=0.0,
            )

        tools_for_final_rerank = fused_tools[:initial_candidates_for_rerank]

        final_results: List[PydanticModel]
        if self.final_reranker_func and tools_for_final_rerank:
            self.logger.info(f"Reranking with {len(tools_for_final_rerank)} tools")
            query_context = " ".join(queries)
            try:
                final_results = await self.final_reranker_func(
                    query_context, tools_for_final_rerank
                )
            except Exception as e:
                self.logger.error(f"Error in reranking: {e}")
                final_results = tools_for_final_rerank
        else:
            self.logger.info(
                f"No reranker provided, returning {len(tools_for_final_rerank)} tools"
            )
            final_results = tools_for_final_rerank

        return final_results[:limit]

    async def retrieve_ordered_ranked(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:
        """
        Retrieves models with ordered processing of queries. For each query,
        it leverages the main `retrieve` method (handling BM25, Embedding, RRF fusion,
        and optional final re-ranking), then applies a budget per query and
        accumulates unique models up to an overall limit.
        The retrieval for individual queries is done concurrently.

        For each query in the input list (maintaining order):
        1. Concurrently calls `self.retrieve` for each query with a specific `limit` (budget_per_query).
           The `retrieve` method handles fetching, RRF fusion, and final re-ranking internally.
        2. Processes the results in the original query order.
        3. Selects models from each query's results, respecting 'budget_per_query'.
        4. Adds these models to the final list, respecting the overall 'limit' and avoiding duplicates.

        Args:
            queries: List of queries to search for
            limit: Maximum number of results to return

        Keyword Args:
            budget_per_query (optional): Number of top candidates per query. Uses limit by default.
            fuse_tools_by_field (optional): Field to fuse tools by. Must be unique for each tool. Uses "procedure" by default.

        Returns:
            List of PydanticModel objects
        """
        if not queries or limit == 0:
            return []

        final_retrieved_tools: List[PydanticModel] = []
        seen_tool_identifiers = set()

        fuse_tools_by_field = kwargs.get("fuse_tools_by_field", "procedure")

        budget_per_query = kwargs.get(
            "budget_per_query", max(1, limit // len(queries) if queries else limit)
        )

        # Execute all retrieve tasks concurrently
        # results_per_query will be a list of lists of PydanticModel, in the same order as queries
        results_per_query: List[List[PydanticModel]] = await semaphore_gather(
            *[
                self.retrieve(queries=[query_content], limit=budget_per_query, **kwargs)
                for query_content in queries
            ]
        )

        # Process results in the original query order
        for tools_from_one_query in results_per_query:
            if len(final_retrieved_tools) >= limit:
                break  # Overall limit reached

            for (
                tool_model
            ) in (
                tools_from_one_query
            ):  # tools_from_one_query is already limited by budget_per_query
                if len(final_retrieved_tools) >= limit:
                    break  # Overall limit reached

                tool_identifier = getattr(tool_model, fuse_tools_by_field, None)
                if tool_identifier is None:  # Fallback
                    tool_identifier = str(tool_model)

                if tool_identifier not in seen_tool_identifiers:
                    final_retrieved_tools.append(tool_model)
                    seen_tool_identifiers.add(tool_identifier)

        return final_retrieved_tools[:limit]

    async def batch_retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[PydanticModel]]:
        if not queries or limit <= 0:
            return []
        try:
            gathered_outcomes = await semaphore_gather(
                *[
                    self.retrieve(
                        queries=[query_text],
                        limit=limit,
                        **kwargs,
                    )
                    for query_text in queries
                ]
            )
            results_per_query: List[List[PydanticModel]] = [
                (outcome if outcome is not None else [])
                for outcome in gathered_outcomes
            ]
            return results_per_query
        except Exception as e:
            self.logger.error(f"Error in batch_retrieve: {e}")
            raise
