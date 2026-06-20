from logging import Logger
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Type

import bm25s
import numpy as np

from ..cross_encoder.rrf_reranker_client import RRFRerankerClient
from ..utils.schemas import PydanticModel, ToolWithEmbedding
from .client import RetrieverClient, semaphore_gather


class BM25Retriever(RetrieverClient[PydanticModel]):  # Made generic
    def __init__(
        self,
        tools_data_path: str,
        model_class: Type[PydanticModel],  # Added model_class
        bm25_fields: List[str],  # Fields from PydanticModel to use
        embedder_func: Optional[
            Callable[[List[str]], Awaitable[List[List[float]]]]
        ] = None,
        reranker_func: Optional[
            Callable[
                [str, List[PydanticModel]], Awaitable[List[PydanticModel]]
            ]  # Uses PydanticModel
        ] = None,
        tools_with_embeddings: Optional[List[ToolWithEmbedding[PydanticModel]]] = None,
        logger: Optional[Logger] = None,
    ):
        # Pass model_class and reranker_func to super
        super().__init__(
            tools_data_path=tools_data_path,
            model_class=model_class,
            embedder_func=embedder_func,
            reranker_func=reranker_func,
            tools_with_embeddings=tools_with_embeddings,
            logger=logger,
        )
        if not bm25_fields:
            raise ValueError(
                "BM25Retriever requires 'bm25_fields' to know which model fields to index."
            )
        self.bm25_fields = bm25_fields
        self.bm25_model: Optional[bm25s.BM25] = None
        # self.reranker_func is set by super()
        self.raw_corpus_texts: List[str] = []
        self._initialize_bm25()

    def _initialize_bm25(self) -> None:
        """Initializes the BM25 model with the corpus built from specified model fields."""
        if not self.tools:  # self.tools is List[PydanticModel]
            self.logger.warning("Warning: No models loaded, BM25 index will be empty.")
            self.raw_corpus_texts = []
            return

        # self._get_text_for_bm25 in RetrieverClient now expects PydanticModel
        self.raw_corpus_texts = [
            self._get_text_for_bm25(tool, self.bm25_fields) for tool in self.tools
        ]

        if not self.raw_corpus_texts:
            self.logger.warning(
                "Warning: BM25 corpus is empty after processing models."
            )
            return

        try:
            tokenized_corpus = bm25s.tokenize(self.raw_corpus_texts)
            self.bm25_model = bm25s.BM25()
            self.bm25_model.index(tokenized_corpus)
        except Exception as e:
            self.logger.error(f"Error initializing BM25S model or indexing corpus: {e}")
            self.bm25_model = None

    def _get_scored_bm25_candidates_for_query_sync(
        self,
        query_text: str,
        initial_candidates_limit: int,
    ) -> List[Tuple[float, PydanticModel]]:
        """
        Synchronously retrieves and scores candidates for a single query using BM25.
        Returns a list of (score, model) tuples, sorted by score, up to the limit.
        """
        if not self.bm25_model or not self.tools or not self.raw_corpus_texts:
            return []

        tokenized_query_terms = bm25s.tokenize(query_text)
        # Retrieve all initially to get full score spectrum before Python-side sorting/limiting
        doc_indices_matrix, scores_matrix = self.bm25_model.retrieve(
            tokenized_query_terms, k=len(self.raw_corpus_texts)
        )

        current_doc_scores = np.full(len(self.tools), -np.inf)
        if doc_indices_matrix.size > 0 and scores_matrix.size > 0:
            retrieved_indices_for_query = doc_indices_matrix[0]
            scores_for_query = scores_matrix[0]
            for i in range(len(retrieved_indices_for_query)):
                original_doc_idx = retrieved_indices_for_query[i]
                score = scores_for_query[i]
                if original_doc_idx < len(current_doc_scores):  # Boundary check
                    current_doc_scores[original_doc_idx] = score

        scored_tools_for_query: List[Tuple[float, PydanticModel]] = []
        for i, score in enumerate(current_doc_scores):
            if score > -np.inf and i < len(self.tools):  # Ensure index is valid
                scored_tools_for_query.append((score, self.tools[i]))

        scored_tools_for_query.sort(key=lambda x: x[0], reverse=True)
        return scored_tools_for_query[:initial_candidates_limit]

    async def retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:  # Return type is List[PydanticModel]
        """
        Retrieves models using BM25 ranking based on textual similarity.
        Queries are processed concurrently for BM25 scoring.
        Assumes queries are chunks for the same task
        and returns a single list of PydanticModel objects.

        Args:
            queries: List of queries to search for
            limit: Maximum number of results to return

        Keyword Args:
            rrf_k_constant (optional): Constant for RRF re-ranking. Uses 1 by default.
            initial_candidates_for_rerank (optional): Number of initial candidates for re-ranking. Uses limit by default.
            fuse_tools_by_field (optional): Field to fuse tools by. Must be unique for each tool. Uses "procedure" by default.

        Returns:
            List of PydanticModel objects
        """
        if not queries:
            return []

        if not self.bm25_model or not self.tools or not self.raw_corpus_texts:
            return []

        rrf_k_constant = kwargs.get("rrf_k_constant", 1)
        initial_candidates_for_rerank = kwargs.get(
            "initial_candidates_for_rerank", limit
        )
        id_field = kwargs.get("fuse_tools_by_field", "procedure")

        list_of_scored_tools_per_query: List[List[Tuple[float, PydanticModel]]] = [
            self._get_scored_bm25_candidates_for_query_sync(
                query_text,
                initial_candidates_for_rerank,
            )
            for query_text in queries
        ]

        rrf_input_lists: List[List[PydanticModel]] = [
            [tool for _, tool in scored_tools_for_one_query]
            for scored_tools_for_one_query in list_of_scored_tools_per_query
        ]

        fused_tools: List[PydanticModel] = []
        if rrf_input_lists:
            fused_tools = RRFRerankerClient.rrf(
                results=rrf_input_lists,
                field_name=id_field,
                rank_const=rrf_k_constant,
                min_score=0.0,
            )

        tools_for_reranking = fused_tools[:initial_candidates_for_rerank]
        final_tools: List[PydanticModel]
        if self.reranker_func and tools_for_reranking:
            self.logger.info(f"Reranking with {len(tools_for_reranking)} tools")
            query_context = " ".join(queries) if isinstance(queries, list) else queries
            try:
                reranked_tools_list = await self.reranker_func(
                    query_context, tools_for_reranking
                )
            except Exception as e:
                self.logger.error(f"Error in reranking: {e}")
                reranked_tools_list = tools_for_reranking
            final_tools = reranked_tools_list[:limit]
        else:
            self.logger.info(
                f"No reranker provided, returning {len(tools_for_reranking)} tools"
            )
            final_tools = tools_for_reranking[:limit]
        return final_tools

    async def _process_query_for_ordered_retrieval_async(
        self,
        query_text: str,
        initial_candidates_for_rerank_per_query: int,
        budget_per_query: int,
    ) -> List[PydanticModel] | None:
        """
        Asynchronously processes a single query for ordered retrieval:
        1. Fetches initial candidates using BM25 (sync, offloaded).
        2. Reranks them if a reranker is provided.
        3. Returns tools up to the budget for this query.

        Args:
            query_text: Query to search for
            initial_candidates_for_rerank_per_query: Number of initial candidates for re-ranking
            budget_per_query: Number of top candidates per query

        Returns:
            List of PydanticModel objects
        """
        try:
            scored_candidates: List[Tuple[float, PydanticModel]] = (
                self._get_scored_bm25_candidates_for_query_sync(
                    query_text,
                    initial_candidates_for_rerank_per_query,
                )
            )

            tools_to_rerank_for_query: List[PydanticModel] = [
                tool for _, tool in scored_candidates
            ]

            if not tools_to_rerank_for_query:
                return []

            reranked_tools_for_query: List[PydanticModel]
            if self.reranker_func:
                reranked_tools_for_query = await self.reranker_func(
                    query_text, tools_to_rerank_for_query
                )
            else:
                reranked_tools_for_query = tools_to_rerank_for_query

            return reranked_tools_for_query[:budget_per_query]
        except Exception as e:
            self.logger.warning(
                f"Warning: Task for query '{query_text}' failed with {type(e).__name__}: {e}"
            )
            return None

    async def retrieve_ordered_ranked(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[PydanticModel]:  # Return type is List[PydanticModel]
        """
        Retrieves models using BM25 ranking, with ordered processing of queries,
        re-ranking, and a budget per query. BM25 and reranking for each query
        are processed concurrently.
        Args:
            queries: List of query strings (sub-queries in order).
            limit: Maximum total number of models to retrieve.

        Keyword Args:
            initial_candidates_for_rerank (optional): Number of initial candidates for re-ranking. Uses limit by default.
            budget_per_query (optional): Number of top candidates per query. Uses limit by default.
            fuse_tools_by_field (optional): Field to fuse tools by. Must be unique for each tool. Uses "procedure" by default.

        Returns:
            List of PydanticModel objects, ordered by query priority and re-ranked scores.
        """
        if not queries or limit <= 0:
            return []

        if not self.bm25_model or not self.tools or not self.raw_corpus_texts:
            return []

        default_budget_per_query = (limit // len(queries)) if queries else 1
        budget_per_query = kwargs.get("budget_per_query", default_budget_per_query)
        if budget_per_query <= 0:
            budget_per_query = 1

        # Ensure enough candidates are available for re-ranking
        initial_candidates_for_rerank_per_query = kwargs.get(
            "initial_candidates_for_rerank",
            max(1, budget_per_query * 2, limit // len(queries) if queries else limit),
        )
        id_field = kwargs.get("fuse_tools_by_field", "procedure")

        final_retrieved_tools: List[PydanticModel] = []
        added_model_identifiers: set[Any] = set()

        try:
            gathered_outcomes = await semaphore_gather(
                *[
                    self._process_query_for_ordered_retrieval_async(
                        query_text=query_text,
                        initial_candidates_for_rerank_per_query=initial_candidates_for_rerank_per_query,
                        budget_per_query=budget_per_query,
                    )
                    for query_text in queries
                ]
            )

            results_per_query: List[List[PydanticModel]] = [
                (outcome if outcome is not None else [])
                for outcome in gathered_outcomes
            ]

            for query_idx in range(len(queries)):
                if len(final_retrieved_tools) >= limit:
                    break

                tools_for_this_query = results_per_query[query_idx]

                for tool_model in tools_for_this_query:
                    if len(final_retrieved_tools) >= limit:
                        break

                    model_identifier = getattr(tool_model, id_field, str(tool_model))
                    if model_identifier not in added_model_identifiers:
                        final_retrieved_tools.append(tool_model)
                        added_model_identifiers.add(model_identifier)

            return final_retrieved_tools[:limit]
        except Exception as e:
            self.logger.error(f"Error in retrieve_ordered_ranked: {e}")
            raise

    async def batch_retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[PydanticModel]]:
        """
        Retrieves models using BM25 ranking, with ordered processing of queries,
        re-ranking, and a budget for batches of queries.
        """
        if not queries or limit <= 0:
            return []

        if not self.bm25_model or not self.tools or not self.raw_corpus_texts:
            return []

        budget_per_query = limit
        # Ensure enough candidates are available for re-ranking
        initial_candidates_for_rerank_per_query = kwargs.get(
            "initial_candidates_for_rerank",
            max(1, budget_per_query * 2, limit),
        )

        try:
            gathered_outcomes = await semaphore_gather(
                *[
                    self._process_query_for_ordered_retrieval_async(
                        query_text=query_text,
                        initial_candidates_for_rerank_per_query=initial_candidates_for_rerank_per_query,
                        budget_per_query=budget_per_query,
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
