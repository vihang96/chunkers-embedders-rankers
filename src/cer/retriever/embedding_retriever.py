from logging import Logger
from typing import Any, Awaitable, Callable, List, Optional, Set, Tuple, Type

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except ImportError:
    cosine_similarity = None

from ..utils.schemas import PydanticModel, ToolWithEmbedding
from .client import RetrieverClient, semaphore_gather


class EmbeddingRetriever(RetrieverClient[PydanticModel]):
    def __init__(
        self,
        tools_data_path: str,
        model_class: Type[PydanticModel],  # Added model_class
        embedder_func: Callable[[List[str]], Awaitable[List[List[float]]]],
        reranker_func: Optional[
            Callable[
                [str, List[PydanticModel]], Awaitable[List[PydanticModel]]
            ]  # Uses PydanticModel
        ] = None,
        tools_with_embeddings: Optional[List[ToolWithEmbedding[PydanticModel]]] = None,
        logger: Optional[Logger] = None,
    ):
        # Pass model_class to super
        super().__init__(
            tools_data_path=tools_data_path,
            model_class=model_class,
            embedder_func=embedder_func,
            reranker_func=reranker_func,
            tools_with_embeddings=tools_with_embeddings,
            logger=logger,
        )
        # self.embedder_func is set by super() if provided
        if not self.embedder_func:  # This check remains valid
            raise ValueError("EmbeddingRetriever requires an embedder_func.")
        if cosine_similarity is None:
            raise ImportError(
                "EmbeddingRetriever requires 'scikit-learn' for optimized cosine similarity. "
                "Please install it (e.g., pip install scikit-learn)."
            )

        # Pre-process tool embeddings for efficient similarity calculation
        self.tool_schemas_ordered: List[PydanticModel] = []
        valid_tool_embeddings: List[List[float]] = []

        for item in self.tools_with_embeddings:
            if (
                item.embedding
                and isinstance(item.embedding, list)
                and len(item.embedding) > 0
            ):
                self.tool_schemas_ordered.append(
                    item.tool_schema
                )  # item.tool_schema is PydanticModel
                valid_tool_embeddings.append(item.embedding)
            else:
                # Generic way to get a name for the warning
                model_name = getattr(
                    item.tool_schema,
                    "procedure",
                    getattr(item.tool_schema, "name", "Unnamed Model"),
                )
                self.logger.warning(
                    f"Warning: Model '{model_name}' has missing or invalid embedding. Skipping."
                )

        if not valid_tool_embeddings:
            self.tool_embeddings_matrix = np.array([])
            self.logger.warning(
                "Warning: No valid model embeddings found to create a matrix."
            )
        else:
            try:
                first_emb_dim = len(valid_tool_embeddings[0])
                if not all(len(emb) == first_emb_dim for emb in valid_tool_embeddings):
                    raise ValueError(
                        "All model embeddings must have the same dimension."
                    )
                self.tool_embeddings_matrix = np.array(valid_tool_embeddings)
            except ValueError as e:
                self.logger.error(
                    f"Error creating model embeddings matrix: {e}. Ensure all embeddings are valid and have consistent dimensions."
                )
                self.tool_embeddings_matrix = np.array([])
                self.tool_schemas_ordered = []

        self.logger.info(
            f"Tool embeddings matrix shape: {self.tool_embeddings_matrix.shape}"
        )
        self.logger.info(f"Number of ordered tools: {len(self.tool_schemas_ordered)}")

    async def retrieve(
        self, queries: List[str], limit: int = 10, **kwargs: Any
    ) -> List[PydanticModel]:
        """
        Retrieves models based on cosine similarity of their embeddings with query embeddings.

        Args:
            queries: List of query strings.
            limit: Maximum number of models to retrieve.

        Keyword Args:
            similarity_threshold (optional): Minimum similarity score for a candidate to be included. Uses 0.6 by default.
            initial_candidates_for_rerank (optional): Number of initial candidates for re-ranking. Uses limit by default.

        Returns:
            List of PydanticModel objects.
        """
        similarity_threshold = kwargs.get("similarity_threshold", 0.6)
        if not isinstance(similarity_threshold, (float, int)):
            raise ValueError("'similarity_threshold' must be a float or int.")

        initial_candidates_for_rerank = kwargs.get(
            "initial_candidates_for_rerank", limit
        )

        if not queries:
            return []
        if self.tool_embeddings_matrix.size == 0:
            return []
        if not self.embedder_func:
            raise ValueError("Embedder function is not available.")

        query_embeddings_list = await self.embedder_func(queries)

        valid_query_embeddings = [
            emb
            for emb in query_embeddings_list
            if emb and isinstance(emb, list) and len(emb) > 0
        ]
        if not valid_query_embeddings:
            self.logger.warning("No valid query embeddings found.")
            return []

        try:
            tool_emb_dim = self.tool_embeddings_matrix.shape[1]
            if not all(len(emb) == tool_emb_dim for emb in valid_query_embeddings):
                raise ValueError(
                    f"All query embeddings must have dimension {tool_emb_dim} to match model embeddings."
                )
            query_embeddings_matrix = np.array(valid_query_embeddings)
            self.logger.info(
                f"Query embeddings matrix shape: {query_embeddings_matrix.shape}"
            )
        except ValueError as e:
            self.logger.error(
                f"Error creating query embeddings matrix: {e}. Ensure consistent dimensions."
            )
            return []

        if query_embeddings_matrix.size == 0:
            return []

        similarity_matrix = cosine_similarity(
            query_embeddings_matrix, self.tool_embeddings_matrix
        )

        if similarity_matrix.shape[1] == 0:
            return []

        max_sim_per_tool = np.max(similarity_matrix, axis=0)
        self.logger.info(f"Max similarity per tool: {max_sim_per_tool}")

        scored_tools: List[Tuple[float, PydanticModel]] = []
        for i, max_sim in enumerate(max_sim_per_tool):
            if max_sim >= similarity_threshold:
                if i < len(self.tool_schemas_ordered):
                    scored_tools.append((max_sim, self.tool_schemas_ordered[i]))

        scored_tools.sort(key=lambda x: x[0], reverse=True)
        self.logger.info(f"Total scored tools: {len(scored_tools)}")

        tools_for_reranking: List[PydanticModel] = [
            tool for _, tool in scored_tools[:initial_candidates_for_rerank]
        ]

        final_results: List[PydanticModel]
        if self.reranker_func and tools_for_reranking:
            # Ensure query_context is a single string if reranker expects that
            self.logger.info(f"Reranking with {len(tools_for_reranking)} tools")
            query_context = " ".join(queries) if isinstance(queries, list) else queries
            try:
                final_results = await self.reranker_func(
                    query_context, tools_for_reranking
                )
            except Exception as e:
                self.logger.error(f"Error in reranking: {e}")
                final_results = tools_for_reranking
        else:
            self.logger.info(
                f"No reranker provided, returning {len(tools_for_reranking)} tools"
            )
            final_results = tools_for_reranking

        return final_results[:limit]

    async def retrieve_ordered_ranked(
        self, queries: List[str], limit: int = 10, **kwargs: Any
    ) -> List[PydanticModel]:
        """
        Retrieves models with ordered processing of sub-queries and re-ranking.
        The underlying logic for retrieving and ranking per query is similar to `retrieve`,
        but applied sequentially with budget and uniqueness constraints.

        Args:
            queries: List of query strings (sub-queries in order).
            limit: Maximum total number of models to retrieve.

        Keyword Args:
            budget_per_query (optional): Number of top candidates per query. Uses limit by default.
            initial_candidates_for_rerank (optional): Number of initial candidates for re-ranking. Uses limit by default.
            similarity_threshold (optional): Minimum similarity score for a candidate to be included. Uses 0.6 by default.
            fuse_tools_by_field (optional): Field to fuse tools by. Must be unique for each tool. Uses "procedure" by default.

        Returns:
            List of PydanticModel objects.
        """
        if not queries or limit <= 0:
            return []
        if self.tool_embeddings_matrix.size == 0:
            return []
        if not self.embedder_func:
            raise ValueError("Embedder function is not available.")

        similarity_threshold = kwargs.get("similarity_threshold", 0.6)
        # Default initial_candidates_for_rerank to be at least budget_per_query
        # or a fraction of limit if budget_per_query is not set.
        default_budget_per_query = (limit // len(queries)) if queries else 1
        budget_per_query = kwargs.get("budget_per_query", default_budget_per_query)
        if budget_per_query <= 0:
            budget_per_query = 1

        # Ensure enough candidates are available for re-ranking
        initial_candidates_for_rerank_per_query = kwargs.get(
            "initial_candidates_for_rerank",
            max(1, budget_per_query * 2, limit // len(queries) if queries else limit),
        )

        fuse_tools_by_field = kwargs.get(
            "fuse_tools_by_field", "procedure"
        )  # For uniqueness

        if budget_per_query <= 0:
            budget_per_query = 1

        final_retrieved_tools: List[PydanticModel] = []
        seen_tool_identifiers: Set[Any] = set()

        all_query_embeddings_list = await self.embedder_func(queries)

        if (
            self.tool_embeddings_matrix.ndim < 2
            or self.tool_embeddings_matrix.shape[1] == 0
        ):
            self.logger.warning(
                "Warning: Tool embeddings matrix is not properly shaped or is empty."
            )
            return []
        tool_emb_dim = self.tool_embeddings_matrix.shape[1]

        for i, query_text in enumerate(queries):
            if len(final_retrieved_tools) >= limit:
                break

            if i >= len(all_query_embeddings_list):
                continue

            query_embedding = all_query_embeddings_list[i]
            if not (
                query_embedding
                and isinstance(query_embedding, list)
                and len(query_embedding) > 0
            ):
                continue

            if len(query_embedding) != tool_emb_dim:
                continue

            query_embedding_matrix = np.array([query_embedding])

            try:
                sim_scores_for_query = cosine_similarity(
                    query_embedding_matrix, self.tool_embeddings_matrix
                )[0]
            except ValueError:
                continue

            scored_initial_candidates: List[Tuple[float, PydanticModel]] = []
            for tool_idx, score in enumerate(sim_scores_for_query):
                if score >= similarity_threshold:
                    if tool_idx < len(self.tool_schemas_ordered):
                        scored_initial_candidates.append(
                            (score, self.tool_schemas_ordered[tool_idx])
                        )

            scored_initial_candidates.sort(key=lambda x: x[0], reverse=True)

            tools_for_reranking: List[PydanticModel] = [
                tool
                for _, tool in scored_initial_candidates[
                    :initial_candidates_for_rerank_per_query
                ]
            ]

            if not tools_for_reranking:
                continue

            processed_query_results: List[PydanticModel]
            if self.reranker_func:
                self.logger.info(f"Reranking with {len(tools_for_reranking)} tools")
                try:
                    processed_query_results = await self.reranker_func(
                        query_text, tools_for_reranking
                    )
                except Exception as e:
                    self.logger.error(f"Error in reranking: {e}")
                    processed_query_results = tools_for_reranking
            else:
                self.logger.info(
                    f"No reranker provided, returning {len(tools_for_reranking)} tools"
                )
                processed_query_results = tools_for_reranking

            added_for_current_query = 0
            for tool_model in processed_query_results:  # tool_model is PydanticModel
                if len(final_retrieved_tools) >= limit:
                    break
                if added_for_current_query >= budget_per_query:
                    break

                # Use fuse_tools_by_field for uniqueness
                tool_identifier = getattr(tool_model, fuse_tools_by_field, None)
                if tool_identifier is None:  # Fallback if field is missing
                    tool_identifier = str(tool_model)

                if tool_identifier not in seen_tool_identifiers:
                    final_retrieved_tools.append(tool_model)
                    seen_tool_identifiers.add(tool_identifier)
                    added_for_current_query += 1

        return final_retrieved_tools[:limit]

    async def batch_retrieve(
        self,
        queries: List[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[List[PydanticModel]]:
        """
        Retrieves models for queries in batches.
        """
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
