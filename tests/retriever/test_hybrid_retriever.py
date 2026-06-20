import asyncio
import json
from pathlib import Path
from typing import Generator, List

import pytest

from cer.retriever.hybrid_retriever import HybridRetriever

from tests.support import ToolSchema


@pytest.fixture
def dummy_file_path_hybrid(tmp_path: Path) -> Generator[str, None, None]:
    dummy_file = tmp_path / "dummy_tools_data_hybrid_rrf.json"
    example_tool_data = [
        {
            "tool_schema": {
                "book_name": "Search",
                "procedure": "WebSearch",
                "signature": "search web for {q}",
                "description": "Finds information using web search.",
                "key_topics": ["web", "internet", "search", "information"],
                "skill_type": "EXTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "synthetic_queries": ["what is X?", "find Y"],
            },
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        {
            "tool_schema": {
                "book_name": "Productivity",
                "procedure": "EmailDraft",
                "signature": "new email to {to} about {subject}",
                "description": "Drafts a new email message.",
                "key_topics": ["email", "compose", "message", "send"],
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "synthetic_queries": [
                    "draft an email to marketing about sales",
                    "compose new mail to boss",
                ],
            },
            "embedding": [0.6, 0.7, 0.8, 0.9, 1.0],
        },
        {
            "tool_schema": {
                "book_name": "Calendar",
                "procedure": "MeetingScheduler",
                "signature": "schedule {event} at {time}",
                "description": "Adds an event or meeting to the calendar system.",
                "key_topics": ["calendar", "meeting", "schedule", "event"],
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "synthetic_queries": [
                    "set up a meeting for tomorrow with team",
                    "schedule a call for next week",
                ],
            },
            "embedding": [0.21, 0.22, 0.31, 0.32, 0.41],
        },
        {
            "tool_schema": {
                "book_name": "Search",
                "procedure": "ImageFinder",
                "signature": "find images of {topic}",
                "description": "Searches for images on a specific topic online.",
                "key_topics": ["image", "picture", "photo", "search"],
                "skill_type": "EXTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "synthetic_queries": [
                    "show me pictures of cats",
                    "find image of a sunset",
                ],
            },
            "embedding": [0.51, 0.41, 0.31, 0.21, 0.11],
        },
        {
            "tool_schema": {
                "book_name": "Files",
                "procedure": "ListMyFiles",
                "signature": "list files in {folder_path}",
                "description": "Lists files in a user specified folder or directory.",
                "key_topics": ["files", "directory", "folder", "list"],
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "synthetic_queries": [
                    "what files are in my documents?",
                    "show directory contents of C drive",
                ],
            },
            "embedding": [],
        },
    ]
    with open(dummy_file, "w") as f:
        json.dump(example_tool_data, f, indent=2)
    yield str(dummy_file)
    # No need to os.remove, tmp_path handles cleanup


async def dummy_async_embedder(texts: List[str]) -> List[List[float]]:
    await asyncio.sleep(0.01)
    dim = 5
    embeddings = []
    for text in texts:
        if text and len(text) > 2:
            embeddings.append([0.05 * len(text) + i * 0.02 for i in range(dim)])
        else:
            embeddings.append([])
    return embeddings


async def dummy_final_reranker(
    query: str, models: List[ToolSchema]
) -> List[ToolSchema]:
    print(
        f"Dummy final reranker called with query: '{query[:50]}...' and {len(models)} models."
    )
    await asyncio.sleep(0.01)
    return models[::-1]  # Simple reversal for testing


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf(dummy_file_path_hybrid: str) -> None:
    print("--- Testing HybridRetriever (RRF Strategy - Refactored) ---")
    bm25_fields = [
        "procedure",
        "signature",
        "description",
        "key_topics",
        "synthetic_queries",
    ]

    retriever = HybridRetriever(
        tools_data_path=dummy_file_path_hybrid,
        model_class=ToolSchema,
        bm25_fields=bm25_fields,
        embedder_func=dummy_async_embedder,
        final_reranker_func=dummy_final_reranker,
    )
    print(f"  Using final reranker: {retriever.final_reranker_func is not None}")

    queries = [
        "search for web information about meetings",
        "find image files online",
        "draft email about sales",
    ]
    test_limit = 2

    print(
        f"\nRetrieving with RRF, limit={test_limit}, similarity_threshold=0.01 (passed via kwargs)"
    )
    results = await retriever.retrieve(
        queries=queries, limit=test_limit, similarity_threshold=0.01
    )

    print("\nFinal RRF Re-ranked Results (then final-reranked):")
    if results:
        for i, model_item in enumerate(results):
            print(
                f"  {i+1}. {model_item.procedure} (Desc: {model_item.description[:30]}...)"
            )
    else:
        print("  No models found after RRF and final re-ranking.")

    assert (
        len(results) <= test_limit
    ), f"Expected max {test_limit} results, got {len(results)}"
    if results:  # Check if results is not empty before accessing elements
        assert (
            results[0].procedure == "MeetingScheduler"
            or results[0].procedure == "WebSearch"
        )  # Order can vary slightly based on dummy scores


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf_single_query_no_reranker(
    dummy_file_path_hybrid: str,
) -> None:
    bm25_fields = [
        "procedure",
        "signature",
        "description",
        "key_topics",
        "synthetic_queries",
    ]
    single_query = ["schedule a meeting with the entire team next week"]
    retriever_no_final_rerank = HybridRetriever(
        tools_data_path=dummy_file_path_hybrid,
        model_class=ToolSchema,
        bm25_fields=bm25_fields,
        embedder_func=dummy_async_embedder,
        final_reranker_func=None,  # Test without final reranker
    )
    results_single = await retriever_no_final_rerank.retrieve(
        queries=single_query, limit=1, similarity_threshold=0.005
    )
    print("\nSingle Query RRF Re-ranked Result (limit 1, no final reranker):")
    if results_single:
        print(f"  1. {results_single[0].procedure}")
    else:
        print("  No models found for single query.")
    assert len(results_single) <= 1
    if results_single:
        assert results_single[0].procedure == "MeetingScheduler"


@pytest.mark.asyncio
async def test_retrieve_ordered_ranked(dummy_file_path_hybrid: str) -> None:
    bm25_fields = ["procedure", "description"]
    retriever = HybridRetriever(
        tools_data_path=dummy_file_path_hybrid,
        model_class=ToolSchema,
        bm25_fields=bm25_fields,
        embedder_func=dummy_async_embedder,
        final_reranker_func=dummy_final_reranker,
    )

    queries = ["search web", "draft email"]
    limit = 3
    results = await retriever.retrieve_ordered_ranked(queries=queries, limit=limit)

    assert len(results) <= limit
    # Add more specific assertions based on expected behavior of retrieve_ordered_ranked
    # For example, check uniqueness if fuse_tools_by_field is used
    procedures = [r.procedure for r in results]
    assert len(procedures) == len(
        set(procedures)
    ), "Expected unique procedures in ordered ranked results"

    # Test with budget_per_query
    results_budgeted = await retriever.retrieve_ordered_ranked(
        queries=queries, limit=limit, budget_per_query=1
    )
    assert len(results_budgeted) <= limit
    # Each query should contribute at most 1 unique tool
    # Since there are 2 queries, we expect at most 2 results if budget_per_query is 1
    # (unless the overall limit is smaller)
    assert len(results_budgeted) <= min(limit, len(queries) * 1)


@pytest.mark.asyncio
async def test_retrieve_empty_queries(dummy_file_path_hybrid: str) -> None:
    bm25_fields = ["procedure"]
    retriever = HybridRetriever(
        tools_data_path=dummy_file_path_hybrid,
        model_class=ToolSchema,
        bm25_fields=bm25_fields,
        embedder_func=dummy_async_embedder,
    )
    results = await retriever.retrieve(queries=[], limit=5)
    assert results == []

    results_ordered = await retriever.retrieve_ordered_ranked(queries=[], limit=5)
    assert results_ordered == []


@pytest.mark.asyncio
async def test_retrieve_limit_zero(dummy_file_path_hybrid: str) -> None:
    bm25_fields = ["procedure"]
    retriever = HybridRetriever(
        tools_data_path=dummy_file_path_hybrid,
        model_class=ToolSchema,
        bm25_fields=bm25_fields,
        embedder_func=dummy_async_embedder,
    )
    queries = ["find stuff"]
    results = await retriever.retrieve(queries=queries, limit=0)
    assert results == []

    results_ordered = await retriever.retrieve_ordered_ranked(queries=queries, limit=0)
    assert results_ordered == []
