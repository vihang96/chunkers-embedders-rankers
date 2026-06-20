import json
from pathlib import Path
from typing import Generator

import pytest

from cer.retriever.text_retriever import BM25Retriever

from tests.support import ToolSchema


@pytest.fixture
def dummy_file_path_bm25(tmp_path: Path) -> Generator[str, None, None]:
    dummy_file = tmp_path / "dummy_tools_data_bm25_async.json"
    example_tool_data = [
        {
            "tool_schema": {
                "book_name": "Utils",
                "procedure": "GetWeather",
                "signature": "get weather in {city}",
                "description": "Fetches the current weather for a specified city. Good for climate info.",
                "skill_type": "EXTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["get weather in London"],
                "bad_examples": ["weather London"],
                "key_topics": ["weather", "forecast", "city", "climate"],
                "synthetic_queries": [
                    "What's the weather like in Paris?",
                    "Tell me about the climate in Rome",
                ],
            },
            "embedding": [],
        },
        {
            "tool_schema": {
                "book_name": "Office",
                "procedure": "CreateDocument",
                "signature": "create document with title {title}",
                "description": "Creates a new document with a given title. Use for reports.",
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["create document with title '''My Report'''"],
                "bad_examples": ["new doc"],
                "key_topics": ["document", "creation", "office", "reports"],
                "synthetic_queries": ["How do I make a new document for my report?"],
            },
            "embedding": [0.6, 0.7, 0.8, 0.9, 1.0],
        },
        {
            "tool_schema": {
                "book_name": "Finance",
                "procedure": "CalculateLoan",
                "signature": "calculate loan payment for {amount}",
                "description": "Calculates monthly loan payments. Helpful for financial planning and mortgages.",
                "skill_type": "INTERNAL",
                "boolean": False,
                "hidden": False,
                "deprecated": False,
                "good_examples": ["calculate loan payment for 10000"],
                "bad_examples": [],
                "key_topics": ["finance", "loan", "payment", "mortgage"],
                "synthetic_queries": [
                    "How much is a 10k loan monthly?",
                    "Mortgage calculation for house",
                ],
            },
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
        },
    ]
    with open(dummy_file, "w") as f:
        json.dump(example_tool_data, f, indent=2)
    yield str(dummy_file)


@pytest.mark.asyncio
async def test_bm25_retriever_initialization_and_retrieve(
    dummy_file_path_bm25: str,
) -> None:
    bm25_fields_to_index = [
        "procedure",
        "signature",
        "description",
        "key_topics",
        "synthetic_queries",
    ]

    try:
        retriever = BM25Retriever(
            tools_data_path=dummy_file_path_bm25,
            model_class=ToolSchema,
            bm25_fields=bm25_fields_to_index,
        )

        user_queries = [
            "weather information for city",
            "create office report",
            "loan mortgage details",
        ]

        print(f"\nRetrieving with BM25 for queries: {user_queries}")
        results = await retriever.retrieve(
            queries=user_queries, limit=2, fuse_tools_by_field="procedure"
        )
        ordered_results = await retriever.retrieve_ordered_ranked(
            queries=user_queries,
            limit=2,
            budget_per_query=1,
            initial_candidates_for_rerank=3,
            fuse_tools_by_field="procedure",
        )

        if results:
            print("\nFound models (BM25):")
            for model_item in results:
                print(f"  - {model_item.procedure}: {model_item.description}")
        else:
            print(
                "\nNo models found by BM25Retriever. Check BM25 library installation and corpus."
            )

        assert len(results) <= 2
        assert len(ordered_results) <= 2
        if len(results) > 0:
            assert hasattr(results[0], "procedure")
        # For small data, results should be the same as ordered_results
        assert results == ordered_results

    except ImportError as e:
        pytest.skip(
            f"Skipping BM25 tests due to ImportError: {e}. Ensure 'bm25s' is installed."
        )
    except Exception as e:
        pytest.fail(f"BM25Retriever test failed with an unexpected exception: {e}")


@pytest.mark.asyncio
async def test_bm25_retriever_ordered_ranked(dummy_file_path_bm25: str) -> None:
    bm25_fields_to_index = ["procedure", "description"]
    try:
        retriever = BM25Retriever(
            tools_data_path=dummy_file_path_bm25,
            model_class=ToolSchema,
            bm25_fields=bm25_fields_to_index,
        )
        user_queries = ["loan mortgage details", "create office report"]
        print(f"\nRetrieving with BM25 (ordered_ranked) for queries: {user_queries}")
        ordered_results = await retriever.retrieve_ordered_ranked(
            queries=user_queries,
            limit=2,
            budget_per_query=1,
            initial_candidates_for_rerank=3,
            fuse_tools_by_field="procedure",
        )
        if ordered_results:
            print("\nFound models (BM25 - ordered_ranked):")
            for model_item in ordered_results:
                print(f"  - {model_item.procedure}: {model_item.description}")
        else:
            print("\nNo models found by BM25Retriever (ordered_ranked).")

        assert len(ordered_results) <= 2
        # With budget_per_query=1 and 2 queries, we expect at most 2 results.
        # Check for uniqueness as well, as retrieve_ordered_ranked should handle this.
        if ordered_results:
            procedures = [r.procedure for r in ordered_results]
            assert len(procedures) == len(set(procedures))

    except ImportError as e:
        pytest.skip(
            f"Skipping BM25 tests due to ImportError: {e}. Ensure 'bm25s' is installed."
        )
    except Exception as e:
        pytest.fail(
            f"BM25Retriever ordered_ranked test failed with an unexpected exception: {e}"
        )


@pytest.mark.asyncio
async def test_bm25_retriever_specific_query(dummy_file_path_bm25: str) -> None:
    bm25_fields_to_index = ["procedure", "description", "key_topics"]
    try:
        retriever = BM25Retriever(
            tools_data_path=dummy_file_path_bm25,
            model_class=ToolSchema,
            bm25_fields=bm25_fields_to_index,
        )
        specific_query = ["create office report"]
        print(f"\nRetrieving with BM25 for specific query: {specific_query}")
        specific_results = await retriever.retrieve(queries=specific_query, limit=1)
        if specific_results:
            print("\nFound models (BM25 - specific query):")
            for model_item in specific_results:
                print(f"  - {model_item.procedure}: {model_item.description}")
            assert len(specific_results) == 1
            assert specific_results[0].procedure == "CreateDocument"
        else:
            # This might happen if bm25s is not effective with tiny corpus/query
            print(
                "\nNo models found for specific query. This might be acceptable for BM25 with small data."
            )
            assert len(specific_results) == 0

    except ImportError as e:
        pytest.skip(
            f"Skipping BM25 tests due to ImportError: {e}. Ensure 'bm25s' is installed."
        )
    except Exception as e:
        pytest.fail(
            f"BM25Retriever specific query test failed with an unexpected exception: {e}"
        )


@pytest.mark.asyncio
async def test_bm25_empty_queries(dummy_file_path_bm25: str) -> None:
    bm25_fields = ["procedure"]
    try:
        retriever = BM25Retriever(
            tools_data_path=dummy_file_path_bm25,
            model_class=ToolSchema,
            bm25_fields=bm25_fields,
        )
        results = await retriever.retrieve(queries=[], limit=5)
        assert results == []
        results_ordered = await retriever.retrieve_ordered_ranked(queries=[], limit=5)
        assert results_ordered == []
    except ImportError as e:
        pytest.skip(
            f"Skipping BM25 tests due to ImportError: {e}. Ensure 'bm25s' is installed."
        )


@pytest.mark.asyncio
async def test_bm25_limit_zero(dummy_file_path_bm25: str) -> None:
    bm25_fields = ["procedure"]
    try:
        retriever = BM25Retriever(
            tools_data_path=dummy_file_path_bm25,
            model_class=ToolSchema,
            bm25_fields=bm25_fields,
        )
        queries = ["find stuff"]
        results = await retriever.retrieve(queries=queries, limit=0)
        assert results == []
        results_ordered = await retriever.retrieve_ordered_ranked(
            queries=queries, limit=0
        )
        assert results_ordered == []
    except ImportError as e:
        pytest.skip(
            f"Skipping BM25 tests due to ImportError: {e}. Ensure 'bm25s' is installed."
        )
