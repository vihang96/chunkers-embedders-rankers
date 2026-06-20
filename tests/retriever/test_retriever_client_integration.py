import json
from pathlib import Path
from typing import List

import pytest
import pytest_asyncio

from cer.embedder import (
    AbstractEmbeddingModel,
    SentenceTransformerEmbedderConfig,
    SentenceTransformerEmbeddingModel,
    generate_object_embeddings,
)
from cer.retriever.embedding_retriever import EmbeddingRetriever
from cer.retriever.hybrid_retriever import HybridRetriever
from cer.utils.schemas import ToolWithEmbedding

from tests.support import ToolSchema

# A small, fast sentence transformer model for testing
# If this model is not available, the test will try to download it.
TEST_EMBEDDING_DIM = 512


@pytest.fixture
def schema_fields_to_index() -> List[str]:
    return [
        "procedure",
        "signature",
        "description",
        "key_topics",
        "synthetic_queries",
    ]


@pytest_asyncio.fixture(scope="module")
async def real_embedder() -> AbstractEmbeddingModel:
    """Provides a real SentenceTransformerEmbeddingModel instance."""
    config = SentenceTransformerEmbedderConfig(embedding_dim=TEST_EMBEDDING_DIM)
    model = SentenceTransformerEmbeddingModel(config=config)
    return model


"""
Text based retrieval and Hybrid retrieval strategies are very sensitive to the schema generated for every tool.
It is important to be prudent about the exact text generated for each schema field specially the description and synthetic questions.

Considerations:
1. Tool description should be long, high quality and descriptive (such as when to use the tool and when not to).
2. Ensure synthetic questions are diverse, mirror future user questions about the tool, and utilize parameters in the question.
3. Generate the key topics based on not only the tool name and description, but also any synthetic questions.
"""


@pytest.fixture
def sample_tools_data() -> List[ToolSchema]:
    """Provides a list of sample ToolSchema objects for testing."""
    return [
        ToolSchema(
            book_name="Calendar",
            procedure="CreateEvent",
            signature="create event titled {title} at {time} on {date}",
            description="Creates a new calendar event with a specified title, time, and date.",
            skill_type="INTERNAL",
            key_topics=["calendar", "event", "schedule", "meeting"],
            synthetic_queries=[
                "add this event to my calendar",
            ],
        ),
        ToolSchema(
            book_name="Email",
            procedure="SendEmail",
            signature="send email to {recipient} with subject {subject} and body {body}",
            description="Sends an email to a specified recipient with a given subject and body.",
            skill_type="EXTERNAL",
            key_topics=["email", "communication", "send", "message"],
            synthetic_queries=[
                "email the ops team for handover",
            ],
        ),
        ToolSchema(
            book_name="FileSystem",
            procedure="ReadFile",
            signature="read file from path {path}",
            description="Reads the content of a file from the specified path.",
            skill_type="INTERNAL",
            key_topics=["file", "read", "filesystem", "storage"],
            synthetic_queries=[
                "load a file from the file system",
            ],
        ),
        ToolSchema(  # A slightly different tool to test distinction
            book_name="Calendar",
            procedure="ListEvents",
            signature="list events on {date}",
            description="Lists all calendar events scheduled for a given date.",
            skill_type="INTERNAL",
            key_topics=["calendar", "event", "list", "view"],
            synthetic_queries=[
                "check my calendar for today",
            ],
        ),
    ]


@pytest_asyncio.fixture
async def tools_with_embeddings_file(
    tmp_path: Path,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
    schema_fields_to_index: List[str],
) -> str:
    """
    Creates a temporary JSON file with tools and their pre-computed embeddings
    using the real_embedder.
    """
    tools_file = tmp_path / "test_tools_with_embeddings.json"

    tools_with_embeddings_list: List[ToolWithEmbedding[ToolSchema]] = []

    if sample_tools_data:
        embeddings = await generate_object_embeddings(
            model=real_embedder,
            objects=sample_tools_data,
            input_type="document",
            fields=schema_fields_to_index,
        )
    else:
        embeddings = []

    # Create ToolWithEmbedding objects
    # Assuming one-to-one mapping between sample_tools_data and embeddings generated
    # This requires careful handling if some tools don't have descriptions or fail embedding

    current_embedding_idx = 0
    for tool in sample_tools_data:
        if current_embedding_idx < len(embeddings):
            embedding = embeddings[current_embedding_idx]
            tools_with_embeddings_list.append(
                ToolWithEmbedding[ToolSchema](tool_schema=tool, embedding=embedding)
            )
            current_embedding_idx += 1
        else:
            print(f"Warning: Missing embedding for tool: {tool.procedure}")

    # Serialize to JSON: Pydantic models need to be converted to dicts
    serializable_data = [item.model_dump() for item in tools_with_embeddings_list]

    with open(tools_file, "w") as f:
        json.dump(serializable_data, f, indent=2)

    return str(tools_file)


@pytest.mark.asyncio
async def test_retriever_client_integration_initialization_and_retrieve(
    tools_with_embeddings_file: str,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
) -> None:
    """
    Integration test for EmbeddingRetriever initialization and the retrieve method
    using a real embedder and pre-computed embeddings.
    """
    retriever = EmbeddingRetriever(
        tools_data_path=tools_with_embeddings_file,
        model_class=ToolSchema,
        embedder_func=real_embedder.get_batch_embedding,
    )

    assert len(retriever.tools_with_embeddings) > 0
    # We expect as many tools as were successfully processed (had descriptions and got embeddings)
    # This count needs to be accurate based on the sample_tools_data and embedding logic.
    # All 4 sample tools have descriptions, so all should be included.
    assert len(retriever.tools_with_embeddings) == len(sample_tools_data)
    assert retriever.tool_embeddings_matrix.shape[0] == len(sample_tools_data)
    assert retriever.tool_embeddings_matrix.shape[1] == TEST_EMBEDDING_DIM

    # Test retrieve method
    query = "How do I make a new calendar event?"
    # The embedder_func in RetrieverClient is used for the query,
    # and then compared against tool_embeddings_matrix.
    retrieved_tools = await retriever.retrieve(
        queries=[query], limit=2, similarity_threshold=0.5
    )

    assert len(retrieved_tools) > 0
    assert len(retrieved_tools) <= 2

    # Check if the top retrieved tool is relevant
    # This is a simple check; more sophisticated checks might be needed
    top_tool = retrieved_tools[0]
    assert isinstance(top_tool, ToolSchema)
    print(
        f"(Embedding) Query: '{query}' -> Top retrieved tool: '{top_tool.procedure}' ({top_tool.description})"
    )

    # Expect "CreateEvent" to be the most relevant for the query
    assert top_tool.procedure == "CreateEvent"

    query_send_email = "send a message to my friend"
    retrieved_email_tools = await retriever.retrieve(
        queries=[query_send_email], limit=1, similarity_threshold=0.5
    )
    assert len(retrieved_email_tools) == 1
    assert retrieved_email_tools[0].procedure == "SendEmail"
    print(
        f"(Embedding) Query: '{query_send_email}' -> Top retrieved tool: '{retrieved_email_tools[0].procedure}' ({retrieved_email_tools[0].description})"
    )


@pytest.mark.asyncio
async def test_retriever_client_integration_batch_retrieve(
    tools_with_embeddings_file: str,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
) -> None:
    """
    Integration test for the batch_retrieve method.
    """
    retriever = EmbeddingRetriever(
        tools_data_path=tools_with_embeddings_file,
        model_class=ToolSchema,
        embedder_func=real_embedder.get_batch_embedding,
    )

    queries = [
        "How do I make a new calendar event?",
        "how to read a file content?",
        "I want to list my appointments",
    ]

    batch_results = await retriever.batch_retrieve(
        queries=queries, limit=1, similarity_threshold=0.5
    )

    assert len(batch_results) == len(queries)

    # Check results for each query
    # Query 1: "How do I make a new calendar event?" -> CreateEvent
    assert len(batch_results[0]) == 1
    assert batch_results[0][0].procedure == "CreateEvent"
    print(
        f"(Embedding) Batch Query 1: '{queries[0]}' -> Top tool: '{batch_results[0][0].procedure}'"
    )

    # Query 2: "how to read a file content?" -> ReadFile
    assert len(batch_results[1]) == 1
    assert batch_results[1][0].procedure == "ReadFile"
    print(
        f"(Embedding) Batch Query 2: '{queries[1]}' -> Top tool: '{batch_results[1][0].procedure}'"
    )

    # Query 3: "I want to list my appointments" -> ListEvents
    assert len(batch_results[2]) == 1
    assert batch_results[2][0].procedure == "ListEvents"
    print(
        f"(Embedding) Batch Query 3: '{queries[2]}' -> Top tool: '{batch_results[2][0].procedure}'"
    )


@pytest.mark.asyncio
async def test_retriever_client_integration_retrieve_ordered_ranked(
    tools_with_embeddings_file: str,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
) -> None:
    """
    Integration test for the retrieve_ordered_ranked method.
    """
    retriever = EmbeddingRetriever(
        tools_data_path=tools_with_embeddings_file,
        model_class=ToolSchema,
        embedder_func=real_embedder.get_batch_embedding,
    )

    queries = [
        "How do I make a new calendar event?",  # Should strongly match CreateEvent
        "Any way to check my schedule?",  # Should match ListEvents or CreateEvent
    ]
    limit = 2
    # budget_per_query means how many top candidates from each query's individual search are considered
    # for the final re-ranking/selection.
    results = await retriever.retrieve_ordered_ranked(
        queries=queries, limit=limit, similarity_threshold=0.5, budget_per_query=1
    )

    assert len(results) <= limit
    if results:
        procedures = [r.procedure for r in results]
        print(f"(Embedding) Ordered ranked results for queries {queries}: {procedures}")
        # With budget_per_query=1, and given the queries,
        # CreateEvent (from query 1) and ListEvents (from query 2) are strong distinct candidates.
        assert "CreateEvent" in procedures
        assert "ListEvents" in procedures
        # Ensure unique tools
        assert len(procedures) == len(set(procedures))


@pytest.mark.asyncio
async def test_hybrid_retriever_client_integration_initialization_and_retrieve(
    tools_with_embeddings_file: str,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
    schema_fields_to_index: List[str],
) -> None:
    """
    Integration test for HybridRetriever initialization and the retrieve method
    using a real embedder and pre-computed embeddings.
    """
    retriever = HybridRetriever(
        tools_data_path=tools_with_embeddings_file,
        model_class=ToolSchema,
        bm25_fields=schema_fields_to_index,
        embedder_func=real_embedder.get_batch_embedding,
    )

    assert len(retriever.tools_with_embeddings) > 0
    # We expect as many tools as were successfully processed (had descriptions and got embeddings)
    # This count needs to be accurate based on the sample_tools_data and embedding logic.
    # All 4 sample tools have descriptions, so all should be included.
    assert len(retriever.tools_with_embeddings) == len(sample_tools_data)
    assert retriever.embedding_retriever.tool_embeddings_matrix.shape[0] == len(
        sample_tools_data
    )
    assert (
        retriever.embedding_retriever.tool_embeddings_matrix.shape[1]
        == TEST_EMBEDDING_DIM
    )

    # Test retrieve method
    query = "Any way to check my schedule?"
    # The embedder_func in RetrieverClient is used for the query,
    # and then compared against tool_embeddings_matrix.
    retrieved_tools = await retriever.retrieve(
        queries=[query], limit=2, similarity_threshold=0.5
    )

    assert len(retrieved_tools) > 0
    assert len(retrieved_tools) <= 2

    # Check if the top retrieved tool is relevant
    top_tool = retrieved_tools[0]
    assert isinstance(top_tool, ToolSchema)
    print(
        f"(Hybrid) Query: '{query}' -> Top retrieved tool: '{top_tool.procedure}' ({top_tool.description})"
    )

    # Expect "ListEvents" to be the most relevant for the query
    assert top_tool.procedure == "ListEvents"

    query_send_email = "send a message to my friend"
    retrieved_email_tools = await retriever.retrieve(
        queries=[query_send_email], limit=1, similarity_threshold=0.5
    )
    assert len(retrieved_email_tools) == 1
    assert retrieved_email_tools[0].procedure == "SendEmail"
    print(
        f"(Hybrid) Query: '{query_send_email}' -> Top retrieved tool: '{retrieved_email_tools[0].procedure}' ({retrieved_email_tools[0].description})"
    )


@pytest.mark.asyncio
async def test_hybrid_retriever_client_integration_retrieve_ordered_ranked(
    tools_with_embeddings_file: str,
    real_embedder: AbstractEmbeddingModel,
    sample_tools_data: List[ToolSchema],
    schema_fields_to_index: List[str],
) -> None:
    """
    Integration test for the retrieve_ordered_ranked method.
    """
    retriever = HybridRetriever(
        tools_data_path=tools_with_embeddings_file,
        model_class=ToolSchema,
        bm25_fields=schema_fields_to_index,
        embedder_func=real_embedder.get_batch_embedding,
    )

    queries = [
        "How do I make a new calendar event?",  # Should strongly match CreateEvent
        "Any way to check my schedule?",  # Should match ListEvents or CreateEvent
    ]
    limit = 2
    # budget_per_query means how many top candidates from each query's individual search are considered
    # for the final re-ranking/selection.
    results = await retriever.retrieve_ordered_ranked(
        queries=queries, limit=limit, similarity_threshold=0.5, budget_per_query=1
    )

    assert len(results) <= limit
    if results:
        procedures = [r.procedure for r in results]
        print(f"(Hybrid) Ordered ranked results for queries {queries}: {procedures}")
        # With budget_per_query=1, and given the queries,
        # CreateEvent (from query 1) and ListEvents (from query 2) are strong distinct candidates.
        assert "CreateEvent" in procedures
        assert "ListEvents" in procedures
        # Ensure unique tools
        assert len(procedures) == len(set(procedures))
