from typing import List, Union

import numpy as np
import pytest
import pytest_asyncio

from cer.embedder import (
    AbstractEmbeddingModel,
    GeminiEmbedderConfig,
    GeminiEmbeddingModel,
    OpenAIEmbedderConfig,
    OpenAIEmbeddingModel,
    SentenceTransformerEmbedderConfig,
    SentenceTransformerEmbeddingModel,
    generate_embeddings,
)
from cer.embedder.base import _chunk_text, _get_model_tokenizer_info

from tests.support import get_gemini_credentials, get_openai_key


class MockEmbeddingModel(AbstractEmbeddingModel):
    def __init__(self, model_name: str = "mock_model", emb_dim: int = 10) -> None:
        self.emb_dim = emb_dim
        self.model_name = model_name
        self.tokenizer = (
            self  # Mock tokenizer aspects if needed by _get_model_tokenizer_info
        )
        self.model_max_length = 512  # For SentenceTransformer-like behavior

    async def get_embedding(self, input_data: Union[str, List[str]]) -> List[float]:
        input_list = [input_data] if isinstance(input_data, str) else input_data
        #  Return a dummy embedding based on text length
        return [float(len(input_list[0]) % (i + 1)) for i in range(self.emb_dim)]

    async def get_batch_embedding(
        self, input_data_list: Union[str, List[str]]
    ) -> List[List[float]]:
        input_list = (
            [input_data_list] if isinstance(input_data_list, str) else input_data_list
        )
        return [await self.get_embedding(text) for text in input_list]

    # Mock tokenizer methods if _get_model_tokenizer_info depends on them for this type
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))  # Simple byte list as tokens

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


# --- Test Data ---
query1 = "What is the capital of France?"
query2 = "How does photosynthesis work?"
doc1 = "Paris is the capital and most populous city of France. It is known for the Eiffel Tower."  # Short doc
doc2 = (
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy..."
    * 50
)  # Long doc for chunking
doc3 = "The quick brown fox jumps over the lazy dog." * 100  # Another long doc


@pytest_asyncio.fixture(scope="module")
async def gemini_model() -> GeminiEmbeddingModel:
    credentials = await get_gemini_credentials()
    api_key, service_account_info = (
        credentials.get("api_key"),
        credentials.get("service_account_info"),
    )
    if not service_account_info and not api_key:
        pytest.skip(
            "Vertex AI credentials are not set, skipping Gemini real model tests"
        )
    return GeminiEmbeddingModel(
        config=(
            GeminiEmbedderConfig(
                location="us-central1",
                project=service_account_info.get("project_id"),
                service_account_info=service_account_info,
            )
            if service_account_info
            else GeminiEmbedderConfig(api_key=api_key)
        )
    )


@pytest_asyncio.fixture(scope="module")
async def openai_model() -> OpenAIEmbeddingModel:
    api_key = await get_openai_key()
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set, skipping OpenAI real model tests")
    return OpenAIEmbeddingModel(config=OpenAIEmbedderConfig(api_key=api_key))


@pytest_asyncio.fixture(scope="module")
async def st_model_e5() -> SentenceTransformerEmbeddingModel:
    try:
        return SentenceTransformerEmbeddingModel(
            config=SentenceTransformerEmbedderConfig(
                embedding_model="intfloat/multilingual-e5-large-instruct"
            )
        )
    except Exception as e:
        pytest.skip(f"Failed to load SentenceTransformer model, skipping: {e}")


@pytest.fixture
def mock_st_model() -> MockEmbeddingModel:
    return MockEmbeddingModel(model_name="mock-e5-model", emb_dim=20)


# --- Tests for generate_embeddings with Real Models (if keys are set) ---
@pytest.mark.asyncio
async def test_generate_embeddings_gemini_real(
    gemini_model: GeminiEmbeddingModel,
) -> None:
    print("\n--- Testing Gemini (Real) --- ")
    gemini_q_emb = await generate_embeddings(gemini_model, query1, "query")
    assert np.array(gemini_q_emb).shape[0] == 1
    assert np.array(gemini_q_emb).shape[1] > 0  # Check embedding dim
    print(f"Gemini Query 1 Embedding shape: {np.array(gemini_q_emb).shape}")

    gemini_d_emb = await generate_embeddings(gemini_model, [doc1, doc3], "document")
    assert np.array(gemini_d_emb).shape[0] == 2
    assert np.array(gemini_d_emb).shape[1] == np.array(gemini_q_emb).shape[1]
    print(f"Gemini Docs Embedding shape: {np.array(gemini_d_emb).shape}")


@pytest.mark.asyncio
async def test_generate_embeddings_openai_real(
    openai_model: OpenAIEmbeddingModel,
) -> None:
    print("\n--- Testing OpenAI (Real) --- ")
    openai_q_emb = await generate_embeddings(openai_model, [query1, query2], "query")
    assert np.array(openai_q_emb).shape[0] == 2
    assert np.array(openai_q_emb).shape[1] > 0
    print(f"OpenAI Queries Embedding shape: {np.array(openai_q_emb).shape}")

    openai_d_emb = await generate_embeddings(openai_model, doc3, "document")
    assert np.array(openai_d_emb).shape[0] == 1
    assert np.array(openai_d_emb).shape[1] == np.array(openai_q_emb).shape[1]
    print(f"OpenAI Doc 3 Embedding shape: {np.array(openai_d_emb).shape}")


@pytest.mark.asyncio
async def test_generate_embeddings_st_e5_real(
    st_model_e5: SentenceTransformerEmbeddingModel,
) -> None:
    print("\n--- Testing SentenceTransformer (e5 Real) --- ")
    st_q_emb = await generate_embeddings(
        st_model_e5, query1, "query", task_description="Question answering task"
    )
    assert np.array(st_q_emb).shape[0] == 1
    assert np.array(st_q_emb).shape[1] > 0
    print(f"ST Query 1 Embedding shape: {np.array(st_q_emb).shape}")

    st_d_emb = await generate_embeddings(st_model_e5, [doc1, doc2], "document")
    assert np.array(st_d_emb).shape[0] == 2
    assert np.array(st_d_emb).shape[1] == np.array(st_q_emb).shape[1]
    print(f"ST Docs Embedding shape: {np.array(st_d_emb).shape}")

    if (
        hasattr(st_model_e5, "resize_and_normalize_embedding")
        and st_d_emb
        and st_d_emb[0]
    ):
        resized_normalized = st_model_e5.resize_and_normalize_embedding(
            st_d_emb[0], 128
        )
        assert np.array(resized_normalized).shape[0] == 128
        print(
            f"ST Doc 1 Resized/Normalized shape: {np.array(resized_normalized).shape}"
        )


# --- Tests for generate_embeddings with Mock Model ---
@pytest.mark.asyncio
async def test_generate_embeddings_mock_model(
    mock_st_model: MockEmbeddingModel,
) -> None:
    print("\n--- Testing with Mock Model --- ")
    mock_emb_dim = mock_st_model.emb_dim
    q_emb = await generate_embeddings(mock_st_model, query1, "query")
    assert np.array(q_emb).shape == (1, mock_emb_dim)

    docs_input = [doc1, doc2]  # doc2 is long and should trigger chunking
    d_emb = await generate_embeddings(mock_st_model, docs_input, "document")
    assert np.array(d_emb).shape == (len(docs_input), mock_emb_dim)

    # Test empty input
    empty_emb = await generate_embeddings(mock_st_model, [], "query")
    assert empty_emb == []

    single_empty_str_emb = await generate_embeddings(mock_st_model, "", "document")
    assert np.array(single_empty_str_emb).shape == (1, mock_emb_dim)


@pytest.mark.asyncio
async def test_chunk_text_logic(mock_st_model: MockEmbeddingModel) -> None:
    # This test uses the mock_st_model as it has mock tokenizer methods.
    # The SentenceTransformer path in _get_model_tokenizer_info will be taken.
    tokenizer, max_tokens, _ = _get_model_tokenizer_info(mock_st_model)

    # Max tokens for SentenceTransformer is typically 512. Our mock is also 512.
    # A long text that will surely be chunked.
    # The exact number of chunks depends on the mock tokenizer's encode/decode behavior.
    # The mock tokenizer uses utf-8 bytes, so 'a' * 1000 is 1000 tokens.
    # With max_tokens = 512 and overlap = 50:
    # Chunk 1: tokens 0-511
    # Next start: 512 - 50 = 462
    # Chunk 2: tokens 462 - (462+512 > 1000 ? 999 : 462+511)
    # So, expected 2 chunks for 'a'*1000
    long_text = "a" * 1000
    chunks = _chunk_text(long_text, tokenizer, max_tokens, overlap=50)
    assert len(chunks) > 1
    # A more precise assertion would require knowing the exact mock tokenizer output
    print(f"Number of chunks for 1000 'a's with mock ST: {len(chunks)}")

    short_text = "a" * 100
    chunks_short = _chunk_text(short_text, tokenizer, max_tokens, overlap=50)
    assert len(chunks_short) == 1
    assert chunks_short[0] == short_text

    # Test empty text
    chunks_empty = _chunk_text("", tokenizer, max_tokens, overlap=50)
    assert chunks_empty == []
