# chunkers-embedders-rankers (`cer`)

Reusable building blocks for retrieval pipelines: document **chunking**, text
**embedding** (OpenAI, Gemini, SentenceTransformers), **retrieval** (BM25,
embedding, hybrid), and **reranking** (BGE cross-encoder, LLM/OpenAI, RRF).

## Installation

The base install is intentionally lightweight (numpy, pydantic, pyyaml,
tiktoken). Heavy backends are opt-in via extras, so you only pull what you use:

| Extra | Enables | Notable deps |
|---|---|---|
| `openai` | OpenAI embeddings + reranker | `openai` |
| `gemini` | Gemini / Vertex AI embeddings | `google-genai`, `google-auth` |
| `local` | SentenceTransformer embeddings + BGE reranker | `sentence-transformers` (torch) |
| `chunking` | Markdown/document chunking | `chunknorris` |
| `retrieval` | BM25 / embedding / hybrid retrieval + similarity | `bm25s`, `scikit-learn`, `nltk` |
| `all` | Everything above | — |

From a git remote (pick the extras you need):

```bash
uv add "chunkers-embedders-rankers[openai,retrieval] @ git+https://github.com/your-org/chunkers-embedders-rankers.git"
# everything:
uv add "chunkers-embedders-rankers[all] @ git+https://github.com/your-org/chunkers-embedders-rankers.git"
# pin a tag/commit:
uv add "chunkers-embedders-rankers[all] @ git+https://github.com/your-org/chunkers-embedders-rankers.git@v0.1.0"
```

For local development (editable, with all extras):

```bash
uv add --editable "/path/to/chunkers-embedders-rankers[all]"
```

> Importing a subpackage whose extra isn't installed (e.g. `cer.chunker` without
> `chunking`) will surface as the relevant symbol being unavailable. Install the
> matching extra to enable it.

Or build a wheel and install it anywhere:

```bash
uv build                       # -> dist/chunkers_embedders_rankers-0.1.0-py3-none-any.whl
pip install dist/chunkers_embedders_rankers-0.1.0-py3-none-any.whl
```

## Usage

```python
from cer.chunker import CustomMarkdownChunker
from cer.embedder import OpenAIEmbeddingModel, generate_embeddings
from cer.retriever import EmbeddingRetriever, HybridRetriever, BM25Retriever
from cer.cross_encoder import BGERerankerClient, OpenAIRerankerClient
```
