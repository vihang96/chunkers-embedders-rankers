from .base import generate_embeddings, generate_object_embeddings
from .client import AbstractEmbeddingModel, EmbedderConfig, normalize_embedding

__all__ = [
    "AbstractEmbeddingModel",
    "EmbedderConfig",
    "normalize_embedding",
    "generate_embeddings",
    "generate_object_embeddings",
]

# Optional backends. Each requires its own extra (cer[openai], cer[gemini],
# cer[local]); skip cleanly when the backing library isn't installed.
try:
    from .openai import OpenAIEmbedderConfig, OpenAIEmbeddingModel
except ImportError:
    pass
else:
    __all__ += ["OpenAIEmbeddingModel", "OpenAIEmbedderConfig"]

try:
    from .gemini import GeminiEmbedderConfig, GeminiEmbeddingModel
except ImportError:
    pass
else:
    __all__ += ["GeminiEmbeddingModel", "GeminiEmbedderConfig"]

try:
    from .sentence_transformer import (
        SentenceTransformerEmbedderConfig,
        SentenceTransformerEmbeddingModel,
    )
except ImportError:
    pass
else:
    __all__ += [
        "SentenceTransformerEmbeddingModel",
        "SentenceTransformerEmbedderConfig",
    ]
