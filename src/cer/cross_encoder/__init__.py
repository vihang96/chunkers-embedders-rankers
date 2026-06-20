from .base import (
    create_bge_reranker_wrapper,
    create_openai_reranker_wrapper,
    get_reranked_objects,
)
from .client import CrossEncoderClient
from .rrf_reranker_client import RRFRerankerClient

__all__ = [
    'CrossEncoderClient',
    'RRFRerankerClient',
    'create_openai_reranker_wrapper',
    'create_bge_reranker_wrapper',
    'get_reranked_objects',
]

# Optional backends. cer[local] for BGE, cer[openai] for the OpenAI reranker.
try:
    from .bge_reranker_client import BGERerankerClient
except ImportError:
    pass
else:
    __all__ += ['BGERerankerClient']

try:
    from .openai_reranker_client import LLMConfig, OpenAIRerankerClient
except ImportError:
    pass
else:
    __all__ += ['OpenAIRerankerClient', 'LLMConfig']
