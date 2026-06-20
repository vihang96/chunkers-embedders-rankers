from typing import List

from ..utils.schemas import PydanticModel, RerankerCallable
from ..utils.yaml_helpers import convert_pydantic_to_yaml, convert_yaml_to_pydantic
from .client import CrossEncoderClient


async def get_reranked_objects(
    query: str,
    objects: List[PydanticModel],
    model: CrossEncoderClient,
) -> List[PydanticModel]:
    object_type = objects[0].__class__
    """Reranks a list of objects using a cross-encoder model."""
    reranked_objects = await model.rank(
        query, [convert_pydantic_to_yaml(obj) for obj in objects]
    )
    reranked_models = [
        convert_yaml_to_pydantic(
            string,
            object_type,
        )
        for (string, _) in reranked_objects
    ]
    return [model for model in reranked_models if model is not None]


def create_bge_reranker_wrapper(
    bge_model_client: CrossEncoderClient,
) -> RerankerCallable:  # type: ignore
    """
    Creates a reranker function compatible with EmbeddingRetriever,
    using a pre-configured BGE CrossEncoderClient.

    Args:
        bge_model_client: An instance of CrossEncoderClient configured for BGE.

    Returns:
        An async function that adheres to the RerankerCallable signature.
    """

    async def bge_reranker(
        query: str,
        objects: List[PydanticModel],
    ) -> List[PydanticModel]:
        """
        Reranks PydanticModel objects using the provided BGE cross-encoder client.
        """
        if not objects:
            return []

        object_type = objects[0].__class__
        reranked_objects = await bge_model_client.rank(
            query, [convert_pydantic_to_yaml(obj) for obj in objects]
        )
        reranked_models = [
            convert_yaml_to_pydantic(
                string,
                object_type,
            )
            for (string, _) in reranked_objects
        ]
        return [model for model in reranked_models if model is not None]

    return bge_reranker


def create_openai_reranker_wrapper(
    openai_model_client: CrossEncoderClient,
) -> RerankerCallable:  # type: ignore
    """
    Creates a reranker function compatible with EmbeddingRetriever,
    using a pre-configured OpenAI CrossEncoderClient.

    Args:
        openai_model_client: An instance of CrossEncoderClient configured for OpenAI.

    Returns:
        An async function that adheres to the RerankerCallable signature.
    """

    async def openai_reranker(
        query: str,
        objects: List[PydanticModel],
    ) -> List[PydanticModel]:
        """
        Reranks PydanticModel objects using the provided OpenAI cross-encoder client.
        """
        if not objects:
            return []

        object_type = objects[0].__class__
        reranked_objects = await openai_model_client.rank(
            query, [convert_pydantic_to_yaml(obj) for obj in objects]
        )
        reranked_models = [
            convert_yaml_to_pydantic(
                string,
                object_type,
            )
            for (string, _) in reranked_objects
        ]
        return [model for model in reranked_models if model is not None]

    return openai_reranker
