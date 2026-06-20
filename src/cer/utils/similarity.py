"""
Utility functions for calculating similarity between ToolSchema documents.
"""

from typing import List, Optional

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from ..embedder import AbstractEmbeddingModel, generate_embeddings
from .schemas import PydanticModel
from .yaml_helpers import convert_pydantic_to_yaml


async def calculate_cosine_similarity_from_texts(
    text1: str, text2: str, embedder: AbstractEmbeddingModel
) -> float:
    """
    Calculates cosine similarity between two texts using the provided embedder.
    """
    if not text1.strip() or not text2.strip():
        return 0.0  # Similarity is 0 if one text is empty
    if text1 == text2:
        return 1.0

    try:
        embeddings = await generate_embeddings(embedder, [text1, text2], "tool")
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return 0.0  # Return neutral similarity on error

    return float(cosine_similarity([embedding1], [embedding2])[0][0])


def calculate_bleu_score(reference_text: str, candidate_text: str) -> float:
    """
    Calculates BLEU score between a candidate text and a reference text.
    Uses NLTK if available, otherwise returns a placeholder value.
    """
    if not reference_text.strip() or not candidate_text.strip():
        return 0.0  # Score is 0 if one text is empty
    if reference_text == candidate_text:
        return 1.0

    # NLTK expects a list of reference token lists, and a candidate token list
    reference_tokens: List[List[str]] = [
        reference_text.lower().split()
    ]  # Single reference
    candidate_tokens: List[str] = candidate_text.lower().split()

    # Using smoothing function to avoid 0 scores for short sentences or no n-gram overlaps
    smoother = SmoothingFunction()
    try:
        score = sentence_bleu(
            references=reference_tokens,
            hypothesis=candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),  # Standard BLEU-4 weights
            smoothing_function=smoother.method1,  # Common smoothing method
        )
        return float(score)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0  # Return low similarity on error


async def calculate_combined_similarity(
    doc1: PydanticModel,
    doc2: PydanticModel,
    embedder: Optional[AbstractEmbeddingModel] = None,
    cosine_weight: float = 0.5,
    bleu_weight: float = 0.5,
) -> float:
    """
    Calculates a combined similarity score (average of cosine and BLEU)
    between two ToolSchema documents.
    """
    text1 = convert_pydantic_to_yaml(doc1)
    text2 = convert_pydantic_to_yaml(doc2)

    if (
        not text1.strip() or not text2.strip()
    ):  # Handle cases where YAML conversion results in empty string
        return 0.0 if text1.strip() != text2.strip() else 1.0

    if text1 == text2:  # If YAML strings are identical, similarity is 1.0
        return 1.0

    if embedder:
        cosine_sim = await calculate_cosine_similarity_from_texts(
            text1, text2, embedder
        )
    else:
        cosine_sim = 0.0
    bleu = calculate_bleu_score(text1, text2)  # text1 as reference, text2 as candidate

    # Weighted average
    if cosine_weight + bleu_weight == 0:  # Avoid division by zero if both weights are 0
        return 0.0

    combined_score = ((cosine_sim * cosine_weight) + (bleu * bleu_weight)) / (
        cosine_weight + bleu_weight
    )
    return float(combined_score)
