from collections import defaultdict
from typing import Any, TypeVar

ObjectType = TypeVar("ObjectType")


class RRFRerankerClient:
    """
    A client for performing Reciprocal Rank Fusion (RRF) reranking.
    """

    @staticmethod
    def rrf(
        results: list[list[ObjectType]],
        field_name: str,
        rank_const: int = 1,
        min_score: float = 0.0,
    ) -> list[ObjectType]:
        """
        Performs Reciprocal Rank Fusion (RRF) on a list of ranked lists of objects.

        Args:
            results: A list of lists, where each inner list is a ranking of objects.
            field_name: The name of the attribute/key in the objects to use for identification and scoring.
            rank_const: The constant k in the RRF formula (default is 1).
            min_score: The minimum score for an object to be included in the final ranking (default is 0.0).

        Returns:
            A list of objects, reranked using RRF.
        """
        scores: dict[Any, float] = defaultdict(float)
        object_map: dict[Any, ObjectType] = {}

        for result_list in results:
            for i, obj in enumerate(result_list):
                try:
                    # Attempt to access the field as an attribute
                    key = getattr(obj, field_name)
                except AttributeError:
                    # If not an attribute, try to access as a dictionary key
                    if isinstance(obj, dict):
                        try:
                            key = obj[field_name]
                        except KeyError:
                            raise ValueError(
                                f"Field '{field_name}' not found in object (dict): {obj}"
                            ) from None
                    else:
                        raise ValueError(
                            f"Field '{field_name}' not found in object (neither attribute nor dict key): {obj}"
                        ) from None

                scores[key] += 1 / (i + rank_const)
                if key not in object_map:
                    object_map[key] = obj

        # Sort by score in descending order
        scored_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Filter by min_score and return the original objects in the new order
        final_reranked_list: list[ObjectType] = []
        for key, score in scored_items:
            if score >= min_score:
                final_reranked_list.append(object_map[key])

        return final_reranked_list
