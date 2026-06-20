import re
from typing import Type

import pytest

from cer.cross_encoder.rrf_reranker_client import RRFRerankerClient


class MockObject:
    def __init__(self, id_val: str, value: int):
        self.id_val = id_val
        self.value = value

    def __repr__(self) -> str:
        return f"MockObject(id_val={self.id_val}, value={self.value})"

    # Need __eq__ for list comparison in asserts
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockObject):
            return NotImplemented
        return self.id_val == other.id_val and self.value == other.value


@pytest.fixture
def rrf_client() -> Type[RRFRerankerClient]:
    return RRFRerankerClient


def test_rrf_empty_results(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with empty results list."""
    assert rrf_client.rrf([], "id") == []


def test_rrf_single_list_dict(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with a single list of dictionaries."""
    results = [[{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]]
    expected = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
    assert rrf_client.rrf(results, "id") == expected


def test_rrf_multiple_lists_dict(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with multiple lists of dictionaries, checking fusion logic."""
    results = [
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],  # a:1, b:0.5, c:0.33
        [{"id": "b"}, {"id": "a"}, {"id": "d"}],  # b:1, a:0.5, d:0.33
    ]
    # Scores: a = 1/1 + 1/2 = 1.5
    #         b = 1/2 + 1/1 = 1.5
    #         c = 1/3         ~ 0.33
    #         d = 1/3         ~ 0.33
    # Expected order: a and b (tie, order depends on insertion/stability), then c and d (tie)
    # The current implementation of rrf is stable for equal scores based on first seen
    # a is seen first from the first list. b is seen second from the first list.
    # In scores dict, 'a' will likely appear before 'b' if using standard dicts for Py3.7+
    # sorted() is stable. object_map preserves the first object seen.

    # Let's trace object_map and scores:
    # List 1:
    # obj {"id":"a"}, i=0: scores["a"] = 1/1 = 1. object_map["a"] = {"id":"a"}
    # obj {"id":"b"}, i=1: scores["b"] = 1/2 = 0.5. object_map["b"] = {"id":"b"}
    # obj {"id":"c"}, i=2: scores["c"] = 1/3 ~ 0.33. object_map["c"] = {"id":"c"}
    # List 2:
    # obj {"id":"b"}, i=0: scores["b"] = 0.5 + 1/1 = 1.5.
    # obj {"id":"a"}, i=1: scores["a"] = 1 + 1/2 = 1.5.
    # obj {"id":"d"}, i=2: scores["d"] = 1/3 ~ 0.33. object_map["d"] = {"id":"d"}
    # scores: {"a": 1.5, "b": 1.5, "c": 0.33, "d": 0.33}
    # sorted_items: [("a", 1.5), ("b", 1.5), ("c", 0.33), ("d", 0.33)] (assuming a before b due to insertion order for keys in scores dict for Python 3.7+)
    # If keys are sorted alphabetically for some reason before sorting by score: [('a', 1.5), ('b', 1.5), ('c', 0.333), ('d', 0.333)]
    # The `sorted` on scores.items() is stable. If a and b have the same score, their relative order from `scores.items()` is preserved.
    # For dicts in Python 3.7+, insertion order is preserved.
    # So, since 'a' was updated to 1.5 before 'b' was, it might appear first.
    # Let's check: a's final score update happens when it's processed in the second list. b's final score update happens when it's processed in the second list. 'b' is processed before 'a' in the second list.
    # So, after list 1: scores = {'a': 1.0, 'b': 0.5, 'c': 0.33}
    # Processing list 2:
    #   item 'b': scores['b'] becomes 0.5 + 1.0 = 1.5
    #   item 'a': scores['a'] becomes 1.0 + 0.5 = 1.5
    #   item 'd': scores['d'] becomes 0.33
    # So scores dict items (before sorting by value): [('a',1.5), ('b',1.5), ('c',0.33), ('d',0.33)] (order from first list, then new items from second)
    # Then sorted by score (desc): [('a', 1.5), ('b', 1.5), ('c', 0.33), ('d', 0.33)] or [('b', 1.5), ('a', 1.5), ('c', 0.33), ('d', 0.33)]
    # Since sorted is stable, the original order of items with equal scores is preserved.
    # The order of items() from a dict is insertion order (for Python 3.7+).
    # 'a' was inserted. Then 'b'. Then 'c'. Then 'd'. So items() will be [('a',1.5), ('b',1.5), ('c',0.33), ('d',0.33)]
    # So output: a, b, c, d

    expected = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    actual = rrf_client.rrf(results, "id")
    assert actual == expected


def test_rrf_with_rank_const(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with a non-default rank_const."""
    results = [
        [{"id": "a"}, {"id": "b"}],  # a: 1/(0+60)=0.0166, b: 1/(1+60)=0.0163
        [{"id": "b"}, {"id": "a"}],  # b: 1/(0+60)=0.0166, a: 1/(1+60)=0.0163
    ]
    # k = 60
    # Scores: a = 1/(0+60) + 1/(1+60) = 1/60 + 1/61 = (61+60)/(60*61) = 121/3660 = 0.03306
    #         b = 1/(1+60) + 1/(0+60) = 1/61 + 1/60 = 121/3660 = 0.03306
    # Expected order: a, b (due to stability as 'a' is encountered first in the first list)
    expected = [{"id": "a"}, {"id": "b"}]
    assert rrf_client.rrf(results, "id", rank_const=60) == expected


def test_rrf_with_min_score(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with a min_score that filters results."""
    results = [
        [{"id": "a"}, {"id": "b"}],  # a:1, b:0.5
        [{"id": "c"}, {"id": "a"}],  # c:1, a:0.5
    ]
    # Scores: a = 1 + 0.5 = 1.5
    #         b = 0.5
    #         c = 1
    # Expected order: a, c, b. If min_score = 0.8, then b is filtered out.
    expected = [{"id": "a"}, {"id": "c"}]
    assert rrf_client.rrf(results, "id", min_score=0.8) == expected


def test_rrf_custom_objects(rrf_client: RRFRerankerClient) -> None:
    """Test RRF with custom objects instead of dictionaries."""
    obj_a = MockObject("a", 100)
    obj_b = MockObject("b", 200)
    obj_c = MockObject("c", 300)
    results = [[obj_a, obj_b], [obj_b, obj_c]]  # a:1, b:0.5  # b:1, c:0.5
    # Scores: a = 1
    #         b = 0.5 + 1 = 1.5
    #         c = 0.5
    # Expected order: b, a, c
    expected = [obj_b, obj_a, obj_c]
    assert rrf_client.rrf(results, "id_val") == expected


def test_rrf_field_not_found_dict(rrf_client: RRFRerankerClient) -> None:
    """Test RRF when field_name is not in a dictionary object."""
    results = [[{"name": "a"}]]
    with pytest.raises(
        ValueError,
        match=re.escape("Field 'id' not found in object (dict): {'name': 'a'}"),
    ):
        rrf_client.rrf(results, "id")


def test_rrf_field_not_found_object(rrf_client: RRFRerankerClient) -> None:
    """Test RRF when field_name is not an attribute of an object."""
    results = [[MockObject("a", 1)]]
    with pytest.raises(
        ValueError,
        match="Field 'non_existent_field' not found in object",
    ):
        rrf_client.rrf(results, "non_existent_field")


def test_rrf_mixed_object_types_with_same_field_name(
    rrf_client: RRFRerankerClient,
) -> None:
    """Test RRF with mixed object types (dict and custom) having the same field name."""
    obj_a = MockObject("a", 100)
    dict_b = {"id_val": "b", "data": 200}
    obj_c = MockObject("c", 300)

    results = [
        [obj_a, dict_b],  # obj_a: 1, dict_b: 0.5
        [dict_b, obj_c],  # dict_b: 1, obj_c: 0.5
    ]
    # Scores:
    # obj_a: 1
    # dict_b: 0.5 (from list1) + 1 (from list2) = 1.5
    # obj_c: 0.5
    # Expected order: dict_b, obj_a, obj_c

    expected_output = [dict_b, obj_a, obj_c]
    actual_output = rrf_client.rrf(results, "id_val")
    assert actual_output == expected_output
