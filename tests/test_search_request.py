import pytest
from pydantic import ValidationError

from app.core.schemas import SearchRequest


@pytest.mark.parametrize("top_k", [-1, -5, -100])
def test_negative_result_counts_are_rejected(top_k):
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="a chart", top_k=top_k)

    assert exc_info.value.errors()[0]["loc"] == ("top_k",)


@pytest.mark.parametrize("top_k", [0, 1, 5, 100])
def test_nonnegative_result_counts_are_preserved(top_k):
    assert SearchRequest(query="a chart", top_k=top_k).top_k == top_k


def test_default_result_count_is_five():
    assert SearchRequest(query="a chart").top_k == 5
