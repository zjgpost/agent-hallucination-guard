"""Tests for causal reasoner."""

import pytest

from causal.builder import from_dict
from causal.causal_reasoner import CausalReasoner


@pytest.fixture
def reasoner():
    data = {
        "smoothing": 1.0,
        "nodes": [
            {"name": "A", "domain": ["high", "normal"], "parents": [], "cpt": {"": [0.5, 0.5]}},
            {"name": "B", "domain": ["high", "normal"], "parents": ["A"], "cpt": {"high": [0.9, 0.1], "normal": [0.1, 0.9]}},
            {"name": "C", "domain": ["high", "normal"], "parents": ["B"], "cpt": {"high": [0.8, 0.2], "normal": [0.2, 0.8]}},
        ],
    }
    scm = from_dict(data)
    return CausalReasoner(scm)


def test_valid_path(reasoner):
    valid, nodes, reason = reasoner.validate_thought("A is high, so B is high, then C is high")
    assert valid
    assert nodes == ["A", "B", "C"]
    assert "valid causal path" in reason


def test_invalid_jump(reasoner):
    valid, nodes, reason = reasoner.validate_thought("C is high, so A is high")
    assert not valid
    assert nodes == ["C", "A"]
    assert "Invalid causal jump" in reason


def test_most_probable_effect(reasoner):
    result = reasoner.most_probable_effect({"A": "high"}, "C")
    assert result == "high"


def test_list_valid_paths(reasoner):
    paths = reasoner.list_valid_paths("A", "C")
    assert paths == [["A", "B", "C"]]


def test_infer_root_cause():
    data = {
        "smoothing": 0.0,
        "nodes": [
            {"name": "CPU", "domain": ["high", "normal"], "parents": [], "cpt": {"": [0.5, 0.5]}},
            {"name": "Memory", "domain": ["high", "normal"], "parents": [], "cpt": {"": [0.5, 0.5]}},
            {"name": "ResponseTime", "domain": ["slow", "normal"], "parents": ["CPU", "Memory"], "cpt": {"high,high": [0.9, 0.1], "high,normal": [0.6, 0.4], "normal,high": [0.6, 0.4], "normal,normal": [0.1, 0.9]}},
            {"name": "RootCause", "domain": ["cpu_bound", "memory_bound", "io_bound", "normal"], "parents": ["CPU", "Memory", "ResponseTime"], "cpt": {"high,normal,slow": [0.85, 0.1, 0.03, 0.02], "normal,high,slow": [0.1, 0.82, 0.05, 0.03], "high,high,slow": [0.45, 0.45, 0.08, 0.02], "normal,normal,slow": [0.1, 0.1, 0.75, 0.05], "high,high,normal": [0.35, 0.35, 0.25, 0.05], "normal,normal,normal": [0.05, 0.05, 0.05, 0.85]}},
        ],
    }
    reasoner = CausalReasoner(from_dict(data))
    cause, prob = reasoner.infer_root_cause(
        {"CPU": "normal", "Memory": "normal", "ResponseTime": "slow"}
    )
    assert cause == "io_bound"
    assert prob > 0.7


def test_shared_effect():
    data = {
        "smoothing": 1.0,
        "nodes": [
            {"name": "X", "domain": ["high", "normal"], "parents": [], "cpt": {"": [0.5, 0.5]}},
            {"name": "Y", "domain": ["high", "normal"], "parents": [], "cpt": {"": [0.5, 0.5]}},
            {"name": "Z", "domain": ["high", "normal"], "parents": ["X", "Y"], "cpt": {"high,high": [0.9, 0.1], "high,normal": [0.7, 0.3], "normal,high": [0.7, 0.3], "normal,normal": [0.1, 0.9]}},
        ],
    }
    reasoner = CausalReasoner(from_dict(data))
    valid, nodes, reason = reasoner.validate_thought("X and Y cause Z")
    assert valid
    assert "effect" in reason
