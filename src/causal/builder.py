"""Build a StructuralCausalModel from JSON config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .scm import StructuralCausalModel


def from_dict(data: Dict[str, Any]) -> StructuralCausalModel:
    """Build SCM from a dictionary."""
    scm = StructuralCausalModel()

    for node in data["nodes"]:
        scm.add_node(node["name"], node["domain"], node.get("parents", []))

    for node in data["nodes"]:
        raw_cpt = node["cpt"]
        cpt: Dict[Any, Any] = {}
        for key, probs in raw_cpt.items():
            parsed_key = tuple(key.split(",")) if key else ()
            cpt[parsed_key] = probs
        scm.set_cpt(node["name"], cpt)

    scm.apply_laplace_smoothing(alpha=data.get("smoothing", 1.0))
    return scm


def from_json(path: str | Path) -> StructuralCausalModel:
    """Build SCM from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data)
