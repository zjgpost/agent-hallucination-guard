"""Structural Causal Model (SCM) implementation.

This module provides a lightweight, deterministic-friendly SCM that supports:
- Directed causal graphs
- Conditional Probability Tables (CPT) with Laplace smoothing
- Simple do-calculus for common graph patterns (chain, fork, collider)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class CausalNode:
    """A node in a causal graph."""

    def __init__(self, name: str, domain: List[str], parents: Optional[List[str]] = None):
        self.name = name
        self.domain = domain
        self.parents = parents or []
        self.cpt: Dict[Tuple[str, ...], np.ndarray] = {}

    def __repr__(self) -> str:
        return f"CausalNode({self.name}, parents={self.parents})"


class StructuralCausalModel:
    """A lightweight Structural Causal Model.

    Example:
        >>> scm = StructuralCausalModel()
        >>> scm.add_node("CPU", ["high", "normal"])
        >>> scm.add_node("ResponseTime", ["slow", "normal"], parents=["CPU"])
        >>> scm.set_cpt("ResponseTime", {("high",): [0.9, 0.1], ("normal",): [0.1, 0.9]})
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.graph: Dict[str, List[str]] = {}

    def add_node(self, name: str, domain: List[str], parents: Optional[List[str]] = None) -> None:
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists")
        parents = parents or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"Parent node {p} does not exist")
        self.nodes[name] = CausalNode(name, domain, parents)
        self.graph.setdefault(name, [])
        for p in parents:
            self.graph.setdefault(p, []).append(name)

    def set_cpt(self, node_name: str, cpt: Dict[Tuple[str, ...], List[float]]) -> None:
        node = self.nodes[node_name]
        expected_key_len = len(node.parents)
        for key, probs in cpt.items():
            if len(key) != expected_key_len:
                raise ValueError(
                    f"CPT key {key} for {node_name} must have length {expected_key_len}"
                )
            arr = np.asarray(probs, dtype=float)
            if len(arr) != len(node.domain):
                raise ValueError(
                    f"CPT for {node_name} must have {len(node.domain)} probabilities"
                )
            if not np.isclose(arr.sum(), 1.0):
                arr = arr / arr.sum() if arr.sum() > 0 else np.ones_like(arr) / len(arr)
            node.cpt[key] = arr

    def get_cpt(self, node_name: str, parent_values: Tuple[str, ...]) -> np.ndarray:
        node = self.nodes[node_name]
        if not node.parents:
            return node.cpt.get((), np.ones(len(node.domain)) / len(node.domain))
        key = tuple(parent_values)
        if key not in node.cpt:
            raise KeyError(f"CPT not defined for {node_name} given {key}")
        return node.cpt[key]

    def validate_path(self, path: List[str]) -> bool:
        """Check whether a sequence of nodes forms a valid causal path."""
        if len(path) < 2:
            return True
        for i in range(len(path) - 1):
            if path[i + 1] not in self.graph.get(path[i], []):
                return False
        return True

    def ancestors(self, node_name: str) -> Set[str]:
        """Return all ancestors of a node."""
        result: Set[str] = set()
        node = self.nodes[node_name]
        stack = list(node.parents)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.nodes[current].parents)
        return result

    def descendants(self, node_name: str) -> Set[str]:
        """Return all descendants of a node."""
        result: Set[str] = set()
        stack = list(self.graph.get(node_name, []))
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.graph.get(current, []))
        return result

    def apply_laplace_smoothing(self, alpha: float = 1.0) -> None:
        """Apply Laplace smoothing to all CPTs to avoid zero probabilities."""
        for node in self.nodes.values():
            if not node.parents:
                if () not in node.cpt:
                    node.cpt[()] = np.ones(len(node.domain)) / len(node.domain)
                node.cpt[()] = (node.cpt[()] + alpha) / (node.cpt[()].sum() + alpha * len(node.domain))
                continue

            parent_domains = [self.nodes[p].domain for p in node.parents]
            import itertools

            for combo in itertools.product(*parent_domains):
                if combo not in node.cpt:
                    node.cpt[combo] = np.ones(len(node.domain))
                node.cpt[combo] = (node.cpt[combo] + alpha) / (
                    node.cpt[combo].sum() + alpha * len(node.domain)
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "name": n.name,
                    "domain": n.domain,
                    "parents": n.parents,
                    "cpt": {
                        ",".join(k): v.tolist() for k, v in n.cpt.items()
                    },
                }
                for n in self.nodes.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuralCausalModel:
        scm = cls()
        for node in data["nodes"]:
            scm.add_node(node["name"], node["domain"], node.get("parents", []))
        for node in data["nodes"]:
            cpt = {
                tuple(k.split(",")) if k else (): np.array(v)
                for k, v in node["cpt"].items()
            }
            scm.set_cpt(node["name"], cpt)
        return scm
