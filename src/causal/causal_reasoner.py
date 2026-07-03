"""Causal reasoner that validates agent thoughts against causal paths."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from .scm import StructuralCausalModel


class CausalReasoner:
    """Validate whether an agent's reasoning follows valid causal paths.

    The reasoner extracts mentioned causal nodes from a thought string and checks
    whether they form a valid causal chain/fork/collider in the SCM.
    """

    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def extract_nodes(self, text: str) -> List[str]:
        """Extract known causal nodes from text in the order they appear."""
        matches = []
        for node_name in self.scm.nodes:
            m = re.search(r"\b" + re.escape(node_name) + r"\b", text)
            if m:
                matches.append((m.start(), node_name))
        matches.sort(key=lambda x: x[0])
        return [name for _, name in matches]

    def validate_thought(self, thought: str) -> Tuple[bool, List[str], str]:
        """Validate a single thought against the causal model.

        Returns:
            (is_valid, extracted_nodes, reason)
        """
        nodes = self.extract_nodes(thought)
        if len(nodes) < 2:
            return True, nodes, "Too few causal concepts to validate."

        if self.scm.validate_path(nodes):
            return True, nodes, "Thought follows a valid causal path."

        # Check every adjacent pair has a valid edge (covers multiple local edges)
        all_adjacent_valid = True
        for i in range(len(nodes) - 1):
            if not self.scm.validate_path([nodes[i], nodes[i + 1]]):
                all_adjacent_valid = False
                break
        if all_adjacent_valid:
            return True, nodes, "Thought follows supported local causal edges."

        # Allow fork structures where the first-mentioned node is a common ancestor
        # that can reach all other mentioned nodes (e.g., "Load causes CPU and Memory")
        candidate = nodes[0]
        descendants = self.scm.descendants(candidate)
        if all(n == candidate or n in descendants for n in nodes):
            return True, nodes, f"Thought follows a fork from common cause '{candidate}'."

        # Allow shared-effect explanations where a mentioned effect is explained by
        # one or more of its causes (e.g., "CPU and Memory cause ResponseTime" or
        # "ResponseTime is slow because of CPU and Memory").
        for effect in nodes:
            effect_node = self.scm.nodes.get(effect)
            if effect_node is None:
                continue
            effect_parents = set(effect_node.parents)
            other_nodes = [n for n in nodes if n != effect]
            if not other_nodes:
                continue
            if len([n for n in other_nodes if n in effect_parents]) >= 1 and all(
                n in effect_parents or n in self.scm.ancestors(effect) for n in other_nodes
            ):
                return (
                    True,
                    nodes,
                    f"Thought explains effect '{effect}' via its causes.",
                )

        return (
            False,
            nodes,
            f"Invalid causal jump: mentioned nodes {nodes} do not form a supported causal structure in the SCM.",
        )

    def suggest_correction(self, thought: str) -> str:
        """Return a hint for why the thought violates the causal model."""
        valid, nodes, reason = self.validate_thought(thought)
        if valid:
            return ""
        return reason

    def most_probable_effect(self, intervention: Dict[str, str], target: str) -> str:
        """Simplified do-calculus: given an intervention, predict target outcome.

        Only supports one-step and two-step causal chains currently.
        """
        if target not in self.scm.nodes:
            raise ValueError(f"Target {target} not in SCM")

        # Direct parent intervention
        for parent, value in intervention.items():
            if parent in self.scm.nodes[target].parents:
                cpt = self.scm.get_cpt(target, (value,))
                return self.scm.nodes[target].domain[int(cpt.argmax())]

        # Two-step: parent -> mediator -> target
        target_parents = self.scm.nodes[target].parents
        for parent, value in intervention.items():
            for mediator in target_parents:
                if parent in self.scm.nodes[mediator].parents:
                    mediator_cpt = self.scm.get_cpt(mediator, (value,))
                    mediator_value = self.scm.nodes[mediator].domain[int(mediator_cpt.argmax())]
                    target_cpt = self.scm.get_cpt(target, (mediator_value,))
                    return self.scm.nodes[target].domain[int(target_cpt.argmax())]

        return "unknown"

    def infer_root_cause(self, evidence: Dict[str, str]) -> Tuple[str, float]:
        """Infer the most likely root cause given observed evidence.

        Uses exact enumeration over the SCM and returns the most probable
        RootCause value along with its probability. Falls back to the target
        node named 'RootCause' if present, otherwise the first leaf node.
        """
        target = "RootCause" if "RootCause" in self.scm.nodes else None
        if target is None:
            # Pick a sink node as a default target.
            for name, node in self.scm.nodes.items():
                if not self.scm.graph.get(name):
                    target = name
                    break
        if target is None:
            raise ValueError("SCM has no identifiable target/root-cause node")

        # Topologically sort nodes (parents before children).
        order: List[str] = []
        visited: Set[str] = set()

        def dfs(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in self.scm.nodes[node].parents:
                dfs(parent)
            order.append(node)

        for name in self.scm.nodes:
            dfs(name)

        evidence = {k: v for k, v in evidence.items() if k in self.scm.nodes}
        target_domain = self.scm.nodes[target].domain
        counts = {val: 0.0 for val in target_domain}

        def backtrack(idx: int, assignment: Dict[str, str], weight: float) -> None:
            if idx == len(order):
                counts[assignment[target]] += weight
                return
            node = order[idx]
            parent_values = tuple(assignment[p] for p in self.scm.nodes[node].parents)
            cpt = self.scm.get_cpt(node, parent_values)
            if node in evidence:
                value = evidence[node]
                domain_idx = self.scm.nodes[node].domain.index(value)
                backtrack(idx + 1, {**assignment, node: value}, weight * float(cpt[domain_idx]))
            else:
                for value in self.scm.nodes[node].domain:
                    domain_idx = self.scm.nodes[node].domain.index(value)
                    backtrack(idx + 1, {**assignment, node: value}, weight * float(cpt[domain_idx]))

        backtrack(0, {}, 1.0)
        total = sum(counts.values())
        if total == 0:
            return "unknown", 0.0
        probs = {k: v / total for k, v in counts.items()}
        best = max(probs, key=probs.get)
        return best, probs[best]

    def list_valid_paths(self, start: str, end: str, max_depth: int = 4) -> List[List[str]]:
        """Enumerate valid causal paths from start to end."""
        results: List[List[str]] = []

        def dfs(current: str, path: List[str]) -> None:
            if len(path) > max_depth:
                return
            if current == end:
                results.append(path[:])
                return
            for child in self.scm.graph.get(current, []):
                if child not in path:
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(start, [start])
        return results
