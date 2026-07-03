"""Feedback loop: self-reflection and memory update."""

from __future__ import annotations

from typing import Any, Dict, List


class FeedbackLoop:
    """Fourth-line defense: close the loop by reflecting on errors.

    When an error or hallucination is detected, generate a reflection record
    that can be stored in long-term memory to avoid future repetition.
    """

    def __init__(self, long_term_memory: Optional[Any] = None):
        self.long_term_memory = long_term_memory
        self.error_patterns: List[Dict[str, Any]] = []

    def reflect(self, query: str, answer: str, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a reflection given detected failures."""
        if not failures:
            return {"action": "none", "reason": "No failures detected."}

        pattern = {
            "query_type": self._classify_query(query),
            "failure_types": [f["type"] for f in failures],
            "lessons": [f"Avoid {f['type']}: {f.get('detail', '')}" for f in failures],
        }
        self.error_patterns.append(pattern)

        if self.long_term_memory is not None:
            self.long_term_memory.store(pattern)

        return {
            "action": "store_pattern",
            "pattern": pattern,
            "summary": f"Stored {len(failures)} failure patterns for {pattern['query_type']} queries.",
        }

    @staticmethod
    def _classify_query(query: str) -> str:
        lowered = query.lower()
        if any(k in lowered for k in ["cpu", "memory", "disk", "response", "server"]):
            return "ops_diagnosis"
        if any(k in lowered for k in ["why", "cause", "reason"]):
            return "causal_explanation"
        return "general"

    def has_similar_pattern(self, query: str) -> bool:
        """Check whether a similar failure pattern has been seen before."""
        if not self.error_patterns:
            return False
        query_type = self._classify_query(query)
        return any(p["query_type"] == query_type for p in self.error_patterns)
