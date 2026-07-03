"""Output guard: fact-checking and consistency verification."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class OutputGuard:
    """Third-line defense at the output layer.

    In a full system this would call a RAG knowledge base; here we provide:
    - A pluggable fact-check callback
    - Self-consistency checks (e.g., contradiction detection)
    - Sensitive keyword filtering
    """

    def __init__(
        self,
        fact_checker: Optional[Any] = None,
        sensitive_patterns: Optional[List[str]] = None,
    ):
        self.fact_checker = fact_checker
        self.sensitive_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (sensitive_patterns or [r"password\s*[:=]\s*\S+", r"api[_-]?key\s*[:=]\s*\S+"])
        ]

    def check_consistency(self, answer: str, context: str) -> Tuple[bool, str]:
        """Simple heuristic: flag answers that contradict explicitly stated facts."""
        # Very lightweight contradiction check: if context says X is "normal" but answer says "high"
        contradiction_markers = [
            (r"\b(CPU|cpu)\b.*?\bnormal\b", r"\b(CPU|cpu)\b.*?\bhigh\b"),
            (r"\b(memory|mem)\b.*?\bnormal\b", r"\b(memory|mem)\b.*?\bhigh\b"),
            (r"\bresponse\b.*?\bnormal\b", r"\bresponse\b.*?\bslow\b"),
        ]
        for positive_re, negative_re in contradiction_markers:
            if re.search(positive_re, context) and re.search(negative_re, answer):
                return False, "Possible contradiction with context."
        return True, "Consistency check passed."

    def check_sensitive_leak(self, answer: str) -> Tuple[bool, List[str]]:
        """Detect potential sensitive information leaks."""
        matches = []
        for pattern in self.sensitive_patterns:
            if pattern.search(answer):
                matches.append(pattern.pattern)
        return len(matches) == 0, matches

    def check_facts(self, answer: str, query: str) -> Tuple[bool, str]:
        """Call external fact checker if available."""
        if self.fact_checker is None:
            return True, "No fact checker configured."
        return self.fact_checker(answer, query)

    def check(self, answer: str, query: str, context: str = "") -> Dict[str, Any]:
        """Run all output checks."""
        consistent, consistency_reason = self.check_consistency(answer, context)
        safe, sensitive_matches = self.check_sensitive_leak(answer)
        factual, fact_reason = self.check_facts(answer, query)

        passed = consistent and safe and factual
        return {
            "passed": passed,
            "consistency": {"passed": consistent, "reason": consistency_reason},
            "sensitive": {"passed": safe, "matches": sensitive_matches},
            "factuality": {"passed": factual, "reason": fact_reason},
        }
