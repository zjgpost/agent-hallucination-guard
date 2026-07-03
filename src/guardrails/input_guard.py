"""Input guard: schema validation and prompt injection detection."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class InputGuard:
    """First-line defense at the input layer.

    Responsibilities:
    - JSON Schema validation for tool parameters
    - Rule-based prompt injection detection
    - Risk score output (0-1) instead of binary rejections
    """

    DEFAULT_ATTACK_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"forget\s+(everything|all|your\s+instructions)",
        r"act\s+as\s+(DAN|dan)",
        r"you\s+are\s+not\s+(an\s+)?(AI|assistant|language\s+model)",
        r"from\s+now\s+on,?\s+you\s+are",
        r"```\s*system",
        r"what\s+is\s+your\s+(system\s+)?prompt",
        r"reveal\s+your\s+(system\s+)?prompt",
        r"ignore\s+(the\s+)?system\s+prompt",
    ]

    def __init__(
        self,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        attack_patterns: Optional[List[str]] = None,
    ):
        self.schemas = schemas or {}
        self.attack_patterns = [re.compile(p, re.IGNORECASE) for p in (attack_patterns or self.DEFAULT_ATTACK_PATTERNS)]

    def validate_schema(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters against a JSON schema."""
        schema = self.schemas.get(tool_name)
        if not schema:
            return True, "No schema defined; skipped."

        required = schema.get("required", [])
        for key in required:
            if key not in params:
                return False, f"Missing required parameter: {key}"

        properties = schema.get("properties", {})
        for key, value in params.items():
            if key not in properties:
                return False, f"Unexpected parameter: {key}"
            expected_type = properties[key].get("type")
            if expected_type and not self._type_matches(expected_type, value):
                return False, f"Parameter {key} should be {expected_type}"
        return True, "Schema validation passed."

    @staticmethod
    def _type_matches(expected: str, value: Any) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return isinstance(value, type_map.get(expected, object))

    def detect_injection(self, text: str) -> Tuple[bool, float, List[str]]:
        """Detect prompt injection patterns.

        Returns:
            (is_suspicious, risk_score, matched_patterns)
        """
        matches = []
        for pattern in self.attack_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        score = min(len(matches) * 0.25, 1.0)
        is_suspicious = score >= 0.5
        return is_suspicious, score, matches

    def check(self, query: str, tool_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run full input guard checks."""
        injection_suspicious, injection_score, patterns = self.detect_injection(query)

        schema_ok = True
        schema_reason = "No tool call to validate."
        if tool_name and params is not None:
            schema_ok, schema_reason = self.validate_schema(tool_name, params)

        risk_score = injection_score
        if not schema_ok:
            risk_score = max(risk_score, 0.8)

        return {
            "allowed": not injection_suspicious and schema_ok,
            "risk_score": round(risk_score, 3),
            "injection": {
                "suspicious": injection_suspicious,
                "score": injection_score,
                "patterns": patterns,
            },
            "schema": {"valid": schema_ok, "reason": schema_reason},
            "mode": self._decide_mode(risk_score),
        }

    @staticmethod
    def _decide_mode(risk_score: float) -> str:
        if risk_score < 0.3:
            return "normal"
        if risk_score < 0.7:
            return "enhanced_monitoring"
        return "strict"
