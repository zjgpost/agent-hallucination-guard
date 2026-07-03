"""Short-term memory with semantic compression and token budget."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional


class ShortTermMemory:
    """Fixed-capacity short-term memory with compression.

    - Keeps last N interactions.
    - When exceeding capacity, oldest interactions are summarized into a single entry.
    """

    def __init__(self, capacity: int = 10, compress_batch: int = 3):
        self.capacity = capacity
        self.compress_batch = compress_batch
        self.interactions: deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.token_budget = 4000

    def add(self, role: str, content: str, importance: float = 0.5) -> None:
        self.interactions.append(
            {
                "role": role,
                "content": content,
                "importance": importance,
                "compressed": False,
            }
        )
        self._maybe_compress()

    def _maybe_compress(self) -> None:
        if len(self.interactions) < self.capacity:
            return
        # Compress oldest compress_batch entries
        to_compress = [self.interactions.popleft() for _ in range(self.compress_batch)]
        summary = self._summarize(to_compress)
        self.interactions.appendleft(
            {
                "role": "summary",
                "content": summary,
                "importance": max(i["importance"] for i in to_compress),
                "compressed": True,
            }
        )

    @staticmethod
    def _summarize(entries: List[Dict[str, Any]]) -> str:
        """Simple rule-based summarizer; in production this calls an LLM."""
        roles = set(e["role"] for e in entries)
        contents = "; ".join(e["content"] for e in entries)
        # Truncate to avoid explosion
        if len(contents) > 200:
            contents = contents[:200] + "..."
        return f"[Summary] roles={sorted(roles)}: {contents}"

    def get_context(self) -> List[Dict[str, Any]]:
        return list(self.interactions)

    def estimate_tokens(self) -> int:
        """Rough token estimate: ~0.75 tokens per Chinese/English char on average."""
        total = sum(len(i["content"]) for i in self.interactions)
        return int(total * 0.75)

    def trim_to_budget(self, budget: Optional[int] = None) -> None:
        """Drop or compress low-importance entries until under token budget."""
        budget = budget or self.token_budget
        while self.estimate_tokens() > budget and self.interactions:
            # Drop lowest importance non-summary first
            droppable = [i for i in self.interactions if i["role"] != "summary"]
            if not droppable:
                droppable = list(self.interactions)
            droppable.sort(key=lambda x: x["importance"])
            self.interactions.remove(droppable[0])
