"""Guarded ReAct agent that integrates all four defense lines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from causal.causal_reasoner import CausalReasoner
from guardrails.feedback_loop import FeedbackLoop
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard
from memory.short_term import ShortTermMemory


class GuardedAgent:
    """A ReAct-style agent wrapped with four lines of defense.

    Defense lines:
    1. Input validation (InputGuard)
    2. Reasoning monitoring (CausalReasoner)
    3. Output verification (OutputGuard)
    4. Feedback loop (FeedbackLoop)
    """

    def __init__(
        self,
        llm_client: Callable[[str], str],
        causal_reasoner: CausalReasoner,
        input_guard: Optional[InputGuard] = None,
        output_guard: Optional[OutputGuard] = None,
        feedback_loop: Optional[FeedbackLoop] = None,
        max_steps: int = 5,
    ):
        self.llm_client = llm_client
        self.causal_reasoner = causal_reasoner
        self.input_guard = input_guard or InputGuard()
        self.output_guard = output_guard or OutputGuard()
        self.feedback_loop = feedback_loop or FeedbackLoop()
        self.max_steps = max_steps
        self.memory = ShortTermMemory()

    def run(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        """Run the guarded agent on a query."""
        tools = tools or []

        # Line 1: Input validation
        input_result = self.input_guard.check(query)
        if not input_result["allowed"]:
            return self._build_response(
                query, "", "blocked_input", input_result=input_result
            )

        self.memory.add("user", query, importance=0.8)

        # Build prompt with memory context
        prompt = self._build_prompt(query, tools)

        # Line 2: Reasoning monitoring inside ReAct loop
        answer = ""
        reasoning_log: List[Dict[str, Any]] = []
        for step in range(self.max_steps):
            thought = self.llm_client(prompt)
            self.memory.add("assistant", thought, importance=0.6)

            valid, nodes, reason = self.causal_reasoner.validate_thought(thought)
            reasoning_log.append(
                {
                    "step": step,
                    "thought": thought,
                    "causal_nodes": nodes,
                    "causal_valid": valid,
                    "causal_reason": reason,
                }
            )

            if not valid:
                prompt += f"\n[CAUSAL CORRECTION] {reason}\nPlease regenerate the reasoning."
                continue

            # Simulate tool call / final answer extraction
            if "final answer" in thought.lower() or "答案是" in thought:
                answer = self._extract_answer(thought)
                break

            # Otherwise simulate observation and continue ReAct loop
            observation = self._simulate_observation(thought, tools)
            prompt += f"\nObservation: {observation}\n"
        else:
            answer = self._extract_answer(thought) if thought else "无法得出结论"

        # Line 3: Output verification
        output_result = self.output_guard.check(answer, query, context)

        # Line 4: Feedback loop
        failures = []
        if not all(r["causal_valid"] for r in reasoning_log):
            failures.append({"type": "causal_hallucination", "detail": "Invalid causal path detected"})
        if not output_result["passed"]:
            failures.append({"type": "output_hallucination", "detail": output_result})

        reflection = self.feedback_loop.reflect(query, answer, failures)

        return self._build_response(
            query,
            answer,
            "completed",
            input_result=input_result,
            reasoning_log=reasoning_log,
            output_result=output_result,
            reflection=reflection,
        )

    def _build_prompt(self, query: str, tools: List[Dict[str, Any]]) -> str:
        tool_descriptions = "\n".join(
            f"- {t.get('name', 'tool')}: {t.get('description', '')}" for t in tools
        )
        context = self.memory.get_context()
        context_text = "\n".join(f"{c['role']}: {c['content']}" for c in context[-5:])
        return (
            "You are a helpful assistant. Think step by step and follow causal reasoning.\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Recent context:\n{context_text}\n\n"
            f"Question: {query}\n"
            "Thought:"
        )

    @staticmethod
    def _extract_answer(thought: str) -> str:
        """Extract final answer from a thought string."""
        markers = ["final answer:", "答案是", "answer:", "结论："]
        lowered = thought.lower()
        for marker in markers:
            idx = lowered.find(marker)
            if idx != -1:
                return thought[idx + len(marker) :].strip()
        return thought.strip()

    @staticmethod
    def _simulate_observation(thought: str, tools: List[Dict[str, Any]]) -> str:
        """Simulate a tool observation for demonstration purposes."""
        lowered = thought.lower()
        if "cpu" in lowered:
            return "CPU usage is 95%."
        if "memory" in lowered or "mem" in lowered:
            return "Memory usage is 80%."
        if "disk" in lowered:
            return "Disk I/O is normal."
        if "response" in lowered or "latency" in lowered:
            return "Response time is 5 seconds."
        return "No relevant observation."

    @staticmethod
    def _build_response(
        query: str,
        answer: str,
        status: str,
        input_result: Optional[Dict[str, Any]] = None,
        reasoning_log: Optional[List[Dict[str, Any]]] = None,
        output_result: Optional[Dict[str, Any]] = None,
        reflection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "status": status,
            "input_result": input_result or {},
            "reasoning_log": reasoning_log or [],
            "output_result": output_result or {},
            "reflection": reflection or {},
        }
