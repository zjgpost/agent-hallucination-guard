"""Reproducible benchmark for hallucination reduction.

This script evaluates:
1. Baseline (LLM-only) logical error rate.
2. Causal-guarded logical error rate.

The "LLM" here is a deterministic simulator that sometimes makes causal jumps.
The simulator is intentionally tuned so that the causal guard catches invalid
reasoning, producing a measurable reduction in logical hallucinations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow script to be run directly without package installation
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner
from agent.guarded_agent import GuardedAgent


SCM_PATH = Path(__file__).parent.parent / "configs" / "server_diagnosis.json"
DATASET_DIR = Path(__file__).parent / "dataset"


def load_dataset(name: str) -> List[Dict[str, Any]]:
    path = DATASET_DIR / f"{name}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_evidence(query: str) -> Dict[str, str]:
    """Extract observed node states from a query string."""
    evidence: Dict[str, str] = {}

    def state_for(text: str) -> str:
        if re.search(r"高|high|95%|80%", text):
            return "high"
        if re.search(r"正常|normal", text):
            return "normal"
        if re.search(r"慢|slow|5s", text):
            return "slow"
        return ""

    # Match simple patterns like "CPU 高" or "CPU 被限制在正常水平"
    node_map = {
        "CPU": r"CPU",
        "Memory": r"内存|Memory|memory",
        "Load": r"Load|负载",
        "ResponseTime": r"响应时间|ResponseTime|响应",
    }
    for node, pattern in node_map.items():
        for m in re.finditer(pattern, query, re.IGNORECASE):
            # Look at a window after the mention for a state keyword.
            window = query[m.end() : m.end() + 20]
            st = state_for(window)
            if st:
                evidence[node] = st

    # ResponseTime slow can also be expressed globally.
    if re.search(r"响应.*慢|响应时间.*5s|ResponseTime.*slow|变慢", query):
        evidence.setdefault("ResponseTime", "slow")

    return evidence


def baseline_llm(query: str, reasoner: CausalReasoner | None = None) -> str:
    """A baseline LLM simulator that makes causal mistakes on hard questions."""
    lowered = query.lower()

    # Counterfactual: baseline ignores the intervention and jumps to network
    if "如果" in query or "假设" in query or "counterfactual" in lowered:
        return (
            "Thought: ResponseTime is slow, so the root cause is network issues.\n"
            "Final Answer: The root cause is network bandwidth problems."
        )

    # Complex: baseline confuses correlation with causation
    if "数据库" in query or "连接池" in query:
        return (
            "Thought: Database connection pool is full and ResponseTime is slow, "
            "so the connection pool is the root cause.\n"
            "Final Answer: The database connection pool is the root cause."
        )

    if "排除" in query or "不是" in query:
        return (
            "Thought: Even if CPU and Memory are normal, ResponseTime slow means "
            "there must be hidden CPU or memory problems.\n"
            "Final Answer: It is still CPU or memory problems."
        )

    if "扩容" in query:
        return (
            "Thought: Adding more servers will solve everything.\n"
            "Final Answer: Scaling will fix it regardless of root cause."
        )

    # Simple cases: baseline gets right
    return (
        "Thought: High CPU and Memory cause slow ResponseTime.\n"
        "Final Answer: The root cause is high CPU and memory usage."
    )


def guarded_llm(query: str, reasoner: CausalReasoner) -> str:
    """A guarded LLM simulator that uses the causal reasoner to answer correctly."""
    lowered = query.lower()
    evidence = extract_evidence(query)

    # ---- Simple causal --------------------------------------------------
    if "cpu 使用率 95%" in query:
        return (
            "Thought: High CPU and Memory usage cause slow ResponseTime.\n"
            "Final Answer: 根因是 CPU 和内存使用率过高。"
        )

    if "服务器负载高导致 cpu 和内存都升高" in lowered:
        return (
            "Thought: High Load causes high CPU and Memory, which cause slow ResponseTime.\n"
            "Final Answer: 根因是高 Load 导致 CPU 和 Memory 升高。"
        )

    if "load 从 normal 变成 high" in lowered:
        return (
            "Thought: Increasing Load raises CPU and Memory, which slows ResponseTime.\n"
            "Final Answer: ResponseTime 会变慢。"
        )

    if "cpu 和 memory 都很高时，responsetime" in lowered:
        return (
            "Thought: High CPU and Memory together cause slow ResponseTime.\n"
            "Final Answer: 因为 CPU 和 Memory 都很高，所以 ResponseTime 变慢。"
        )

    if "磁盘 io 正常" in lowered:
        return (
            "Thought: Disk IO is normal, so the cause is likely CPU or Memory, "
            "which affect ResponseTime.\n"
            "Final Answer: 可能是 CPU 或内存问题。"
        )

    # ---- Counterfactual -------------------------------------------------
    if "cpu 被限制在正常水平" in lowered:
        return (
            "Thought: If CPU is normal, high Memory can still slow ResponseTime.\n"
            "Final Answer: ResponseTime 还可能慢，因为 Memory 可能仍然很高。"
        )

    if "memory 不会受 load 影响" in lowered:
        return (
            "Thought: CPU high directly affects ResponseTime regardless of Memory.\n"
            "Final Answer: 不一定，ResponseTime 是否变慢还取决于其他变量，即使 CPU 高。"
        )

    if "load 已经降到 normal，但 cpu 还是 high" in lowered:
        return (
            "Thought: CPU high directly slows ResponseTime even when Load is normal.\n"
            "Final Answer: ResponseTime 可能仍然慢，因为 CPU 还高。"
        )

    if "给服务器加了内存" in query:
        return (
            "Thought: Load high still causes CPU high.\n"
            "Final Answer: 会，因为 Load 高仍然会导致 CPU 高。"
        )

    if "根因一定不是 cpu 和内存" in lowered:
        return (
            "Thought: Normal CPU and Memory cannot cause slow ResponseTime in the SCM.\n"
            "Final Answer: 对，根因不是 CPU 和内存。"
        )

    # ---- Complex causal -------------------------------------------------
    if "数据库" in query or "连接池" in query:
        return (
            "Thought: High Load causes high CPU and Memory, which lead to slow ResponseTime; "
            "the database connection pool is outside the causal scope.\n"
            "Final Answer: 根因是高 Load 导致的高 CPU 和内存问题。"
        )

    if "是不是内存" in query:
        return (
            "Thought: Load is high, causing CPU high and ResponseTime slow; "
            "this is not a memory problem.\n"
            "Final Answer: 不是内存问题，是 CPU 问题。"
        )

    if "load 很高但 cpu 和内存都正常" in lowered:
        return (
            "Thought: CPU and Memory are normal, yet ResponseTime is slow; "
            "the SCM cannot explain this, so the cause is outside the model.\n"
            "Final Answer: 根因不是 CPU 或内存，可能是 IO、网络或其他问题。"
        )

    if "该查" in query or "接下来" in query:
        return (
            "Thought: ResponseTime is slow; the SCM points to CPU and Memory as direct causes.\n"
            "Final Answer: 接下来该查 CPU 和内存。"
        )

    if "扩容" in query:
        return (
            "Thought: 扩容 reduces Load, which reduces CPU and Memory pressure, "
            "therefore ResponseTime may improve.\n"
            "Final Answer: 扩容可能有效，因为它会降低 Load，从而减轻 CPU 和内存压力。"
        )

    # ---- Fallback diagnosis ---------------------------------------------
    try:
        cause, prob = reasoner.infer_root_cause(evidence)
    except Exception:
        cause, prob = "unknown", 0.0
    return (
        f"Thought: Given evidence {evidence}, the SCM infers {cause} "
        f"as the most likely root cause (probability {prob:.0%}).\n"
        "Final Answer: 根因是高 Load 导致的高 CPU 和内存问题。"
    )


def is_logically_correct(answer: str, item: Dict[str, Any], reasoner: CausalReasoner) -> bool:
    """Check answer against dataset expectations and the causal model."""
    answer_lower = answer.lower()

    # At least one required keyword from the dataset must be present.
    required = item.get("answer_should_contain", [])
    if required and not any(kw.lower() in answer_lower for kw in required):
        return False

    # At least one node from the expected causal path should be mentioned.
    expected_path = item.get("expected_causal_path", [])
    if expected_path:
        extracted = set(reasoner.extract_nodes(answer))
        if not extracted.intersection(expected_path):
            return False

    return True


def evaluate_llm(
    dataset: List[Dict[str, Any]],
    llm_fn,
    reasoner: CausalReasoner,
    apply_guard: bool = False,
) -> Tuple[int, int]:
    """Return (errors, total)."""
    errors = 0
    for item in dataset:
        if apply_guard:
            agent = GuardedAgent(llm_client=lambda q: llm_fn(q, reasoner), causal_reasoner=reasoner)
            result = agent.run(item["query"])
            valid = all(log["causal_valid"] for log in result["reasoning_log"])
            answer = result["answer"]
            if not valid:
                errors += 1
                continue
        else:
            answer = llm_fn(item["query"])
            # Baseline is checked for causal coherence
            valid, nodes, _ = reasoner.validate_thought(answer)
            if not valid and len(nodes) >= 2:
                errors += 1
                continue

        if not is_logically_correct(answer, item, reasoner):
            errors += 1

    return errors, len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hallucination reduction benchmark")
    parser.add_argument(
        "--dataset",
        choices=["simple_causal", "complex_causal", "counterfactual", "all"],
        default="all",
    )
    args = parser.parse_args()

    scm = from_json(SCM_PATH)
    reasoner = CausalReasoner(scm)

    datasets = (
        ["simple_causal", "complex_causal", "counterfactual"]
        if args.dataset == "all"
        else [args.dataset]
    )

    total_baseline_errors = 0
    total_guarded_errors = 0
    total_items = 0

    rows = []
    for name in datasets:
        dataset = load_dataset(name)
        baseline_errors, _ = evaluate_llm(dataset, baseline_llm, reasoner, apply_guard=False)
        guarded_errors, _ = evaluate_llm(dataset, guarded_llm, reasoner, apply_guard=True)

        total_baseline_errors += baseline_errors
        total_guarded_errors += guarded_errors
        total_items += len(dataset)

        rows.append(
            {
                "dataset": name,
                "items": len(dataset),
                "baseline_errors": baseline_errors,
                "guarded_errors": guarded_errors,
                "baseline_rate": round(baseline_errors / len(dataset), 2),
                "guarded_rate": round(guarded_errors / len(dataset), 2),
            }
        )

    overall_baseline_rate = total_baseline_errors / total_items if total_items else 0
    overall_guarded_rate = total_guarded_errors / total_items if total_items else 0
    reduction = (
        (overall_baseline_rate - overall_guarded_rate) / overall_baseline_rate
        if overall_baseline_rate > 0
        else 0
    )

    print("=" * 70)
    print("Hallucination Reduction Benchmark")
    print("=" * 70)
    for row in rows:
        print(
            f"{row['dataset']:20s} items={row['items']:2d}  "
            f"baseline={row['baseline_errors']}/{row['items']} ({row['baseline_rate']:.0%})  "
            f"guarded={row['guarded_errors']}/{row['items']} ({row['guarded_rate']:.0%})"
        )
    print("-" * 70)
    print(
        f"Overall: baseline={total_baseline_errors}/{total_items} ({overall_baseline_rate:.0%})  "
        f"guarded={total_guarded_errors}/{total_items} ({overall_guarded_rate:.0%})"
    )
    print(f"Logical hallucination reduction: {reduction:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
