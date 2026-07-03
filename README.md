# Agent Hallucination Guard

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade framework for suppressing AI Agent hallucinations via **four-line defense** and **structural causal model (SCM) reasoning**. Built-in benchmark: baseline logical error rate 53% → guarded 0%.

> 📖 [中文文档](README_CN.md)  
> Corresponds to the resume project: *AI Agent Hallucination Suppression and Causal-Reasoning Architecture*  
> GitHub: https://github.com/zjgpost/agent-hallucination-guard

---

## ✨ Highlights

- **Four-line defense**: input validation → reasoning monitoring → output verification → feedback loop.
- **Causal reasoning**: uses a Structural Causal Model (SCM) to constrain the Agent’s reasoning path instead of pure probability guessing.
- **Reproducible benchmark**: `benchmarks/run_hallucination_eval.py` outputs hallucination-reduction metrics on a built-in dataset.
- **Memory engineering**: short-term memory with semantic compression and token-budget trimming.
- **Input guard**: rule-based Prompt Injection detection + JSON Schema validation + risk scoring.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Line 1: Input Guard                                        │
│  - Prompt Injection detection                               │
│  - JSON Schema validation                                   │
│  - Risk score → normal / enhanced / strict                  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Line 2: Reasoning Monitor (Causal Reasoner)                │
│  - Validate each ReAct Thought against the SCM              │
│  - Trigger causal correction on invalid paths               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Line 3: Output Guard                                       │
│  - Consistency check                                        │
│  - Sensitive information filtering                          │
│  - Pluggable RAG fact-checker                               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Line 4: Feedback Loop                                      │
│  - Reflect on failures                                      │
│  - Store error patterns for long-term memory                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/zjgpost/agent-hallucination-guard.git
cd agent-hallucination-guard
pip install -r requirements.txt

# Run the demo
python examples/server_diagnosis_demo.py

# Run the benchmark
python benchmarks/run_hallucination_eval.py

# Run tests
python -m pytest tests -v
```

---

## 📊 Benchmark Results

Run on the built-in example dataset (15 questions, simple / complex / counterfactual causal chains):

| Dataset           | Baseline Error Rate | Guarded Error Rate |
|-------------------|--------------------:|-------------------:|
| Simple causal     | 0%                  | 0%                 |
| Complex causal    | 60%                 | 0%                 |
| Counterfactual    | 100%                | 0%                 |
| **Overall**       | **53%**             | **0%**             |

**Logical hallucination reduction: 100%**

> These numbers are produced by the included benchmark script on a curated demo dataset. Replace the dataset with your own production evaluation set for real-world metrics.

---

## 🧠 Causal Reasoning Example

```python
from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner

scm = from_json("configs/server_diagnosis.json")
reasoner = CausalReasoner(scm)

valid, nodes, reason = reasoner.validate_thought(
    "High Load causes high CPU and Memory, which makes ResponseTime slow."
)
print(valid, nodes, reason)
# True ['Load', 'CPU', 'Memory', 'ResponseTime'] Thought follows a fork from common cause 'Load'.
```

---

## 🛠️ Defining Your Own SCM

Create a JSON file with nodes, domains, parents, and CPTs:

```json
{
  "smoothing": 1.0,
  "nodes": [
    {
      "name": "A",
      "domain": ["high", "normal"],
      "parents": [],
      "cpt": {"": [0.5, 0.5]}
    },
    {
      "name": "B",
      "domain": ["high", "normal"],
      "parents": ["A"],
      "cpt": {
        "high": [0.9, 0.1],
        "normal": [0.1, 0.9]
      }
    }
  ]
}
```

Load and use it:

```python
from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner

scm = from_json("configs/my_domain.json")
reasoner = CausalReasoner(scm)
```

---

## 🔌 Connecting a Real LLM

`GuardedAgent` accepts any callable that takes a prompt string and returns a response string:

```python
from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner
from agent.guarded_agent import GuardedAgent

scm = from_json("configs/server_diagnosis.json")
reasoner = CausalReasoner(scm)

# Example with an OpenAI-compatible client
def llm_client(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

agent = GuardedAgent(llm_client=llm_client, causal_reasoner=reasoner)
result = agent.run("CPU 使用率 95%，内存 80%，响应时间 5s，根因是什么？")
print(result["answer"])
```

When a Thought violates the causal model, the agent appends a `[CAUSAL CORRECTION]` note to the prompt and asks the LLM to regenerate.

> Make sure to install the LLM client you use, e.g. `pip install openai`.

---

## 🧱 Extending Guardrails

- **InputGuard**: pass custom `attack_patterns` or JSON `schemas` to validate tool parameters.
- **OutputGuard**: plug in a `fact_checker` callback that queries your RAG knowledge base.
- **FeedbackLoop**: supply a `long_term_memory` object to persist error patterns across sessions.

See `src/guardrails/` for the interfaces.

---

## 🧪 Tests

```bash
python -m pytest tests -v
```

Current status: **13 tests passing**.

---

## 📝 Related Blog Posts

- [Agent 幻觉不是 LLM 的错：一个基于 SCM 因果推理的系统性抑制框架](https://juejin.cn/spost/7657956930891939867)（掘金）  
  [CSDN 镜像](https://blog.csdn.net/janguo_qql/article/details/162548915?sharetype=blogdetail&sharerId=162548915&sharerefer=PC&sharesource=janguo_qql&spm=1011.2480.3001.8118)
- (Placeholder) From Paper to Code: Engineering Structural Causal Models for LLMs
- (Placeholder) Memory Engineering in Agent Systems

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🤝 Contributing

Issues and PRs are welcome. Please keep changes focused and add tests for new guardrails or causal patterns.
