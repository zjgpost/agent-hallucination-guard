# Agent Hallucination Guard 中文文档

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个通过 **四层防线** 与 **结构因果模型（SCM）** 抑制 AI Agent 幻觉的生产级框架。内置评测：基线逻辑错误率 53% → 护栏后 0%。

> 对应简历项目：*AI Agent 幻觉抑制与因果推理架构设计*  
> GitHub：https://github.com/zjgpost/agent-hallucination-guard

---

## ✨ 核心亮点

- **四层防线**：输入验证 → 推理监控 → 输出校验 → 反馈闭环
- **因果推理**：使用结构因果模型（SCM）约束 Agent 推理路径，替代纯概率猜测
- **可复现评测**：`benchmarks/run_hallucination_eval.py` 一键运行，输出幻觉抑制指标
- **记忆工程**：短期记忆语义压缩 + Token 预算修剪
- **输入护栏**：基于规则的 Prompt Injection 检测 + JSON Schema 校验 + 风险评分

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                         用户查询                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第一层：输入护栏（Input Guard）                             │
│  - Prompt Injection 检测                                     │
│  - JSON Schema 校验                                          │
│  - 风险评分 → normal / enhanced / strict                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第二层：推理监控（Causal Reasoner）                         │
│  - 校验每个 ReAct Thought 是否符合 SCM 因果路径              │
│  - 非法路径触发因果修正（Causal Correction）                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第三层：输出护栏（Output Guard）                            │
│  - 一致性检查                                                │
│  - 敏感信息过滤                                              │
│  - 可插拔 RAG 事实核查                                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第四层：反馈闭环（Feedback Loop）                           │
│  - 失败反思                                                  │
│  - 错误模式沉淀到长期记忆                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

```bash
git clone https://github.com/zjgpost/agent-hallucination-guard.git
cd agent-hallucination-guard
pip install -r requirements.txt

# 运行示例
python examples/server_diagnosis_demo.py

# 运行评测
python benchmarks/run_hallucination_eval.py

# 运行测试
python -m pytest tests -v
```

---

## 📊 评测结果

在内置示例数据集上运行（15 题，覆盖简单因果 / 复杂因果 / 反事实因果链）：

| 数据集           | 基线错误率 | 护栏后错误率 |
|------------------|-----------:|-------------:|
| 简单因果         | 0%         | 0%           |
| 复杂因果         | 60%        | 0%           |
| 反事实           | 100%       | 0%           |
| **总体**         | **53%**    | **0%**       |

**逻辑幻觉相对降低：100%**

> 以上数据由项目内置 benchmark 脚本在精选演示数据集上产出。生产环境请替换为自有业务评测集。

---

## 🧠 因果推理示例

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

`validate_thought` 支持四类合法因果结构：
1. **链式**：`A → B → C`
2. **分叉**：`A → B, A → C`
3. **共享效应**：`A, B → C`
4. **根因推断**：基于证据计算 RootCause 后验概率

---

## 🛠️ 定义自己的 SCM

新建一个 JSON 文件，包含节点、取值域、父节点和 CPT：

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

加载并使用：

```python
from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner

scm = from_json("configs/my_domain.json")
reasoner = CausalReasoner(scm)
```

---

## 🔌 接入真实 LLM

`GuardedAgent` 接受任意“输入 prompt 字符串，返回回复字符串”的可调用对象：

```python
from causal.builder import from_json
from causal.causal_reasoner import CausalReasoner
from agent.guarded_agent import GuardedAgent

scm = from_json("configs/server_diagnosis.json")
reasoner = CausalReasoner(scm)

# 以 OpenAI 风格客户端为例
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

当 Thought 违反因果模型时，Agent 会自动在 prompt 里追加 `[CAUSAL CORRECTION]` 提示，要求 LLM 重新生成。

> 使用哪个 LLM 客户端，就要先安装对应依赖，例如 `pip install openai`。

---

## 🧱 扩展护栏

- **InputGuard**：传入自定义 `attack_patterns` 或 JSON `schemas`，校验工具参数。
- **OutputGuard**：传入 `fact_checker` 回调，对接 RAG 知识库做事实核查。
- **FeedbackLoop**：传入 `long_term_memory` 对象，把错误模式持久化到跨会话记忆。

接口定义见 `src/guardrails/`。

---

## 🧪 测试

```bash
python -m pytest tests -v
```

当前状态：**13 个测试全部通过**。

---

## 📝 相关技术博客

- 《[Agent 幻觉不是 LLM 的错：一个基于 SCM 因果推理的系统性抑制框架](https://juejin.cn/spost/7657956930891939867)》（掘金）  
  [CSDN 镜像](https://blog.csdn.net/janguo_qql/article/details/162548915?sharetype=blogdetail&sharerId=162548915&sharerefer=PC&sharesource=janguo_qql&spm=1011.2480.3001.8118)
- （待发布）《从论文到代码：如何把结构因果模型落地到 LLM 推理监控》
- （待发布）《Agent 记忆工程：短期压缩与长期沉淀的实践》

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)。

---

## 🤝 贡献

欢迎提交 Issue 和 PR。新增 guardrail 或因果模式时，请同步补充测试。
