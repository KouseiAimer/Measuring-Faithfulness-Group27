# FUR 扩展实验方向与参考文献整理

## 对话 1：扩展实验方向

### 用户提问

我现在想要在这个文章的基础上有一些扩展的实验，你觉得我应该从事什么方面？

### 回答

我建议你优先做 **“Verifier-Guided Efficient FUR”** 这个方向，也就是：

> **用一个 verifier / step-ranker 先预测哪些 CoT steps 最重要，然后只对这些 top-k steps 做 FUR unlearning。**

这个方向最适合在原文基础上扩展，因为它直接解决原论文的一个明显瓶颈：FUR 需要对每个 CoT step 都单独做 unlearning，长 CoT 下计算成本很高。作者在 discussion 中也提到，长 reasoning chain 会带来大量步骤，逐步干预会非常耗时，并且他们设想可以用 verifier-like models 来检测和排序最重要的 CoT steps，再优先 erase。

---

## 最推荐方向：Verifier-Guided Efficient FUR

### 核心问题

原文 FUR 是：

> 对 CoT 中每一个 step 都做 unlearning，然后看哪个 step 会改变答案。

你的扩展可以改成：

> 先预测哪些 step 最可能影响答案，只对这些 step 做 unlearning。

这样你的研究问题就是：

> **Can we reduce the computational cost of FUR while preserving its ability to detect faithful reasoning steps?**

---

## 你可以怎么做

### Step 1：复现小规模 FUR

先选一个模型和一个数据集，例如：

- Model: LLaMA-3.2-3B-Instruct 或 Qwen2.5-3B-Instruct
- Dataset: StrategyQA 或 OpenBookQA
- Samples: 50–100 个即可

对每个问题生成 CoT，并切成 reasoning steps。

---



### Step 2：设计 step ranking 方法

给每个 CoT step 打一个重要性分数。可以用几种简单方法作为对比：

| Ranking 方法                       | 思路                                            |
| ---------------------------------- | ----------------------------------------------- |
| Random step                        | 随机选 step，作为弱 baseline                    |
| Final step only                    | 只选最后一步，因为最后一步通常直接支持答案      |
| LLM-as-a-ranker                    | 让 GPT/Qwen 判断哪个 step 最关键                |
| Verifier score                     | 用 verifier 判断 step 是否支持最终答案          |
| Leave-one-out context perturbation | 删除某 step，看答案是否变，作为 cheap heuristic |

你不需要一开始训练 verifier。可以先用 **LLM-as-a-ranker**，比如 prompt 它：

```text
Given the question, answer choices, model prediction, and CoT steps,
rank the reasoning steps by how important they are for supporting the final answer.
```

然后只取 top-1 或 top-2 step 做 FUR。

---

### Step 3：只对 top-k steps 做 unlearning

原文是 all-step FUR：

\[
\text{FUR}_{all}: \text{unlearn every CoT step}
\]

你的方法是 selective FUR：

\[
\text{FUR}_{top-k}: \text{unlearn only the top-k ranked steps}
\]

然后比较两者效果。

---

## 你的实验对比可以这样设计

| 方法                |             Unlearn steps | 目的                       |
| ------------------- | ------------------------: | -------------------------- |
| Full FUR            |                 all steps | 原文方法，作为 upper bound |
| Random-FUR          |              random top-k | 检查随机选择是否有效       |
| Last-step-FUR       |                final step | 检查最后一步是否足够       |
| LLM-ranker-FUR      |          LLM-ranked top-k | 你的主要方法               |
| Deletion-ranker-FUR | perturbation-ranked top-k | 低成本启发式方法           |

---

## 评价指标

你可以保留原文的核心指标：

### 1. FF-HARD recovery

看 selective FUR 能恢复多少 full FUR 发现的 faithful cases：

\[
\text{Recovery}
=
\frac{
\# \text{instances where selective FUR changes prediction}
}{
\# \text{instances where full FUR changes prediction}
}
\]

### 2. Cost reduction

看减少了多少 unlearning 次数：

\[
\text{Cost Reduction}
=
1 - \frac{k}{N_{\text{steps}}}
\]

比如平均每个 CoT 有 6 个 steps，你只 unlearn top-2，那么成本大约减少：

\[
1 - \frac{2}{6} = 66.7\%
\]

### 3. Efficacy / Specificity

沿用原文控制指标：

- Efficacy：目标 step 是否真的被 unlearn；
- Specificity：无关样本答案是否保持；
- General capability：如果算力允许，可以用小规模 MMLU 或其他 general benchmark。

原文也强调，FUR 依赖 machine unlearning 的有效性和精确性，因此需要控制 unlearning 是否成功、是否影响无关能力。

---

## 预期贡献

这个方向的贡献可以写成三点：

1. **Efficiency**  
   原 FUR 对每个 step 都做 unlearning，成本高；你的方法通过 step ranking 降低 unlearning 次数。

2. **Interpretability**  
   你不仅知道哪个 step 被 unlearning 后会影响答案，还能分析 verifier 认为重要的 step 和 FUR 实际发现的 faithful step 是否一致。

3. **Practicality**  
   对 long-CoT 或 reasoning model 更实用，因为它们可能生成很长的 reasoning chain，不能逐步全部 unlearn。

---

## 你的 proposal 可以这样命名

**中文题目：**

> 基于步骤重要性排序的高效 CoT 参数忠实性评估

**英文题目：**

> Verifier-Guided Efficient Parametric Faithfulness Evaluation for Chain-of-Thought Reasoning

或者更简洁：

> Efficient FUR via Step Importance Ranking

---

## 为什么我最推荐这个方向

因为它同时满足三个条件：

第一，它和原文联系非常紧。原文的 FUR、FF-HARD、FF-SOFT、NPO+KL 都可以继续使用，不需要完全重做一个新框架。

第二，它有明确创新。你不是简单换数据集，而是解决 FUR 的效率问题。

第三，它适合小规模实验。你可以只做 50–100 个样本，比较 full FUR 和 top-k FUR，就能形成一个完整实验闭环。

---

## 备选方向

如果你觉得这个方向太工程化，也可以考虑下面两个。

### 备选 1：不同 prompting 策略的 faithfulness 比较

研究问题：

> Does better-looking reasoning imply higher parametric faithfulness?

比较：

- Zero-shot CoT
- Few-shot CoT
- Self-consistency CoT
- Direct answer + post-hoc explanation

这个方向实现更简单，主要改 prompt。你可以比较不同 CoT 生成方式下的 FF-HARD 和 FF-SOFT。

---

### 备选 2：多语言 FUR，尤其是中文 CoT

研究问题：

> Is Chinese CoT parametrically faithful in the same way as English CoT?

可以设置：

- 中文问题 + 中文 CoT
- 英文问题 + 英文 CoT
- 中文问题 + 英文 CoT
- 英文问题 + 中文 CoT

这个方向适合中文报告，也有一定创新性。原文实验主要限制在英文 MCQA 任务，作者也明确说这是一个限制。

---

## 我的最终建议

你现在最适合做：

> **Verifier-Guided Efficient FUR**

一句话描述就是：

> 在原文 FUR 的基础上，加入一个 step ranking 阶段，只对最可能 faithful 的 reasoning steps 做 unlearning，从而降低计算成本，并检验 verifier 预测的重要步骤是否真的具有 parametric faithfulness。

这个方向既能体现你理解了原文，又能自然形成一个新的实验贡献。

---

## 对话 2：相关参考文献

### 用户提问

这个话题有什么参考文献吗？

### 回答

有。你这个扩展方向可以围绕 **“Verifier-Guided Efficient FUR / Step Importance Ranking for Parametric Faithfulness”** 来组织参考文献。文献可以分成 5 组：**原始 FUR、CoT faithfulness、step-level verifier、reasoning-step importance、machine unlearning/model editing**。

---

## 1. 核心基准文献：FUR / PFF 原文

这是你的直接出发点。

**Tutek et al., 2025. _Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps._ EMNLP 2025.**  
这篇提出了 **Parametric Faithfulness Framework (PFF)** 和 **Faithfulness by Unlearning Reasoning Steps (FUR)**。核心思想是从模型参数中 unlearn 某个 CoT step，然后观察模型答案是否变化；如果变化，说明该 step 具有 parametric faithfulness。

你可以在 proposal 中写：

> Our work builds on FUR, which measures parametric faithfulness by unlearning each reasoning step separately. However, FUR is computationally expensive for long reasoning chains. We aim to reduce this cost by introducing a step-ranking stage before unlearning.

---

## 2. CoT faithfulness 的经典前置工作

这些文献解释为什么需要研究 CoT 是否 faithful。

### Wei et al., 2022  

**_Chain-of-Thought Prompting Elicits Reasoning in Large Language Models._**

这是 CoT 的基础文章。它证明让模型生成中间推理步骤可以提升复杂推理任务表现。你的研究背景可以从这里开始：CoT 有用，但它是否 faithful 仍然不清楚。

### Kojima et al., 2022  

**_Large Language Models are Zero-Shot Reasoners._**

提出经典的 zero-shot CoT prompt：“Let’s think step by step.” 如果你实验中使用 zero-shot CoT，这篇需要引用。

### Turpin et al., 2023  

**_Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting._**

这是 CoT unfaithfulness 的经典文章。它指出 CoT explanation 可能系统性地误表征模型真正的预测原因，即模型写出来的推理不一定是它实际使用的推理。

### Lanham et al., 2023  

**_Measuring Faithfulness in Chain-of-Thought Reasoning._**

这是 FUR 最重要的前置工作之一。它通过对 CoT 做上下文扰动，比如添加错误、paraphrase、删除步骤等，观察模型答案是否变化。FUR 的一个关键论点就是：这类方法主要测的是 **contextual faithfulness**，而不是 **parametric faithfulness**。

---

## 3. Step-level verifier / reasoning-chain verification 文献

如果你要做 **Verifier-Guided Efficient FUR**，这组文献最重要。你的方法可以说是：用 verifier 或 step-ranker 先找重要步骤，再对 top-k steps 做 FUR。

### Jacovi et al., 2024  

**_A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains._**

这篇提出 REVEAL benchmark，用于评估 reasoning-chain verifier。它对每个 reasoning step 标注 relevance、evidence attribution 和 logical correctness。它非常适合支持你的 “step-level verification / step ranking” 动机。

你可以这样引用：

> Prior work has shown that reasoning chains can be evaluated at the step level, but automatic verifiers still struggle with logical correctness and contradiction detection.

### Vacareanu et al., 2024  

**_General Purpose Verification for Chain of Thought Prompting._**

这篇提出用 verifier 检查 reasoning step 是否满足 relevance、mathematical accuracy 和 logical consistency。它和你的思路很接近，因为你也可以用这些维度来给 CoT steps 排序。

### Chowdhury & Caragea, 2025  

**_Zero-Shot Verification-guided Chain of Thoughts._**

这篇研究 zero-shot setting 下如何用 LLM 自我验证 reasoning steps，并用 verifier scores 引导推理。它适合支撑你用 LLM-as-a-ranker 或 LLM-as-a-verifier 的实验设计。

---

## 4. Reasoning-step importance / thought anchors 文献

这组文献和你的创新点最接近：不是只判断 step 对不对，而是判断 **which reasoning steps matter**。

### Bogdan et al., 2025  

**_Thought Anchors: Which LLM Reasoning Steps Matter?_**

这篇非常适合你的方向。它从 sentence-level 分析 reasoning traces，提出 “thought anchors”，也就是对后续推理有不成比例影响的重要 reasoning steps。它使用了黑盒、白盒和因果归因方法来识别重要句子。你的方法可以与它形成对应：它识别重要 reasoning steps，而你用 FUR 验证这些 steps 是否具有 parametric faithfulness。

你可以在 proposal 中写：

> Inspired by the idea of thought anchors, we investigate whether step-importance signals can guide FUR to focus on a small subset of reasoning steps.

### Lightman et al., 2024  

**_Let’s Verify Step by Step._**

这篇是 process supervision / step-level verification 的代表工作。虽然主要面向数学推理，但它说明 step-level supervision 比只看 final answer 更能改善推理质量。你的 step-ranker 可以借鉴这种 process-level 思想。

---

## 5. Machine unlearning / parameter intervention 文献

因为 FUR 的核心是 machine unlearning，所以这组文献用于支撑你的干预方法。

### Cao & Yang, 2015  

**_Towards Making Systems Forget with Machine Unlearning._**

这是 machine unlearning 的早期经典论文，定义了让模型“忘记”特定信息的问题。

### Chen & Yang, 2023  

**_Unlearn What You Want to Forget: Efficient Unlearning for LLMs._**

这篇更接近 LLM 场景，讨论如何从语言模型中删除目标知识，同时保持其他能力。FUR 里的 forget set / retain set 思想和这类工作高度相关。

### Zhang et al., 2024  

**_Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning._**

这是 FUR 使用的关键 unlearning 方法 NPO 的来源。FUR 使用 **NPO + KL regularization**，其中 NPO 用于降低模型对目标 reasoning content 的偏好，KL 用于保持模型在 retain data 上接近原模型。

### Meng et al., 2022  

**_Locating and Editing Factual Associations in GPT._**

也就是 ROME。它是 model editing 经典论文，用来定位和编辑模型中的 factual associations。FUR 虽然没有直接采用 ROME，但它和参数级干预的思想相关。

### Meng et al., 2023  

**_Mass-Editing Memory in a Transformer._**

也就是 MEMIT。它扩展了 ROME，用于批量编辑模型知识。可以作为 parameter intervention 的相关工作。

### Geva et al., 2021  

**_Transformer Feed-Forward Layers Are Key-Value Memories._**

这篇解释了为什么 Transformer FFN 层可以被视为存储知识的 memory。FUR 选择更新 Transformer MLP 中的 FF2 matrix，也和这条文献线有关。

---

## 你这个方向最建议优先读的 8 篇

如果时间有限，我建议按这个顺序读：

1. **Tutek et al., 2025 — Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps**  
2. **Lanham et al., 2023 — Measuring Faithfulness in Chain-of-Thought Reasoning**  
3. **Turpin et al., 2023 — Language Models Don’t Always Say What They Think**  
4. **Jacovi et al., 2024 — A Chain-of-Thought Is as Strong as Its Weakest Link**  
5. **Bogdan et al., 2025 — Thought Anchors: Which LLM Reasoning Steps Matter?**  
6. **Vacareanu et al., 2024 — General Purpose Verification for Chain of Thought Prompting**  
7. **Zhang et al., 2024 — Negative Preference Optimization**  
8. **Geva et al., 2021 — Transformer Feed-Forward Layers Are Key-Value Memories**

---

## 可以放进 proposal 的 Related Work 结构

你可以这样写 related work：

```text
Related Work

(1) Chain-of-Thought Faithfulness.
Prior work has shown that CoT explanations can be unfaithful and may not reflect the actual reasons behind model predictions. Existing methods often intervene on the CoT context, such as adding mistakes or deleting reasoning steps.

(2) Parametric Faithfulness and FUR.
FUR addresses this limitation by intervening on model parameters through machine unlearning. However, it requires unlearning every reasoning step independently, which becomes expensive for long CoTs.

(3) Step-level Verification and Reasoning Importance.
Recent works study step-level verification and identify important reasoning steps in CoTs. Inspired by these works, we propose to use a verifier or step-ranker to select a small set of candidate faithful steps before applying FUR.

(4) Machine Unlearning for LLMs.
Our intervention follows the machine unlearning setting, where target information is removed while preserving general model behavior.
```

---

## 最贴合你扩展方向的一句话文献定位

你的方法可以定位为：

> We combine step-level reasoning verification with parametric faithfulness evaluation: verifiers are used to select potentially important CoT steps, and FUR is then applied only to these selected steps to reduce computational cost while preserving faithfulness detection.

这句话背后主要对应三条文献线：  
**FUR / parametric faithfulness**，**CoT faithfulness**，以及 **step-level verifier / thought anchors**。