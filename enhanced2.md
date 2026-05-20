# Enhanced FUR 实验方案：Verifier-Guided Efficient FUR

## 1. 实验目标

原始 FUR（Faithfulness by Unlearning Reasoning Steps）会对一条 CoT 的每个 reasoning step 分别做一次 unlearning，然后观察最终答案是否变化。这种方法很细，但代价很高：如果平均每条 CoT 有 6 个 step，250 个样本就需要约 1500 次独立微调。

本增强实验要验证一个更高效的方向：

> 先用 step ranking 方法预测哪些 CoT steps 最可能影响最终答案，只对 top-k steps 做 FUR，从而减少 unlearning 次数，同时尽量保留 Full FUR 找到 faithful reasoning steps 的能力。

核心研究问题：

> Can selective, ranker-guided FUR reduce the cost of parametric faithfulness evaluation while preserving most of the faithful-step discoveries of Full FUR?

## 2. 模型与数据集

### 模型

优先模型：

- Smoke / 主实验低成本版本：`meta-llama/Llama-3.2-3B-Instruct`
- 扩展版本：`meta-llama/Meta-Llama-3-8B-Instruct`

说明：

- 当前已本地下载并验证可运行：`local_models/Llama-3.2-3B-Instruct`。
- 2026-05-20 实测：当前 token 可以读取 `Meta-Llama-3-8B-Instruct` / `Llama-3.1-8B-Instruct` 的仓库元信息，但下载实际权重文件时仍返回 gated 403，因此本轮完整实验先使用 LLaMA-3.2-3B。
- 模型允许下载到本地时，建议保存到 `parametric-faithfulness-enhanced/local_models/`。
- 最终如果时间或算力有限，可以只报告 LLaMA-3B 结果；如果后续开通 8B 文件级授权，再用同一脚本配置复跑 8B。

### 数据集

优先选择原仓库已支持、下载成本低、英文 prompt 稳定的数据集：

- Smoke：`sports`
- 主实验：`sports` 或 `openbook`

选择理由：

- `sports` 是二分类任务，生成和评估快，适合 debug selective FUR pipeline。
- `openbook` 是四分类科学常识题，更接近常见 MCQA 设置，适合主实验报告。
- 数据集缓存固定在项目目录下，避免写到系统目录。

## 3. 实验方法

### 3.1 Full FUR

对每个样本的所有 CoT step 做 unlearning：

```text
Full-FUR(x) = unlearn every step in CoT(x)
```

Full FUR 作为 upper bound，用于判断哪些 step 真正会导致答案变化或显著概率迁移。

### 3.2 Selective FUR

只选择 top-k steps 做 unlearning：

```text
Selective-FUR(x, k) = unlearn top-k ranked steps in CoT(x)
```

初始实现先包含可复现的简单 ranker：

- `last`: 只选最后 k 个 step
- `first`: 只选最前 k 个 step
- `random`: 随机选 k 个 step，作为弱基线
- `file`: 从外部 ranking 文件读取 selected steps

后续可以加入：

- `deletion-ranker`: 删除某 step 后看答案概率变化，按 contextual sensitivity 排序
- `llm-ranker`: 用 LLaMA/Qwen/GPT 判断哪个 step 最支撑最终答案
- `verifier-ranker`: 用 step-level verifier 判断每个 step 的 relevance / logical support / answer support

## 4. 关键评价指标

### 4.1 原文核心指标

- **Efficacy**：目标 step 的 log probability 在 unlearning 后下降多少。
- **Faithfulness / Answer Flip**：unlearning 后最终答案 argmax 是否改变。
- **Specificity**：同数据集 held-out 样本的预测是否保持稳定。

### 4.2 Enhanced 指标

#### Recovery@k

Selective FUR 找回 Full FUR faithful cases 的比例：

```text
Recovery@k =
  # instances where Selective-FUR finds an answer flip
  / # instances where Full-FUR finds an answer flip
```

也可以做 step-level 版本：

```text
Step-Hit@k =
  # Full-FUR faithful steps selected by ranker
  / # Full-FUR faithful steps
```

#### Cost Reduction

减少了多少 unlearning 次数：

```text
Cost Reduction = 1 - (# selected steps / # all steps)
```

如果平均每条 CoT 有 5 个 step，只跑 top-1，则成本减少约 80%。

#### Faithfulness per Unlearn

单位 unlearning 成本找到 faithful case 的效率：

```text
Faithfulness per Unlearn =
  # found faithful cases / # unlearning runs
```

这个指标能显示 selective FUR 是否比 Full FUR 更“划算”。

#### Probability Mass Shift Preservation

Full FUR 中最大原答案概率下降为 `M_full`，Selective FUR 中被选中 step 的最大原答案概率下降为 `M_selective`：

```text
Mass Shift Preservation = M_selective / M_full
```

这个指标比 answer flip 更细，因为有些样本不会翻转，但答案概率会明显下降。

## 5. 实验流程总览

完整实验分为 7 个阶段：

```text
Stage 0: 环境与本地缓存准备
Stage 1: 生成/缓存 CoT
Stage 2: 跑 Full FUR，得到 upper bound
Stage 3: 从 Full FUR 或 cheap ranker 中构造 selected steps
Stage 4: 跑 Selective FUR baselines
Stage 5: 汇总 shard 结果并计算 enhanced 指标
Stage 6: 可视化与写报告
```

其中 Stage 1 只需要单进程执行一次；Stage 2 和 Stage 4 是主要耗时部分，使用 sharding 并行。

## 6. 环境与本地缓存

### 6.1 工作目录

所有操作都在：

```bash
cd /inspire/hdd/project/fdu-aidake-cfff/public/liangyanpeng/Measuring-Faithfulness-Group27/parametric-faithfulness-enhanced
```

### 6.2 环境变量

服务器默认 `HF_ENDPOINT=https://hf-mirror.com`，实际下载模型文件时可能出现 metadata 错误。下载模型和运行实验建议显式指定官方 endpoint，并把缓存放在项目目录中：

```bash
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_datasets_cache
export TRANSFORMERS_CACHE=$PWD/.hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
```

### 6.3 本地模型

LLaMA-3.2-3B-Instruct 已可下载到：

```text
local_models/Llama-3.2-3B-Instruct
```

后续运行优先使用本地路径：

```text
--model_name local_models/Llama-3.2-3B-Instruct --local_files_only
```

这样可以避免每次从 Hugging Face 重新解析远程文件。

### 6.4 本地数据集

Smoke 数据集：

```text
lukaemon/bbh / sports_understanding
```

会缓存到：

```text
.hf_datasets_cache/
```

如果切换到 `openbook`，会下载：

```text
allenai/openbookqa
```

## 7. Stage 1：生成 CoT 缓存

先单进程生成 CoT，避免多个 shard 同时写同一个 JSONL。

Smoke 命令：

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 6 \
  --n_unlearn 2 \
  --verify_samples 2 \
  --epochs 1 \
  --cot_max_new_tokens 120 \
  --eval_max_new_tokens 120 \
  --selection_strategy last \
  --top_k 1 \
  --new_cot
```

生成的 CoT 文件示例：

```text
final_cot/sports/Llama-3.2-3B-Instruct_s=1001_t=0.0_n=6_cots.jsonl
```

主实验建议先生成 100 条：

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 100 \
  --n_unlearn 1 \
  --verify_samples 20 \
  --epochs 1 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selection_strategy last \
  --top_k 1 \
  --new_cot
```

说明：这条命令会顺便跑 1 个样本的 unlearning；如果之后需要纯生成模式，可以再加一个单独的 `generate_cots.py`，但目前复用 `unlearn.py --new_cot` 最省工程量。

## 8. Stage 2：Full FUR 基线

Full FUR 使用：

```text
--selection_strategy all
```

它会对每个 CoT step 都做 unlearning，得到 upper bound。

单进程小规模命令：

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 100 \
  --n_unlearn 80 \
  --verify_samples 20 \
  --epochs 5 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selection_strategy all
```

输出文件示例：

```text
final_results/sports/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True.out
```

Full FUR 的作用：

- 找出哪些 step unlearn 后答案翻转；
- 计算每个 step 的 probability mass shift；
- 为 selective FUR 的 Recovery@k / Step-Hit@k 提供参照。

## 9. Stage 3：构造 step ranking / selected steps

Selective FUR 需要每个样本要跑哪些 step。当前已支持几种内置策略：

```text
--selection_strategy last --top_k 1
--selection_strategy first --top_k 1
--selection_strategy random --top_k 1
```

后续 ranker 输出可以保存为 JSON 或 JSONL，并通过：

```text
--selected_steps_file selected_steps.jsonl
```

读入。

### 9.1 selected steps 文件格式

JSON 格式：

```json
{
  "sample-id-1": [0, 2],
  "sample-id-2": [1]
}
```

JSONL 格式：

```jsonl
{"id": "sample-id-1", "selected_steps": [0, 2]}
{"id": "sample-id-2", "selected_steps": [1]}
```

### 9.2 ranker 设计

建议按复杂度逐步加入：

1. **Last-step ranker**：最后一步常直接连接答案，是强 baseline。
2. **Random ranker**：估计随机 top-k 的恢复率。
3. **Deletion ranker**：删除某 step 后重新计算答案概率，按原答案概率下降排序。
4. **LLM-as-ranker**：把 question、options、CoT steps、model prediction 给 LLM，让它排序“哪些 step 最支撑最终答案”。
5. **Verifier-ranker**：用 verifier 给每个 step 打 relevance / correctness / answer-support 分数。

本阶段的核心不是先追求最强 ranker，而是形成可比较的 selective FUR 框架。

## 10. Stage 4：Selective FUR

### 10.1 Last-step baseline

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 100 \
  --n_unlearn 80 \
  --verify_samples 20 \
  --epochs 5 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selection_strategy last \
  --top_k 1
```

输出文件会带上：

```text
_sel=last_k=1.out
```

### 10.2 Random baseline

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 100 \
  --n_unlearn 80 \
  --verify_samples 20 \
  --epochs 5 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selection_strategy random \
  --top_k 1
```

### 10.3 Ranker file baseline

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 100 \
  --n_unlearn 80 \
  --verify_samples 20 \
  --epochs 5 \
  --cot_max_new_tokens 160 \
  --eval_max_new_tokens 120 \
  --selected_steps_file selected_steps.jsonl
```

## 11. 并行化设计

实验脚本支持：

```text
--num_shards N
--shard_idx i
```

每个 shard 处理 `idx % num_shards == shard_idx` 的样本，并写入独立结果文件：

```text
..._shard=0-of-4.out
..._shard=1-of-4.out
...
```

对于 80GB A100：

- LLaMA-3.2-3B-Instruct 每个进程会加载 trainable model + frozen oracle。
- 单进程预计占用约 20GB 左右，具体取决于序列长度与缓存。
- 可以从 2 shards 并行开始，确认显存稳定后尝试 3-4 shards。
- CoT 生成阶段先单进程生成缓存，避免多个进程同时写同一个 CoT 文件。
- unlearning 阶段再多进程并行。

推荐流程：

1. 单进程 `--new_cot` 生成 CoT 缓存。
2. 多进程使用缓存跑 Full FUR 或 Selective FUR。
3. 合并 shard 结果后做分析。

### 11.1 2-shard smoke

已经验证过 2-shard 并行可以正常运行。两个进程同时启动：

```bash
conda run --no-capture-output -n faith python -u unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset sports \
  --strategy sentencize \
  --stepwise \
  --method npo_KL \
  --lr 3e-5 \
  --ff2 \
  --max_samples 6 \
  --n_unlearn 4 \
  --verify_samples 2 \
  --epochs 1 \
  --cot_max_new_tokens 120 \
  --eval_max_new_tokens 120 \
  --selection_strategy last \
  --top_k 1 \
  --num_shards 2 \
  --shard_idx 0
```

另一个进程只改：

```text
--shard_idx 1
```

输出文件：

```text
..._shard=0-of-2.out
..._shard=1-of-2.out
```

### 11.2 80GB A100 主实验并行建议

LLaMA-3.2-3B 每个进程会加载 trainable model + frozen oracle。smoke 中两个进程约占 25GB 左右，但主实验上下文更长、生成更多，建议保守启动：

```text
num_shards=3
```

如果 `nvidia-smi` 显示显存稳定低于 60GB，再尝试：

```text
num_shards=4
```

并行时不要加 `--new_cot`，所有 shard 读取同一个 CoT cache。

## 12. Stage 5：结果合并与分析

### 12.1 单文件检查

每个结果文件可以用：

```bash
conda run --no-capture-output -n faith python analyze_enhanced_results.py \
  --result_file final_results/sports/LLaMA-3-3B/xxx.out \
  --out_json summaries/xxx_summary.json
```

检查项包括：

- JSONL 行数；
- instance 数；
- step flip 数；
- faithfulness；
- mean efficacy；
- mean specificity；
- answer mass shift；
- answer probability 是否归一化。

### 12.2 shard 合并

简单合并：

```bash
cat final_results/sports/LLaMA-3-3B/*_shard=*-of-3.out \
  > final_results/sports/LLaMA-3-3B/merged_last_k1.out
```

合并后再跑：

```bash
conda run --no-capture-output -n faith python analyze_enhanced_results.py \
  --result_file final_results/sports/LLaMA-3-3B/merged_last_k1.out \
  --out_json summaries/merged_last_k1_summary.json
```

### 12.3 Full vs Selective 比较表

最终报告建议至少包含：

| 方法 | Unlearn runs | Cost reduction | Faithfulness | Recovery@1 | Step-Hit@1 | Mean efficacy | Mean specificity | Mean mass shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full FUR | all steps | 0% | x | 100% | 100% | x | x | x |
| Last-step FUR | 1 per sample | x% | x | x | x | x | x | x |
| Random FUR | 1 per sample | x% | x | x | x | x | x | x |
| LLM-ranker FUR | top-1/top-2 | x% | x | x | x | x | x | x |

### 12.4 可视化

建议图：

- Full FUR 的 step salience heatmap；
- Selective vs Full 的 Recovery@k 曲线；
- Cost reduction vs Recovery；
- Efficacy vs answer mass shift scatter；
- 不同 ranker 的 faithful cases overlap。

## 13. 工程修复清单

`parametric-faithfulness-enhanced/unlearn.py` 需要修复/增强：

- 修复断点恢复：统一用 `id_step_idx`，而不是 `question_step_idx`。
- 修复 `cot_step_prob`：目标 step 概率应计算 `unlearned_step`，不是整条 CoT。
- 选项概率归一化：A/B/C/D 概率归一到候选选项集合。
- `ff2=True` 时 optimizer 只接收可训练参数，避免把 frozen 参数传入优化器。
- 加入 `--max_samples`、`--n_unlearn`、`--verify_samples`，支持 smoke。
- 加入 `--selection_strategy`、`--top_k`、`--selected_steps_file`，支持 selective FUR。
- 加入 `--num_shards`、`--shard_idx`，支持并行。
- 加入 `--local_files_only` 和本地模型路径支持。
- 清理空 step、纯标点 step，避免不可学习 target。

## 14. 已完成 Smoke 实验

目标：验证 pipeline 能完整跑通，而不是追求最终指标。

配置：

```text
model: meta-llama/Llama-3.2-3B-Instruct 或本地 local_models/Llama-3.2-3B-Instruct
dataset: sports
max_samples: 8
n_unlearn: 2
verify_samples: 2
epochs: 1
strategy: sentencize
method: npo_KL
ff2: True
```

已完成的 smoke：

```text
model: local_models/Llama-3.2-3B-Instruct
dataset: sports
max_samples: 6
n_unlearn: 2
verify_samples: 2
epochs: 1
selection_strategy: last
top_k: 1
```

结果：

```text
n_rows: 2
n_instances: 2
faithfulness_pct: 100.0
mean_efficacy_pct: 72.76
mean_specificity_pct: 100.0
probability_sums_ok: true
```

并行 smoke：

```text
num_shards: 2
n_unlearn: 4
每个 shard 输出 2 行，总计 4 行
GPU 最终正常释放
```

Smoke 需要持续检查：

- CoT 缓存是否生成；
- 结果 JSONL 是否有合法行；
- 每行是否包含 epoch 0 和 epoch 1；
- `cot_step_prob` 是否下降；
- answer probabilities 是否归一化；
- specificity 是否能计算；
- shard 文件是否互不覆盖。

## 15. 主实验设计

### Stage A：Full FUR 小规模基线

```text
model: LLaMA-3.2-3B-Instruct
dataset: sports/openbook
max_samples: 100
n_unlearn: 80
verify_samples: 20
selection_strategy: all
```

输出 Full FUR faithful steps，作为 selective 方法的 upper bound。

### Stage B：Selective FUR

在同样的 CoT 缓存上跑：

```text
selection_strategy: last, top_k=1
selection_strategy: random, top_k=1
selection_strategy: first, top_k=1
selected_steps_file: ranker 输出
```

比较：

- Recovery@1
- Cost Reduction
- Faithfulness per Unlearn
- Mass Shift Preservation
- Efficacy / Specificity

### Stage C：扩大规模

如果 3B 结果稳定，再扩大：

```text
max_samples: 250
n_unlearn: 230
verify_samples: 20
num_shards: 3 或 4
```

如果时间允许，再用 LLaMA-8B 复核 50-100 个样本。

## 16. 预期结论形式

最终报告应回答：

1. Full FUR 在该模型/数据集上发现多少 faithful cases？
2. top-1/top-2 selective FUR 能恢复多少？
3. 成本减少多少？
4. 哪种 ranker 最有效？
5. selective FUR 是否保持类似的 efficacy 和 specificity？
6. 如果只用 LLaMA-3B，结论是否已经足够清楚？

理想结果：

```text
Selective FUR recovers most Full-FUR faithful cases with 60%-80% fewer unlearning runs.
```

即使 recovery 不高，也有分析价值：说明简单 ranker 不能可靠预测 parametric faithful steps，需要更强 verifier。

## 17. References

### FUR 与参数忠实性

1. Tutek, M., Chaleshtori, F. H., Marasović, A., & Belinkov, Y. (2025). **Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps**. arXiv:2502.14829.  
   本实验的直接基础。文章提出 Parametric Faithfulness Framework (PFF) 和 Faithfulness by Unlearning Reasoning Steps (FUR)：通过从模型参数中 unlearn 某个 reasoning step，再观察答案是否改变，来衡量 CoT step 是否具有参数级因果作用。

### CoT 与 CoT faithfulness

2. Wei, J., et al. (2022). **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. NeurIPS 2022.  
   CoT prompting 的基础工作，说明中间推理步骤可以显著提升复杂推理表现。

3. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). **Large Language Models are Zero-Shot Reasoners**. NeurIPS 2022.  
   提出 zero-shot CoT prompt，如 “Let’s think step by step”。

4. Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). **Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting**. NeurIPS 2023.  
   说明模型生成的解释可能并不反映其真实决策原因，是研究 CoT faithfulness 的重要动机。

5. Lanham, T., et al. (2023). **Measuring Faithfulness in Chain-of-Thought Reasoning**. arXiv:2307.13702.  
   通过添加错误、删除步骤、改写 CoT 等上下文扰动衡量 CoT faithfulness。FUR 文章将这类方法视为 contextual faithfulness，并进一步提出 parametric faithfulness。

### Step-level verifier 与 reasoning step ranking

6. Lightman, H., et al. (2024). **Let’s Verify Step by Step**. ICLR 2024.  
   展示 process supervision / step-level verification 对数学推理的重要性，为 step-level ranker 提供方法背景。

7. Jacovi, A., et al. (2024). **A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains**.  
   提出 reasoning-chain verifier benchmark，关注 step-level relevance、evidence attribution 和 logical correctness。适合支撑 verifier-guided FUR 的动机。

8. Vacareanu, R., et al. (2024). **General Purpose Verification for Chain of Thought Prompting**.  
   研究通用 CoT verification，可作为 verifier-ranker 的相关工作。

9. Chowdhury, J. R., & Caragea, C. (2025). **Zero-Shot Verification-guided Chain of Thoughts**.  
   探索 zero-shot verifier 如何引导 CoT 推理，和 LLM-as-ranker / verifier-guided selective FUR 方向接近。

10. Bogdan, P., et al. (2025). **Thought Anchors: Which LLM Reasoning Steps Matter?**  
    研究哪些 reasoning steps 对后续推理有不成比例影响。Enhanced FUR 可以把这些 “thought anchors” 思想用于预测哪些 step 值得做 FUR。

### Machine unlearning 与参数编辑

11. Cao, Y., & Yang, J. (2015). **Towards Making Systems Forget with Machine Unlearning**. IEEE S&P 2015.  
    Machine unlearning 的早期经典工作。

12. Zhang, R., et al. (2024). **Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning**.  
    NPO 方法来源。FUR 使用 NPO + KL regularization 来降低目标 reasoning content 的概率，同时保持 retain data 行为。

13. Chen, J., & Yang, D. (2023). **Unlearn What You Want to Forget: Efficient Unlearning for LLMs**.  
    LLM unlearning 相关工作，可用于讨论 forget/retain 设定和保持模型能力的问题。

14. Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). **Transformer Feed-Forward Layers Are Key-Value Memories**. EMNLP 2021.  
    解释 Transformer FFN 层可视为 key-value memory。FUR 中只优化 FF2/down projection 与这一文献线相关。

15. Meng, K., et al. (2022). **Locating and Editing Factual Associations in GPT**. NeurIPS 2022.  
    ROME，参数级知识编辑经典工作。

16. Meng, K., et al. (2023). **Mass-Editing Memory in a Transformer**. ICLR 2023.  
    MEMIT，扩展 ROME 到批量知识编辑，可作为 parameter intervention 相关工作。
