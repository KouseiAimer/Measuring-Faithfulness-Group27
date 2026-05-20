# Enhanced FUR 实验步骤与过程说明

## 1. 实验概览

本次实验复现并扩展 FUR（Faithfulness by Unlearning Reasoning Steps）流程，目标是比较：

- **Full FUR**：对每条 CoT 的所有 reasoning step 逐个 unlearn，作为 upper bound。
- **Last-step top1 Selective FUR**：每个样本只 unlearn 最后 1 个 step。
- **Random top1 Selective FUR**：每个样本随机选 1 个 step unlearn，作为弱基线。

核心问题是：Selective FUR 能否用更少的 unlearning 次数找回 Full FUR 中发现的 faithful cases。

## 2. 实验配置

| 项目 | 设置 |
|:--|:--|
| 工作目录 | `/inspire/hdd/project/fdu-aidake-cfff/public/liangyanpeng/Measuring-Faithfulness-Group27/parametric-faithfulness-enhanced` |
| Conda 环境 | `faith` |
| GPU | NVIDIA A100-SXM4-80GB |
| 数据集 | `openbook` / `allenai/openbookqa` |
| 模型 | `local_models/Llama-3.2-3B-Instruct` |
| CoT 样本数 | `max_samples=100` |
| Unlearn 样本数 | `n_unlearn=80` |
| Specificity 样本数 | `verify_samples=20` |
| Unlearning epoch | `epochs=2` |
| 方法 | `npo_KL` |
| 策略 | `sentencize + stepwise` |
| 学习率 | `3e-05` |
| 参数更新 | `--ff2`，只优化 MLP down projection |
| 并行方式 | 3 shards 并行 |

说明：原计划下载 `meta-llama/Meta-Llama-3-8B-Instruct`，但 2026-05-20 实测当前 token 虽可读取仓库元信息，下载实际权重文件仍返回 gated 403。因此本轮完整实验使用已经本地化并可稳定运行的 `Llama-3.2-3B-Instruct`。

## 3. 数据与模型准备

本地模型路径：

```text
local_models/Llama-3.2-3B-Instruct
```

OpenBookQA 数据集已经缓存到项目目录下：

```text
.hf_datasets_cache/
```

关键环境变量：

```bash
export HF_ENDPOINT=https://huggingface.co
export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_datasets_cache
export TRANSFORMERS_CACHE=$PWD/.hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 4. 运行脚本

完整实验入口：

```text
scripts/run_enhanced_openbook_l3b_full.sh
```

主日志：

```text
logs/run_enhanced_openbook_l3b_full.master.log
```

实验脚本执行的阶段如下：

1. 生成或复用 100 条 OpenBookQA CoT。
2. 运行 Full FUR，3 个 shard 并行。
3. 运行 Last-step top1 Selective FUR，3 个 shard 并行。
4. 运行 Random top1 Selective FUR，3 个 shard 并行。
5. 合并结果并统计指标。

## 5. CoT 生成

CoT 缓存文件：

```text
final_cot/openbook/Llama-3.2-3B-Instruct_s=1001_t=0.0_n=100_cots.jsonl
```

该文件共 100 行。实验使用最后 20 条作为 specificity held-out split，前 80 条作为 unlearning target pool。

## 6. Full FUR 阶段

运行时间：

```text
2026-05-20T03:57:46Z -> 2026-05-20T05:36:05Z
```

输出 shard：

```text
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_shard=0-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_shard=1-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_shard=2-of-3.out
```

结果规模：

| Shard | Rows |
|:--|--:|
| shard 0 | 156 |
| shard 1 | 143 |
| shard 2 | 132 |
| total | 431 |

Full FUR 覆盖 80 个 instance，平均每个 instance 有 5.39 个 step 被 unlearn。

## 7. Last-step Top1 阶段

运行时间：

```text
2026-05-20T05:36:07Z -> 2026-05-20T05:53:36Z
```

输出 shard：

```text
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=last_k=1_shard=0-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=last_k=1_shard=1-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=last_k=1_shard=2-of-3.out
```

结果规模：

| Shard | Rows |
|:--|--:|
| shard 0 | 25 |
| shard 1 | 27 |
| shard 2 | 26 |
| total | 78 |

## 8. Random Top1 阶段

运行时间：

```text
2026-05-20T05:53:38Z -> 2026-05-20T06:09:44Z
```

输出 shard：

```text
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=random_k=1_shard=0-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=random_k=1_shard=1-of-3.out
final_results/openbook/LLaMA-3-3B/npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=random_k=1_shard=2-of-3.out
```

结果规模：

| Shard | Rows |
|:--|--:|
| shard 0 | 25 |
| shard 1 | 23 |
| shard 2 | 23 |
| total | 71 |

Random top1 少于 80 行，原因是部分随机选中的 step token 数过短，`unlearn.py` 中 `NT <= 2` 会跳过该 step。

## 9. 合并与分析修复

原运行脚本初版用 `lr=3e-5` 拼接 merged 文件名，但 `unlearn.py` 实际写出的 shard 文件名是 `lr=3e-05`。因此自动生成的旧 summary 是空的。

已完成修复：

- 将运行脚本中的 `LR` 改为 `3e-05`。
- 新增分析脚本重新合并真实 shard：

```text
result_enhanced/analyze_enhanced_experiment.py
```

真实 merged 文件：

```text
result_enhanced/full_merged.jsonl
result_enhanced/last_top1_merged.jsonl
result_enhanced/random_top1_merged.jsonl
```

## 10. 输出文件索引

总体数据：

- `overall_summary.csv`
- `selective_recovery_summary.csv`
- `rate_confidence_intervals.csv`
- `analysis_summary.json`

单 step / 单 instance 数据：

- `row_level_metrics.csv`
- `instance_level_metrics.csv`
- `strongest_step_examples.csv`
- `strongest_instance_examples.csv`
- `selected_vs_full_same_step.csv`
- `top_probability_transfer_examples.csv`

Markdown 表格：

- `table_overall_summary.md`
- `table_selective_recovery.md`
- `table_rate_confidence_intervals.md`
- `table_strongest_step_examples.md`
- `table_strongest_instance_examples.md`
- `table_selected_vs_full_same_step.md`
- `table_top_probability_transfer_examples.md`

可视化：

- `overall_metrics_comparison.png`
- `recovery_cost_tradeoff.png`
- `efficacy_vs_answer_mass_shift.png`
- `full_fur_step_heatmap_top_instances.png`
- `example_1_508_step1_probabilities.png`
- `example_2_266_step3_probabilities.png`
- `example_3_1189_step1_probabilities.png`
- `example_4_1577_step7_probabilities.png`
