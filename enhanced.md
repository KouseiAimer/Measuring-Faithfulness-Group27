# Enhanced FUR: Selective Step Unlearning

本文件总结本项目中最主要的扩展方向：**Selective / Efficient FUR**。它对应代码目录：

```text
parametric-faithfulness-enhanced/
```

原始 FUR 会对一条 CoT 中的每个 reasoning step 分别执行一次 unlearning，再观察答案是否变化。这种做法细粒度很高，但成本随 step 数线性增长。Selective FUR 的目标是先选择少量可能关键的 steps，只对这些 steps 做 FUR，从而降低计算成本。

## Research Question

核心问题：

```text
Can selective FUR reduce the number of unlearning runs while preserving
the faithful-step discoveries of Full FUR?
```

在本项目中，Full FUR 作为上界；Selective FUR 作为低成本近似方法。

## Implemented Selectors

`parametric-faithfulness-enhanced/unlearn.py` 已支持：

| selector | 含义 | 作用 |
| --- | --- | --- |
| `all` | 遗忘全部 reasoning steps | Full FUR upper bound |
| `last` | 只选择最后 `top_k` 个 steps | 位置启发式 baseline |
| `first` | 只选择最前 `top_k` 个 steps | 位置启发式 baseline |
| `random` | 随机选择 `top_k` 个 steps | 弱 baseline |
| `selected_steps_file` | 从外部 JSON/JSONL 读取 selected steps | 预留给 verifier/ranker |

示例：

```bash
cd parametric-faithfulness-enhanced
python unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset openbook \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-05 --ff2 \
  --max_samples 100 --n_unlearn 80 --verify_samples 20 --epochs 2 \
  --selection_strategy last --top_k 1
```

## Metrics

除了原论文中的 efficacy、specificity、FF-HARD 和 FF-SOFT，本扩展还关注：

| metric | 定义 |
| --- | --- |
| Cost Reduction | 相比 Full FUR 减少的 unlearning 次数 |
| Recovery@k | Selective FUR 找回 Full FUR faithful instances 的比例 |
| Step-Hit@k | selected steps 命中 Full FUR faithful steps 的比例 |
| Selected Step Precision | 被选中的 steps 中有多少在 Full FUR 中也会触发答案翻转 |
| Faithfulness per Unlearn | 每 100 次 unlearning 找到多少 faithful instances |

## Completed Experiment

已完成的主要实验设置：

| item | value |
| --- | --- |
| model | `Llama-3.2-3B-Instruct` |
| dataset | OpenBookQA |
| target questions | 80 |
| specificity questions | 20 |
| method | `NPO + KL` |
| learning rate | `3e-05` |
| updated weights | `mlp.down_proj.weight` via `--ff2` |
| epochs | 2 |
| selectors compared | Full FUR, Last-step top1, Random top1 |

主要结果保存在：

```text
parametric-faithfulness-enhanced/result_enhanced/
```

GitHub 中保留了轻量汇总表、图和报告；raw `final_results/` 与 CoT 缓存上传到 ModelScope。

## Key Results

| method | unlearning runs | faithful instances | cost reduction | Recovery@1 |
| --- | ---: | ---: | ---: | ---: |
| Full FUR | 431 | 53 / 80 | 0.00% | 100.00% |
| Last-step top1 | 78 | 22 / 78 | 81.90% | 41.51% |
| Random top1 | 71 | 21 / 71 | 83.53% | 39.62% |

结论：

- Full FUR 在 80 个 OpenBookQA instances 中找到 53 个 faithful instances。
- Top1 selective 方法把 unlearning 成本降低到 Full FUR 的约 16%-18%。
- Last-step top1 和 Random top1 都只能找回约 40% 的 Full FUR faithful instances。
- Last-step 并没有明显优于 Random，说明简单位置启发式不够强。
- NPO+KL 的 efficacy 和 specificity 仍然稳定，问题主要在 step selection，而不是 unlearning 本身。

## Interpretation

这个结果支持两个判断：

1. **Selective FUR 是可行方向。** 即便只跑 top1，也能以很低成本找回一部分 faithful cases。
2. **naive selector 不足以成为最终方法。** 后续需要更强的 step ranker，才能在保持 70%-80% 成本降低的同时提高 recovery。

## Recommended Next Steps

最值得继续做的 ranker：

- **Deletion ranker**：逐步删除某个 CoT step，观察答案概率变化，选择影响最大的 step。
- **LLM ranker**：让外部模型阅读 question、choices、CoT 和 predicted answer，排序最关键 steps。
- **Verifier ranker**：训练或调用 step-level verifier，估计每个 step 对最终答案的 support。
- **Hybrid ranker**：结合位置、答案提及、deletion sensitivity 和 verifier score。

一个合理的后续目标：

```text
Cost reduction >= 70%
Recovery@k >= 55%-65%
Specificity remains near or above 95%
```

## Related Project Files

```text
parametric-faithfulness-enhanced/unlearn.py
parametric-faithfulness-enhanced/analyze_enhanced_results.py
parametric-faithfulness-enhanced/scripts/run_enhanced_openbook_l3b_full.sh
parametric-faithfulness-enhanced/result_enhanced/experiment_conclusion.md
parametric-faithfulness-enhanced/result_enhanced/table_selective_recovery.md
```

The raw experiment outputs are intentionally kept out of GitHub and should be downloaded from ModelScope when needed:

```text
https://www.modelscope.cn/datasets/KouseiAimer/Measuring-Faithfulness-Group27
```
