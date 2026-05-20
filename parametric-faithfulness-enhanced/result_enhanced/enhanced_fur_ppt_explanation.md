# Enhanced FUR PPT Explanation

本文档用于解释 `enhanced_fur_results_beamer.pdf` 这份 5 页英文 PPT，方便期末报告时组织口头讲述。PPT 对应文件：

- `enhanced_fur_results_beamer.tex`
- `enhanced_fur_results_beamer.pdf`

实验主题是：在原始 Full FUR 的基础上，探索是否可以只 unlearn 少量被选中的 reasoning steps，从而降低 faithfulness evaluation 的计算成本。

## Slide 1: Enhanced FUR Experimental Setup

这一页介绍实验动机和基本设置。

左侧说明为什么要做 enhanced FUR。原始 Full FUR 会对每条 CoT 的每个 reasoning step 分别进行一次 unlearning，然后观察最终答案是否变化。这个方法很细，但成本高。如果每条 CoT 平均有 5 个 step，那么 80 个样本就要运行约 400 次 unlearning。

本实验的问题是：

> Can we unlearn only one selected step per instance and still recover many faithful cases found by Full FUR?

也就是：只做 top-1 selective unlearning，是否能保留 Full FUR 的主要发现。

右侧列出实验配置：

- 模型：`Llama-3.2-3B-Instruct`
- 数据集：OpenBookQA
- CoT 缓存：100 条
- unlearning target：80 个样本
- specificity split：20 个 held-out 样本
- 方法：NPO-KL，stepwise，sentencize
- 训练设置：2 epochs，学习率 `3e-05`
- 并行方式：3 shards on A100 80GB

报告时可以强调：原计划使用 LLaMA-8B，但权重下载时 gated access 被拒绝，因此完整实验使用本地可运行的 LLaMA-3B。这个限制不会影响 enhanced pipeline 的有效性，但最终结论应表述为 LLaMA-3B 上的结果。

## Slide 2: Overall Results

这一页对比 Full FUR、Last-step top1 和 Random top1 的总体表现。

左侧表格是关键结果：

| Method | Rows | Faithful instances | Specificity |
|---|---:|---:|---:|
| Full FUR | 431 | 53/80 = 66.25% | 94.51% |
| Last top1 | 78 | 22/78 = 28.21% | 94.74% |
| Random top1 | 71 | 21/71 = 29.58% | 93.73% |

这里的 rows 是 unlearning step-runs 的数量。Full FUR 需要 431 次 step-level unlearning，而 top1 方法只需要 78 或 71 次，成本显著降低。

Faithful instances 表示：至少有一个 step 被 unlearn 后导致答案翻转的样本数。Full FUR 的 53/80 明显高于两个 top1 baseline，这是因为 Full FUR 会尝试所有 step，而 top1 方法每个样本只尝试一个 step。

右侧图 `overall_metrics_comparison.png` 展示四个维度：

1. unlearning cost
2. faithful instance rate
3. mean original-answer probability drop
4. mean specificity

主要结论：

- Full FUR 的 faithful instance discovery 最强。
- Selective top1 的计算成本低很多。
- 三种方法 specificity 都在 94% 左右，说明 unlearning 没有明显破坏 held-out 样本预测。
- selective 方法的 mean answer mass shift 并不低，说明被选中的 step 一旦有效，也能产生较明显概率变化。

报告时可以说：Full FUR 是 upper bound，而 top1 selective 是 cost-saving baseline。

## Slide 3: Recovery-Cost Tradeoff

这一页是 enhanced 实验最重要的一页，说明 selective FUR 的“成本-召回率”关系。

表格中的关键指标：

| Metric | Last top1 | Random top1 |
|---|---:|---:|
| Cost reduction | 81.90% | 83.53% |
| Faithful cases found | 22/53 | 21/53 |
| Recovery@1 | 41.51% | 39.62% |
| Step-Hit@1 | 17.21% | 18.85% |
| Selected-step precision | 26.92% | 32.39% |

指标解释：

- **Cost reduction**：相比 Full FUR 少做了多少 unlearning。
- **Faithful cases found**：Full FUR 找到 53 个 faithful instances，selective 方法找回了其中多少。
- **Recovery@1**：selective top1 找回 Full FUR faithful instances 的比例。
- **Step-Hit@1**：selective 选中的 step 是否命中 Full FUR 中真正会导致答案翻转的 faithful steps。
- **Selected-step precision**：selective 选中的 step 里，有多少在 Full FUR 中也是 faithful step。

右侧图 `recovery_cost_tradeoff.png` 可视化了 cost reduction 和 Recovery@1。

主要结论：

- Last top1 和 Random top1 都能节省超过 80% 的 unlearning 成本。
- 但它们只能找回约 40% 的 Full FUR faithful instances。
- Last-step top1 没有明显优于 random top1。

这说明 selective FUR 这个方向是有价值的，因为成本确实大幅降低；但 naive selector 还不够强。下一步需要更合理的 step ranker，例如 deletion-based ranker 或 verifier-guided ranker。

报告时可以把这一页作为 enhanced experiment 的核心贡献与不足。

## Slide 4: Strong Positive Cases

这一页展示 Full FUR 中最显著的 probability transfer 个例。

左侧表格列出原答案概率下降最大的几个 step：

| ID | Step | Answer change | Drop | Efficacy |
|---:|---:|---|---:|---:|
| 508 | 1 | D -> A | 0.927 | 91.56% |
| 266 | 3 | D -> C | 0.926 | 96.09% |
| 1189 | 1 | B -> A | 0.925 | 92.04% |
| 1577 | 7 | B -> D | 0.900 | 95.81% |
| 1577 | 11 | B -> A | 0.894 | 97.16% |

这些样本说明：unlearning 某个 reasoning step 后，原本占主导的答案选项概率几乎消失，另一个选项成为新的 argmax。这是 FUR 中最强的 faithful reasoning evidence。

右侧两张图分别是：

- `example_1_508_step1_probabilities.png`
- `example_2_266_step3_probabilities.png`

图中蓝色表示 before unlearning，橙色表示 after unlearning。可以看到概率质量从原答案转移到另一个选项。

报告时可以强调：

- 这些不是微弱变化，而是接近完整的 probability mass transfer。
- 因此这些 CoT steps 很可能对模型最终答案有因果支撑。
- 这也说明 unlearning 方法确实能发现 faithful reasoning steps。

## Slide 5: Global Pattern and Final Takeaways

最后一页总结整体分布和最终结论。

左图 `efficacy_vs_answer_mass_shift.png` 展示每个 step 的目标 step probability reduction 与 answer probability drop 的关系。

观察：

- 很多点的 efficacy 很高，但 answer probability drop 接近 0。
- 这说明目标 step 被成功遗忘，并不必然导致最终答案改变。
- 因此只报告 efficacy 不够，必须同时报告 answer flip 和 probability mass shift。

右图 `full_fur_step_heatmap_top_instances.png` 展示 Full FUR 中 top instances 的 step-level salience。红色越深，说明 unlearning 该 step 后原答案概率下降越大。

观察：

- faithful steps 在不同 instance 中分布不均匀。
- 有些样本多个 step 都有强影响，例如 ID `508`、`1577` 等。
- 有些 step 虽然被 unlearn 成功，但对最终答案几乎没有影响。

底部 bullets 是最终结论：

1. High efficacy does not guarantee an answer flip; probability-shift metrics are necessary.
2. Full FUR is the best upper bound, but it is expensive.
3. Top-1 selective FUR saves more than 80% of the cost, but recovers only about 40% of faithful instances.
4. Next step: replace naive top-1 selection with deletion-based or verifier-guided ranking.

报告时可以这样总结：

> Enhanced FUR validates the cost-saving direction, but simple top1 heuristics are not enough. The next meaningful improvement is to build a better step selector, so that we can preserve most of Full FUR's faithful-step discoveries while avoiding exhaustive per-step unlearning.

## Recommended Oral Summary

如果时间有限，可以用下面这段作为口头总结：

> In the enhanced experiment, Full FUR found 53 faithful instances out of 80, but required 431 step-level unlearning runs. Selective top1 methods reduced the cost by more than 80%, using only 71 to 78 unlearning runs. However, they recovered only about 40% of the faithful instances found by Full FUR, and the last-step heuristic was not clearly better than random selection. This suggests that selective FUR is promising for efficiency, but the key missing component is a stronger step ranker, such as a deletion-based or verifier-guided selector.

## Files Used by the PPT

PPT source and compiled PDF:

- `enhanced_fur_results_beamer.tex`
- `enhanced_fur_results_beamer.pdf`

Figures:

- `overall_metrics_comparison.png`
- `recovery_cost_tradeoff.png`
- `example_1_508_step1_probabilities.png`
- `example_2_266_step3_probabilities.png`
- `efficacy_vs_answer_mass_shift.png`
- `full_fur_step_heatmap_top_instances.png`

Main data files:

- `analysis_summary.json`
- `overall_summary.csv`
- `selective_recovery_summary.csv`
- `row_level_metrics.csv`
- `instance_level_metrics.csv`
- `top_probability_transfer_examples.csv`
