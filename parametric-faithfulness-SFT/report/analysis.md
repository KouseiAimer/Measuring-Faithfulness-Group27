# SFT 参数忠实性分析摘要

本文件由 `analysis_pipeline.py` 从正式实验输出自动生成；详细方法、图表和讨论见 `report.tex`/`report.pdf`。

## 关键结论

- SFT 后直接答案准确率从 70.00% 提高到 73.00%，CoT 答案准确率从 73.00% 提高到 77.00%。
- 推理被显著压缩：平均步骤从 6.37 降到 3.12，平均词数从 81.33 降到 40.49。
- FUR 控制指标更强：Efficacy 从 92.58% 升到 97.62%，Specificity 从 94.04% 升到 98.79%。
- 按各模型自己的 Direct-CoT 一致题计算，FF-HARD 从 70.73% 降到 65.88%，FF-SOFT 从 54.09% 升到 56.07%。
- 在双方都满足一致性的 75 道严格配对题上，FF-HARD 变化为 -4.00 百分点（95% bootstrap CI [-16.00, 8.00]），FF-SOFT 变化为 +1.12 百分点（95% bootstrap CI [-7.95, 9.95]）。

## Teacher 数据过滤

| 状态 | 数量 | 占比 (%) |
| --- | ---: | ---: |
| accepted | 1852 | 92.60 |
| wrong_teacher_answer | 88 | 4.40 |
| invalid_json | 35 | 1.75 |
| answer_leak | 13 | 0.65 |
| sentence_count | 10 | 0.50 |
| rationale_length | 2 | 0.10 |

## 解释边界

一次 SFT 的结果支持“蒸馏后的简洁 CoT 令模型更容易被 FUR 有效且局部地干预”，但不能仅依据本轮结果声称参数忠实性整体提升。原因是 FF-HARD 与 FF-SOFT 指向不同，且 SFT 与 FUR 都作用于 `down_proj`，需要额外控制实验区分真正的因果忠实性提升与干预可塑性提高。
