# SFT 报告复现说明

本目录将 `artifacts/` 中的正式实验结果汇总为中文技术报告，并生成与
*Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps*
（arXiv:2502.14829）
一致风格的控制图及忠实性比较图。

## 一键生成

```bash
cd parametric-faithfulness-SFT/report
bash build_report.sh
```

脚本使用 `faith` 环境的 Python 重新计算统计量，再用 `lualatex` 编译
`report.tex`。无需重新执行生成、SFT 或 FUR 实验。

## 目录产物

- `report.pdf`：中文完整报告。
- `analysis.md`：由统计脚本生成的短摘要。
- `data/`：逐题指标、逐状态指标、配对统计、轨迹与审计 CSV/JSON。
- `tables/`：可直接在 LaTeX 或 Markdown 中使用的表格。
- `figures/`：论文式矢量 PDF 图以及便于预览的 PNG 图。

## 统计口径

- 论文协议结果在各模型自身的 Direct--CoT 一致题上报告。
- 严格比较使用 Base 与 SFT 同时 Direct--CoT 一致的题目，避免分母变化造成偏差。
- 配对差值的 95% 置信区间使用固定种子 `1001` 下的 10,000 次 bootstrap。
- 二元指标额外报告 exact McNemar 检验。
