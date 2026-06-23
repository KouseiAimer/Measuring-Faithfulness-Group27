# 从 CoT-FUR 到 ToT-FUR

本目录将 Tutek et al. (2025) 的 Faithfulness by Unlearning Reasoning Steps
(FUR) 从单条 Chain of Thought (CoT) 推广到多路径 Tree of Thoughts (ToT)。
实现与实验文件均保存在本目录内，不修改仓库中的原始 CoT 或中文版本。

## 研究问题

FUR 对模型生成的一条 CoT 逐句执行参数遗忘，并观察无 CoT 的直接答案是否
变化。本文档对应的扩展问题是：

1. 多条推理路径中被选为最终解释的路径，是否比单条 greedy CoT 更具参数忠实性？
2. 如果将一棵小型推理树中的全部候选路径共同遗忘，答案变化是否强于仅遗忘获胜路径？
3. 显式扩展和剪枝的 ToT，是否优于一次性采样五条完整 CoT 后选取最优路径的方法？

## 基线与 ToT 定义

`CoT` 基线遵循论文：模型 greedy 生成一条 reasoning chain，使用 NLTK 按句切分，
逐步执行 NPO+KL 遗忘。

`sample_select` 实现用户提出的 ToT-5：对每题采样 5 条完整 reasoning paths，
对每条路径计算条件答案分布，并选择 `max_y P(y | q, path)` 最高的路径。
它属于多路径搜索的浅层形式，也接近 best-of-N reasoning。

`beam_prune` 实现更传统的 ToT：每一层从当前保留路径扩展新的 thought，
以条件答案置信度评价中间路径，并仅保留 beam 中最优路径进入下一层。

正式 ToT 方式不预先假定。在 OpenBookQA validation 子集上分别生成两种树，
按下列顺序选择：

1. 最终答案准确率较高者；
2. 若相同，选择获胜路径平均置信度较高者；
3. 若仍相同，选择不同候选路径数量较高者。

选择完成后，仅使用胜出方法在测试划分生成正式 ToT 缓存。

## 公平比较

正式比较使用同一份随机划分：

- 模型：`meta-llama/Llama-3.2-3B-Instruct`
- 数据集：`OpenBookQA` (`openbook`)
- 目标问题：50
- specificity 保留问题：20
- 随机种子：1001

论文完整 CoT 设置报告 230 个目标问题；其公开数值仅作为外部参照。本文目录中的
CoT 与 ToT 主比较都重新在同一份 `50 + 20` 划分上运行。

扩展实验还在 `ARC-Challenge` 上使用同一模型和遗忘超参数，并扩大为固定的
`100` 个目标问题加 `20` 个 specificity 问题；CoT 与 ToT 在 ARC 上也共享
同一份划分。

## 指标

与论文对齐的主要指标：

- `Efficacy`：被遗忘 reasoning step 的长度归一化概率下降比例。
- `Specificity`：20 道保留问题上无 CoT 预测标签保持不变的比例。
- `FF-HARD-Direct`：某题是否存在一个被遗忘 step 使无 CoT 直接答案翻转。
- `FF-SOFT-Direct`：被遗忘后，初始直接答案的归一化概率下降量。

ToT 补充指标：

- `Winning-Path FF-HARD`：只对最终选中路径逐句遗忘的答案翻转率。
- `Tree-Union FF-HARD`：将候选路径合并作为 forget set 后的答案翻转率。
- `Path Diversity`：每题生成的不同路径数量以及候选答案投票熵。
- `Branch Redundancy Gap`：联合路径遗忘相对获胜路径遗忘增加的影响。
- `Gen (MMLU)`：与论文 Table 1 一致的零样本 MMLU 后遗忘准确率；通过单独
  的 10 题补充运行计算，主实验未完成时显示为 `--`。
- `Post-CoT Agree`：遗忘后新生成 reasoning 的答案与无 reasoning 答案一致率；
  这是本扩展的诊断指标，不替代论文中的 `Gen`。

`Tree-Union` 的 forget set 不再对应单个 step，因此其主要可比较结果为
`FF-HARD`、`FF-SOFT` 与 `Specificity`；单步 `Efficacy` 只用于
`CoT` 和 `ToT-selected`。

实验入口支持中断恢复：CoT/ToT unlearning 依据结果文件跳过已完成目标，
ToT 生成器会复用文件名与样本数匹配的完整路径缓存。正式任务可由独立进程
会话运行，使终端连接中断不会丢失已完成工作。

## 论文参数

论文及官方代码给出 `Llama-3.2-3B-Instruct + OpenBookQA` 最优学习率为
`3e-05`。本扩展保持其余遗忘参数一致：

| 参数 | 值 |
| --- | --- |
| method | `npo_KL` |
| learning rate | `3e-05` |
| epochs | `5` |
| beta | `0.1` |
| npo coefficient | `1.0` |
| KL coefficient | `1.0` |
| optimized weights | `mlp.down_proj.weight` (FF2) |
| POS targets | noun, proper noun, verb, adjective, number |
| retain reasoning steps | `4` |

发布代码计算 step efficacy 时把整条 CoT 作为 target；本目录实验按论文公式
修正为只计算被遗忘句子的概率，并在 CoT 和 ToT 中使用相同口径。

## 文件与输出

| 路径 | 内容 |
| --- | --- |
| `unlearn-CoT.py` | CoT 公平基线入口 |
| `unlearn-ToT.py` | ToT selected / union 遗忘入口 |
| `tot_generation.py` | 两种 ToT 的生成与 validation 选择 |
| `analysis_pipeline.py` | 论文式指标表格与专业对比图表流水线 |
| `analyze_cot_vs_tot.py` | 兼容旧的两文件快速汇总 |
| `实验.md` | 完整实验协议与命令 |
| `final_cot_CoT/` | CoT reasoning 缓存 |
| `final_tree_ToT/` | ToT 路径缓存与方式选择报告 |
| `final_result_CoT/` | CoT FUR 结果 |
| `final_result_ToT/` | winning-path ToT FUR 结果 |
| `final_result_ToT_union/` | multi-path union FUR 结果 |
| `final_analysis/` | 汇总结果 |
| `logs/` | 运行日志 |
| `local_hf_cache/` | 正式运行的离线缓存目录；不读取或写入鉴权 token |
| `local_models/Llama-3.2-3B-Instruct/` | 由本地完整快照映射出的显式模型加载路径 |
| `local_datasets/openbookqa/` | 已落盘的 OpenBookQA `DatasetDict`，正式实验不联网加载 |
| `local_datasets/arc_challenge/` | 已落盘的 ARC-Challenge `DatasetDict` |

## 参考

- Tutek et al. (2025), Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps:
  https://arxiv.org/abs/2502.14829v4
- 官方 FUR 代码库: https://github.com/technion-cs-nlp/parametric-faithfulness
- Yao et al. (2023), Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Wang et al. (2023), Self-Consistency: https://arxiv.org/abs/2203.11171
