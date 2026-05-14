# quickstard: 项目文件作用说明

本项目复现论文 "Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps"。核心思想是：先让模型在多个选择题数据集上生成 CoT，再对某个 CoT 步骤做 unlearning，观察答案是否变化、其他样本是否保持稳定，从而衡量 CoT 步骤的 faithfulness。

## 一、最重要的运行链路

主入口是 `parametric-faithfulness-main/unlearn.py`。一次实验的大致流程是：

1. `unlearn.py` 读取命令行参数，例如模型、数据集、学习率、unlearning 方法。
2. `dataload.py` 中的 `DATASETS` 找到对应数据集处理器，负责下载 Hugging Face 数据集并构造 prompt。
3. `data.py` 的 `load_or_generate_dataset_cots` 读取已有 CoT 缓存，或调用 `evaluate.py` 生成新的 CoT。
4. `data.py` 的 `cot_to_otfd` 把一个目标 CoT 切成待遗忘样本和保留样本。
5. `unlearn.py` 的 `unlearn_single` 加载模型和 oracle model，用 NPO 类损失训练若干 epoch。
6. `unlearn.py` 的 `evaluate` 在每轮后计算 efficacy、faithfulness、specificity 相关输出。
7. 结果以 jsonl 形式写入 `final_results/` 或 `ablation/`。

一个典型命令：

```powershell
cd parametric-faithfulness-main
python unlearn.py --model_name meta-llama/Llama-3.2-3B-Instruct --strategy sentencize --stepwise --dataset sqa --lr 3e-05 --pos --ff2 --method npo_KL
```

## 二、根目录文件

`readme.md`
环境配置和本地快速启动说明，包含 conda 环境、pip 依赖、NLTK 数据、Hugging Face token 和启动示例。

`requirements.txt`
Python 依赖清单。已经包含 GPU 版 PyTorch、Transformers、Datasets、spaCy、NLTK、lm-eval、notebook/画图依赖等。服务器复现时优先使用它安装。

`quickstard.md`
当前文件，用来解释项目结构、每个文件的功能以及不同功能如何串起来实现。

`server.md`
服务器复现实验的操作说明，重点覆盖 NLTK 数据、Hugging Face 登录、结果目录放置、Slurm/命令行运行方式。

`.gitignore`
用于防止把大结果、模型 checkpoint、缓存、token 文件提交到 GitHub。现在已经忽略 `ablation/`、`final_results/`、`ablation.zip`、`final_results.zip` 等。

`ablation/`、`final_results/`
你本地拿到的大型实验结果目录，分别对应消融实验和最终实验。它们用于分析和画图，但不应该上传 GitHub。

`ablation.zip`、`final_results.zip`
上述结果目录的压缩包，也已被 `.gitignore` 屏蔽。

## 三、核心 Python 文件

`parametric-faithfulness-main/unlearn.py`
主实验脚本。它实现 NPO / NPO+KL / NPO+grad-diff 的训练循环。关键函数包括：

- `make_parser`：定义命令行参数。
- `compute_loss`：实现不同 unlearning 损失。
- `unlearn_single`：对一个样本或一个 CoT 步骤执行 unlearning。
- `evaluate`：在 unlearning 前后生成概率、预测、specificity 和新 CoT。
- `run_lm_eval`：可选调用 `lm_eval` 做 MMLU/GSM8K 评估。

实现方式：该文件同时加载当前模型和 oracle model。当前模型被训练，oracle model 冻结，用于计算参考概率或 KL 保持项。

`parametric-faithfulness-main/dataload.py`
数据集适配层。它把不同来源的数据集统一成同一种接口。核心是 `DataHandler` 基类和多个子类：

- `SQA`：StrategyQA。
- `ARC`：ARC-Easy / ARC-Challenge。
- `OpenQA`：OpenBookQA。
- `Sports`：BBH sports_understanding。
- `MMLU`、`CQA`、`BoolQ`、`Aqua` 等：其他选择题或布尔题数据集。

实现方式：每个 handler 负责 `get_dataset_splits`、`make_cot_prompt`、`make_answer_prompt`、`correct_answer_letter`、`get_answer_choices`。这样上层实验不需要关心不同数据集字段名。

`parametric-faithfulness-main/data.py`
CoT 数据缓存和训练数据构造。主要做三件事：

- 读取或生成 CoT 缓存：`load_or_generate_dataset_cots`。
- 把 question + CoT 编码成 token 和 label：`qcot_encoder`。
- 构造 forget/retain 数据集：`SegmentOTFDataset`、`FRCollator`、`cot_to_otfd`。

实现方式：目标 CoT 被切成句子或完整文本作为 forget 样本，其他样本的 CoT 作为 retain 样本。collator 会左 padding，并把 question 部分的 label 设为 `-100`，避免训练时遗忘问题本身。

`parametric-faithfulness-main/evaluate.py`
推理和概率计算工具。主要负责：

- `complete`：让模型续写 CoT。
- `answer_probabilities`：计算答案选项首 token 的概率。
- `generate_dataset_cots`：批量生成数据集 CoT 缓存。
- `generation_fixed_cot`：固定一个 CoT 后再看答案概率。
- `completion_probabilities`：计算某段 completion 在模型下的 log probability。

实现方式：这些函数直接调用 Hugging Face `model.generate` 和 forward logits，把多选答案映射到 A/B/C/D/E 的 token 概率。

`parametric-faithfulness-main/models.py`
模型加载封装。`load_model_and_tokenizer` 使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载模型，默认 `torch.bfloat16` 和 `device_map="auto"`。

`parametric-faithfulness-main/segment.py`
CoT 分句和词性过滤工具。包含：

- `sentencize`：用 NLTK 分句。
- `pos_tag`：用 spaCy 词性标注。
- `align_cot_to_pos`：把词性结果对齐回 tokenizer 的 subword span。

实现方式：当命令行加 `--pos` 时，项目会尽量只 unlearn 内容词，例如名词、动词、形容词、数字，而过滤功能词。

`parametric-faithfulness-main/util.py`
通用工具函数。负责随机种子、jsonl 读写、结果分组、结果过滤、概率归一化、根据路径读取指定结果等。

`parametric-faithfulness-main/stats.py`
结果统计指标。主要计算：

- efficacy：目标 CoT 或目标步骤概率下降多少。
- specificity：held-out 样本预测是否保持稳定。
- faithfulness：unlearning 后原样本答案是否改变。
- mass shift：初始答案概率质量移动幅度。

`parametric-faithfulness-main/plotting.py`
论文图表绘制工具。`scatter_results` 画 efficacy/specificity/faithfulness 关系，`probs_barplot` 画答案概率柱状图。

`parametric-faithfulness-main/vis_samples.py`
单个样本可视化工具。`highlight_steps` 可以把问题、选项、CoT 步骤和 salience 分数画成高亮图，用于人工查看某个 CoT 步骤的重要性。

`parametric-faithfulness-main/mmlu.py`
本地 MMLU 评估脚本。用 `lm_eval.simple_evaluate` 对指定模型跑 MMLU。

`parametric-faithfulness-main/mistakes_repro.py`
复现“给 CoT 步骤加入错误”的实验。它读取已有 mistake 结果，把某一步 CoT 替换为错误版本，再计算答案是否翻转。

`parametric-faithfulness-main/mistakes_const.py`
存放加入错误实验的 prompt 模板，包括 few-shot mistake prompt 和 paraphrase prompt。

`parametric-faithfulness-main/run_scripts.py`
辅助生成服务器批处理命令。它按模型、数据集、学习率打印 `sbatch` 命令模板，适合在 Slurm 集群上批量提交实验。实际 `.job` 文件不在当前仓库中时，需要按服务器环境自己补。

`parametric-faithfulness-main/const.py`
全局常量。包含论文主实验使用的数据集、模型短名映射、Hugging Face 模型路径、每个数据集/模型组合的最佳学习率。

## 四、Notebook 文件

`Ablations.ipynb`
消融实验分析主 notebook。通常读取 `ablation/` 和 `final_results/`，计算统计量并生成论文中的消融图表。

`Generate_CoT_heatmaps.ipynb`
生成 CoT 步骤热力图和样例可视化，依赖 `vis_samples.py`、`stats.py`、`util.py`。

`Generate_annotation_data.ipynb`
从实验结果中抽样，生成 annotation study 需要的标注数据。

`Annotation analysis.ipynb`
分析人工标注结果，读取 `annotation_results/`。

`Adding mistakes repro.ipynb`
使用 OpenAI API 复现给 CoT 步骤加入错误的流程，对应 `mistakes_const.py` 的 prompt。

`CoT LLM as judge.ipynb`
使用 LLM 判断 unlearning 前后 CoT 是否改变了答案立场，读取/生成 `LM_judge_cot/` 下的 jsonl。

## 五、数据和结果目录

`annotation_data/`
人工标注实验的抽样 CSV，体积较小，可以作为论文复现实例保留在仓库中。

`annotation_results/`
人工标注结果 CSV。

`LM_judge_cot/`
LLM-as-judge 的判断结果 jsonl。

`minimal_mistake_results/`
加入错误实验的最小示例结果，体积较小，保留在仓库中方便演示。

`figures/`
论文图表和示意图，包含 png/pdf。

`final_results/`
最终主实验结果，体积大，已在根目录 `.gitignore` 屏蔽。

`ablation/`
消融实验结果，体积大，已在根目录 `.gitignore` 屏蔽。

`final_cot/`、`atomic_cot/`
运行 `unlearn.py` 时生成的 CoT 缓存目录。如果没有已有缓存，首次运行会生成。

`chkp/`、`gen_cap/`
可选 MMLU/GSM 评估时产生的临时 checkpoint 和评估输出。它们不适合上传 GitHub。

## 六、如何修改或扩展功能

新增数据集：
在 `dataload.py` 新建一个 `DataHandler` 子类，实现 prompt、答案选项、正确答案字段等方法，然后加入 `DATASETS` 字典。

新增模型：
在 `const.py` 增加 `model_name_dict` 和 `model_name_to_path` 映射；如果使用 `--pos`，还要确认 `segment.py` 的 `WHITESPACE_CHARS` 支持该 tokenizer。

新增 unlearning 方法：
在 `unlearn.py` 的 `compute_loss` 增加新的 `loss_type` 分支，再通过 `--method` 传入方法名。

新增统计指标：
在 `stats.py` 里写单样本统计函数，再在 `make_stats` 或 notebook 中调用。

新增图表：
优先在 `plotting.py` 写通用绘图函数，再在 notebook 中加载结果并调用。

服务器批量复现：
可以参考 `run_scripts.py` 输出的 `sbatch` 命令结构；如果服务器没有对应 `.job` 文件，需要按 `server.md` 里的 Slurm 模板补一个。
