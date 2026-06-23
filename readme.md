# Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps

本仓库是复现与扩展论文 **"Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps"** 的课程项目代码库。原论文提出 Faithfulness by Unlearning Reasoning Steps (FUR)：对模型生成的 Chain-of-Thought (CoT) 中某个 reasoning step 做参数遗忘，如果遗忘后模型的最终答案或答案概率发生显著变化，则说明该 step 与模型参数化决策之间存在更强联系。

本项目在官方代码基础上完成了四类工作：

- 复现原始 FUR 流程；
- 将 FUR 扩展到中文 C-Eval/Qwen3 设置；
- 实现 Selective / Efficient FUR，用更少的 step unlearning 近似 Full FUR；
- 探索 ToT-FUR 与 DeepSeek-revision SFT 后测两条扩展路线。

代码仓库只保存源码、脚本、文档和轻量示例文件。完整实验结果、模型权重、本地缓存、日志和大体积报告产物不进入 GitHub，统一放在 ModelScope 数据集仓库：

https://www.modelscope.cn/datasets/KouseiAimer/Measuring-Faithfulness-Group27

## Repository Layout

```text
.
├── parametric-faithfulness-main/        # 原论文英文 FUR 复现主体
├── parametric-faithfulness-main-cn/     # 中文 C-Eval + Qwen3 复现与分析
├── parametric-faithfulness-enhanced/    # Selective / Efficient FUR 扩展
├── parametric-faithfulness-ToT/         # Tree-of-Thoughts FUR 扩展
├── parametric-faithfulness-SFT/         # DeepSeek revision SFT + FUR 后测扩展
├── requirements.txt                     # Python 依赖
├── quickstart.md                        # 主要文件和运行链路说明
├── enhanced.md                          # Efficient FUR 选题与文献整理
└── enhanced2.md                         # Enhanced FUR 实验方案
```

## Original FUR Reproduction

主入口位于：

```text
parametric-faithfulness-main/unlearn.py
```

一次实验的大致流程如下：

1. `dataload.py` 构造数据集 prompt 和选项；
2. `evaluate.py` 生成 CoT 并计算答案概率；
3. `data.py` 将 CoT 切分为 forget/retain 训练样本；
4. `unlearn.py` 使用 NPO / NPO+KL 对目标 step 做 unlearning；
5. `stats.py` 和 notebook 汇总 efficacy、specificity、FF-HARD、FF-SOFT 等指标。

示例命令：

```bash
cd parametric-faithfulness-main
python unlearn.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --strategy sentencize --stepwise \
  --dataset sqa \
  --lr 3e-05 --pos --ff2 \
  --method npo_KL
```

## Extensions

### Chinese C-Eval / Qwen3

目录：

```text
parametric-faithfulness-main-cn/
```

该版本将 FUR 迁移到中文 C-Eval 选择题，并适配 Qwen3 chat template。生成可见中文 CoT 时显式关闭 hidden thinking：

```python
enable_thinking=False
```

示例：

```bash
cd parametric-faithfulness-main-cn
python unlearn.py \
  --model_name Qwen/Qwen3-4B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
  --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
  --new_cot
```

### Selective / Efficient FUR

目录：

```text
parametric-faithfulness-enhanced/
```

原始 Full FUR 会对每条 CoT 的所有 reasoning steps 分别 unlearn，代价较高。本扩展加入 step selection：

- `all`: Full FUR，上界；
- `last`: 只遗忘最后 `top_k` 个 step；
- `first`: 只遗忘最前 `top_k` 个 step；
- `random`: 随机选择 `top_k` 个 step；
- `selected_steps_file`: 从外部 ranker 文件读取 step 选择结果。

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

该扩展支持 sharding：

```bash
python unlearn.py ... --num_shards 3 --shard_idx 0
```

### Tree-of-Thoughts FUR

目录：

```text
parametric-faithfulness-ToT/
```

该扩展将单条 CoT 的 FUR 推广到多路径 reasoning：

- `sample_select`: 采样多条完整 reasoning paths 并选择置信度最高路径；
- `beam_prune`: 按层扩展 thought，并用答案置信度剪枝；
- `selected-path FUR`: 只遗忘最终获胜路径；
- `tree-union FUR`: 将候选路径合并为 forget set。

主要入口：

```text
tot_generation.py
unlearn-CoT.py
unlearn-ToT.py
analysis_pipeline.py
```

### DeepSeek-Revision SFT + FUR

目录：

```text
parametric-faithfulness-SFT/
```

该扩展研究外部 teacher 修订学生推理后，LoRA-SFT 是否改变学生模型后续 CoT 的参数忠实性。流程为：

1. LLaMA-3B 在 OpenBookQA train 上生成 draft rationale；
2. DeepSeek API 读取题目、选项和 draft，返回修订后的短 rationale；
3. 用 gold answer 离线过滤 teacher 输出；
4. 对 LLaMA-3B 做一次 `down_proj` LoRA-SFT；
5. Base 与 SFT 模型分别在 OpenBookQA test 上重新生成 CoT；
6. 分别运行独立 FUR 后测。

一键脚本：

```bash
cd parametric-faithfulness-SFT
bash scripts/run_full_experiment.sh
```

API key 不应写入代码或日志。推荐运行时使用环境变量或本地 `.env`，该文件已被 `.gitignore` 排除。

## Installation

建议使用 Python 3.10 和独立 conda 环境：

```bash
conda create -y -n faith python=3.10 pip
conda activate faith
pip install --no-cache-dir -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

如果使用 POS filtering，需要 spaCy 英文模型。`requirements.txt` 已包含 `en_core_web_sm` 的 wheel URL。

Hugging Face gated 模型需要提前申请权限并登录：

```bash
huggingface-cli login
```

或使用临时环境变量：

```bash
export HF_TOKEN="hf_xxx"
```

## Data and Results

GitHub 仓库不包含以下内容：

- 模型权重和 LoRA/merged checkpoints；
- Hugging Face / ModelScope 本地缓存；
- `final_results/`、`final_cot/`、`artifacts/` 等实验产物；
- 大体积日志、图表、PDF、report data；
- `.env`、token、API key。

这些文件会上传到 ModelScope：

```text
https://www.modelscope.cn/datasets/KouseiAimer/Measuring-Faithfulness-Group27
```

建议下载结果后放回对应子目录，例如：

```text
parametric-faithfulness-enhanced/final_results/
parametric-faithfulness-SFT/artifacts/
parametric-faithfulness-ToT/final_analysis/
parametric-faithfulness-main-cn/report/
```

具体路径以 ModelScope 数据集中的目录结构为准。

## Notes for Reproduction

- FUR 会同时加载 trainable model 和 frozen oracle model，显存需求较高。
- `--ff2` 只更新 MLP down projection，是本项目多数实验采用的设置。
- `--pos` 会过滤 function tokens，只对内容词进行 unlearning。
- 中文 Qwen3 实验不使用 hidden thinking，只测量可见中文 CoT。
- Selective FUR 的 naive selectors 主要作为 baseline；更强的 deletion/verifier ranker 是后续改进方向。

## Citation

```text
Tutek, M., Chaleshtori, F. H., Marasovic, A., & Belinkov, Y. (2025).
Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps.
arXiv:2502.14829.
```

官方代码库：

```text
https://github.com/technion-cs-nlp/parametric-faithfulness
```
