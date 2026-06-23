# Quickstart

本页给出本项目从下载代码到运行实验的最短路径。更完整的项目说明见 `readme.md`；各扩展目录也有自己的 `README`、`quickstart` 或实验协议。

## 1. 环境准备

推荐使用 Python 3.10：

```bash
conda create -y -n faith python=3.10 pip
conda activate faith
pip install --no-cache-dir -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

如果运行 gated Hugging Face 模型，需要提前申请模型权限并登录：

```bash
huggingface-cli login
```

也可以在当前 shell 临时设置：

```bash
export HF_TOKEN="hf_xxx"
```

## 2. 数据与大文件

GitHub 只保存源码、脚本、文档、轻量图表和报告。模型权重、缓存、raw 结果、日志和大型 artifacts 不放入 GitHub。

这些文件上传到 ModelScope 数据集：

```text
https://www.modelscope.cn/datasets/KouseiAimer/Measuring-Faithfulness-Group27
```

下载后按 ModelScope 中的目录结构放回仓库根目录即可。常见大文件目录包括：

```text
parametric-faithfulness-enhanced/final_results/
parametric-faithfulness-enhanced/final_cot/
parametric-faithfulness-ToT/final_result_CoT/
parametric-faithfulness-ToT/final_result_ToT/
parametric-faithfulness-ToT/final_tree_ToT/
parametric-faithfulness-SFT/artifacts/
parametric-faithfulness-SFT/local_models/
parametric-faithfulness-main-cn/final_results/
```

## 3. 原论文 FUR 复现

主入口：

```bash
cd parametric-faithfulness-main
python unlearn.py --help
```

示例运行：

```bash
python unlearn.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --dataset sqa \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-05 \
  --pos --ff2
```

核心输出默认写到：

```text
final_cot/
final_results/
```

这两个目录属于实验产物，应上传到 ModelScope 而不是 GitHub。

## 4. 中文 C-Eval / Qwen3

目录：

```bash
cd parametric-faithfulness-main-cn
```

完整 C-Eval 示例：

```bash
python unlearn.py \
  --model_name Qwen/Qwen3-4B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
  --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
  --new_cot
```

小显存 smoke test 可降低样本数，并把 oracle 放到 CPU：

```bash
python unlearn.py \
  --model_name Qwen/Qwen3-1.7B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-5 --ff2 \
  --max_samples 8 --n_unlearn 1 --verify_samples 2 --epochs 1 \
  --cot_max_new_tokens 160 --eval_max_new_tokens 160 \
  --new_cot --oracle_device cpu
```

## 5. Selective / Efficient FUR

目录：

```bash
cd parametric-faithfulness-enhanced
```

Full FUR：

```bash
python unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset openbook \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-05 --ff2 \
  --max_samples 100 --n_unlearn 80 --verify_samples 20 --epochs 2 \
  --selection_strategy all
```

只遗忘最后一步：

```bash
python unlearn.py \
  --model_name local_models/Llama-3.2-3B-Instruct \
  --local_files_only \
  --dataset openbook \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-05 --ff2 \
  --max_samples 100 --n_unlearn 80 --verify_samples 20 --epochs 2 \
  --selection_strategy last --top_k 1
```

并行分片示例：

```bash
python unlearn.py ... --num_shards 3 --shard_idx 0
python unlearn.py ... --num_shards 3 --shard_idx 1
python unlearn.py ... --num_shards 3 --shard_idx 2
```

项目已提供批量脚本：

```bash
bash scripts/run_enhanced_openbook_l3b_full.sh
```

## 6. Tree-of-Thoughts FUR

目录：

```bash
cd parametric-faithfulness-ToT
```

常用脚本：

```bash
bash scripts/run_openbook_cot_full.sh
bash scripts/run_openbook_tot_select_and_full.sh
bash scripts/run_openbook_tot_union.sh sample_select
bash scripts/run_arc_cot_full.sh
bash scripts/run_arc_tot_select_and_full.sh
```

主要入口：

```text
tot_generation.py
unlearn-CoT.py
unlearn-ToT.py
analysis_pipeline.py
```

## 7. DeepSeek-Revision SFT + FUR

目录：

```bash
cd parametric-faithfulness-SFT
```

该扩展需要本地 `.env` 或环境变量中的 `DEEPSEEK_API_KEY`。不要把 key 写入 git。

完整流程：

```bash
bash scripts/run_full_experiment.sh
```

分阶段运行：

```bash
bash scripts/generate_training_data.sh drafts
bash scripts/generate_training_data.sh revisions
bash scripts/train_once.sh
bash scripts/generate_eval_cots.sh base
bash scripts/generate_eval_cots.sh sft
bash scripts/run_base_fur.sh
bash scripts/run_sft_fur.sh
bash scripts/analyze.sh
```

## 8. 提交与归档规则

提交 GitHub 前建议检查：

```bash
git status --short
git status --ignored --short
```

不应提交：

```text
.env
*.token
local_models/
local_datasets/
.hf_cache/
final_cot/
final_results/
artifacts/
logs/
*.jsonl
*.out
```

单文件小于 10MB 的图表、PDF、表格和报告源文件可以留在 GitHub，便于直接查看实验结论。
