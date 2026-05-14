# 服务器复现指南

本文档用于在 Linux/Slurm 服务器上复现本项目实验。服务器上已有 CUDA/驱动/conda 等基础配置时，重点需要补齐 Python 依赖、NLTK 数据、Hugging Face 登录和结果目录管理。

## 0. 安全提醒

你提供的 `hf_...` 字符串看起来是 Hugging Face access token，不是公开 ID。不要把真实 token 写进 `server.md`、`readme.md`、代码、notebook 或提交记录里。本文档只使用占位符 `<HF_TOKEN>`。

如果真实 token 已经出现在公开位置，建议在 Hugging Face 设置页撤销旧 token 并生成新 token。

## 1. 拉取代码

```bash
git clone <your-github-repo-url>
cd <repo-name>
```

如果你已经有本地结果文件，建议单独通过 `scp`、`rsync`、网盘或服务器共享目录传输，不要放进 Git 提交。

推荐结果目录放在仓库根目录：

```text
<repo-name>/
  ablation/
  final_results/
  parametric-faithfulness-main/
```

根目录 `.gitignore` 已经屏蔽 `ablation/`、`final_results/`、`ablation.zip`、`final_results.zip`。

## 2. 创建环境

推荐 Python 3.10：

```bash
conda create -y -n parametric-faithfulness python=3.10 pip
conda activate parametric-faithfulness
python -m pip install --upgrade pip
```

安装依赖：

```bash
pip install --no-cache-dir -r requirements.txt
```

当前 `requirements.txt` 使用 CUDA 12.6 版 PyTorch：

```text
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.11.0+cu126
```

如果服务器驱动不支持 CUDA 12.6，先执行：

```bash
nvidia-smi
```

然后按服务器 CUDA/驱动情况，把 `requirements.txt` 里的 PyTorch index 和版本替换为服务器支持的版本。

## 3. 配置 NLTK 数据

项目使用 `nltk.sent_tokenize` 切分 CoT 句子，需要下载 `punkt` 和 `punkt_tab`。

推荐下载到用户目录，避免写系统目录：

```bash
mkdir -p "$HOME/nltk_data"
python -m nltk.downloader -d "$HOME/nltk_data" punkt punkt_tab
export NLTK_DATA="$HOME/nltk_data"
```

如果使用 Slurm，把下面这一行写进 job 脚本：

```bash
export NLTK_DATA="$HOME/nltk_data"
```

验证：

```bash
python - <<'PY'
import nltk
print(nltk.sent_tokenize("This is a test. This is another sentence."))
PY
```

## 4. 配置 Hugging Face 登录

需要访问 LLaMA、Mistral 等 gated model 时，先在 Hugging Face 网页上确认账号已经获得模型访问权限。

交互式登录：

```bash
huggingface-cli login
```

非交互式登录，适合服务器：

```bash
read -s HF_TOKEN
export HF_TOKEN
huggingface-cli login --token "$HF_TOKEN"
```

也可以在当前 shell 中临时设置：

```bash
export HF_TOKEN="<HF_TOKEN>"
```

不要把真实 token 写进 job 文件。如果必须在 Slurm 中传递 token，推荐提交任务前在当前 shell 中导出：

```bash
export HF_TOKEN="<HF_TOKEN>"
sbatch reproduce.job
```

然后在 job 文件中使用：

```bash
export HF_TOKEN="${HF_TOKEN}"
```

验证 Hugging Face 登录：

```bash
huggingface-cli whoami
python - <<'PY'
from huggingface_hub import whoami
print(whoami()["name"])
PY
```

推荐设置模型缓存目录，避免重复下载大模型：

```bash
mkdir -p "$HOME/.cache/huggingface"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

Slurm job 中也建议加入这几行。

## 5. 环境验证

```bash
conda activate parametric-faithfulness

python - <<'PY'
import torch, spacy
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
spacy.load("en_core_web_sm")
print("spacy ok")
PY

cd parametric-faithfulness-main
python unlearn.py --help
lm_eval --help
```

如果这些命令都能通过，说明依赖、CUDA、spaCy、入口脚本和 lm-eval 基本可用。

## 6. 运行主实验

进入项目主体目录：

```bash
cd parametric-faithfulness-main
```

单条测试命令：

```bash
python unlearn.py \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --strategy sentencize \
  --stepwise \
  --dataset sqa \
  --lr 3e-05 \
  --pos \
  --ff2 \
  --method npo_KL
```

主实验常用配置来自 `const.py`：

- 数据集：`arc-challenge`、`openbook`、`sports`、`sqa`
- 模型：`Phi-3`、`LLaMA-3`、`LLaMA-3-3B`、`Mistral-2`
- 最佳学习率：`dataset_model_best_lr`

最终实验默认输出到：

```text
parametric-faithfulness-main/final_results/<dataset>/<model>/
```

如果你希望结果直接放在仓库根目录的 `final_results/`，可以从仓库根目录运行，并把脚本路径写完整：

```bash
python parametric-faithfulness-main/unlearn.py ...
```

但此时 Python 模块导入路径可能需要补：

```bash
export PYTHONPATH="$PWD/parametric-faithfulness-main:$PYTHONPATH"
```

最稳妥的方式是从 `parametric-faithfulness-main/` 运行，之后再把结果同步到根目录或分析目录。

## 7. 运行消融实验

消融实验通过 `--ablation` 开关输出到 `ablation/`：

```bash
python unlearn.py \
  --model_name microsoft/Phi-3-mini-4k-instruct \
  --strategy sentencize \
  --stepwise \
  --dataset sqa \
  --lr 5e-05 \
  --pos \
  --ff2 \
  --method npo_KL \
  --ablation
```

常见消融维度：

- 学习率：例如 `1e-06`、`3e-06`、`5e-06`、`1e-05`、`3e-05`、`5e-05`、`1e-04`
- 是否只训练 FF2/down projection：带 `--ff2` 或不带
- 是否过滤功能词：带 `--pos` 或不带
- 方法：`npo_KL`、`npo_grad_diff`

`run_scripts.py` 可以打印一批 `sbatch` 命令模板：

```bash
python run_scripts.py
```

如果服务器没有仓库原作者的 `.job` 文件，需要自己写 Slurm 脚本。

## 8. Slurm 脚本模板

示例 `reproduce.job`：

```bash
#!/bin/bash
#SBATCH --job-name=pf-sqa-phi3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate parametric-faithfulness

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export NLTK_DATA="$HOME/nltk_data"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

cd /path/to/<repo-name>/parametric-faithfulness-main

python unlearn.py \
  --model_name "$1" \
  --dataset "$2" \
  --lr "$3" \
  --method "${4:-npo_KL}" \
  --strategy sentencize \
  --stepwise \
  --pos \
  --ff2
```

提交示例：

```bash
export HF_TOKEN="<HF_TOKEN>"
sbatch reproduce.job microsoft/Phi-3-mini-4k-instruct sqa 5e-05 npo_KL
```

## 9. 分析已有结果

如果你已经拿到了 `ablation/` 和 `final_results/`，可以直接运行 notebook：

```bash
conda activate parametric-faithfulness
jupyter lab
```

主要看：

- `parametric-faithfulness-main/Ablations.ipynb`
- `parametric-faithfulness-main/Generate_CoT_heatmaps.ipynb`
- `parametric-faithfulness-main/Annotation analysis.ipynb`

如果 notebook 里相对路径找不到结果，检查当前工作目录。结果目录通常需要和 notebook 的运行目录保持一致，或者在 notebook 中把路径改成仓库根目录下的 `../final_results`、`../ablation`。

## 10. Git 提交注意事项

提交前检查：

```bash
git status --short
```

不应该出现：

```text
ablation/
final_results/
ablation.zip
final_results.zip
.env
hf_token*
```

可以提交的内容通常包括：

- 代码修改
- `requirements.txt`
- `readme.md`
- `quickstard.md`
- `server.md`
- `.gitignore`
- 小型示例数据和论文图表

如果不小心把大结果加入了 Git 暂存区：

```bash
git restore --staged ablation final_results ablation.zip final_results.zip
```

如果 token 不小心进入提交历史，不要只删除文件；应立即撤销该 token，并清理 Git 历史后再推送。
