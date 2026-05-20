# Parametric Faithfulness 项目快速启动

项目主体在 `parametric-faithfulness-main/`，主要实验入口是 `unlearn.py`。根目录的 `requirements.txt` 已整理好项目运行、评估脚本和 notebook 分析所需的 Python 依赖。

## 1. 创建并激活 conda 环境

本机已创建环境：

```powershell
conda create -y -n parametric-faithfulness python=3.10 pip
conda activate parametric-faithfulness
```

如果环境已经存在，只需要执行：

```powershell
conda activate parametric-faithfulness
```

## 2. 安装依赖

`requirements.txt` 使用 CUDA 12.6 版 PyTorch，适配本机 NVIDIA 驱动显示的 CUDA 12.6：

```powershell
pip install --no-cache-dir -r requirements.txt
```

其中包括：

- 训练/推理：`torch==2.11.0+cu126`、`transformers`、`accelerate`、`datasets`
- 分词和文本处理：`nltk`、`spacy`、`en_core_web_sm`
- 评估：`lm-eval[hf]`
- notebook/画图：`jupyterlab`、`ipykernel`、`matplotlib`、`pandas`、`scipy`、`seaborn`
- LLM API notebook：`openai`

## 3. 下载 NLTK 数据

项目会用 `nltk.sent_tokenize` 做 CoT 句子切分，需要补齐 NLTK 数据：

```powershell
python -m nltk.downloader punkt punkt_tab
```

## 4. 配置 Hugging Face 权限

如果运行 LLaMA、Mistral 等需要授权的 Hugging Face 模型，需要先申请模型访问权限，并配置 token。任选一种方式即可：

```powershell
huggingface-cli login
```

或在当前 PowerShell 会话中设置：

```powershell
$env:HF_TOKEN="hf_xxx"
```

如果希望长期保存到系统环境变量：

```powershell
setx HF_TOKEN "hf_xxx"
```

新打开终端后再 `conda activate parametric-faithfulness`。

## 5. 验证环境

```powershell
python -c "import torch, spacy; print(torch.__version__); print(torch.cuda.is_available()); spacy.load('en_core_web_sm'); print('ok')"
cd parametric-faithfulness-main
python unlearn.py --help
lm_eval --help
```

本机已验证：

- PyTorch：`2.11.0+cu126`
- CUDA 可用：`True`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- `spacy.load('en_core_web_sm')` 可用
- `python unlearn.py --help` 可用
- `lm_eval --help` 可用

## 6. 启动实验示例

进入项目主体目录后运行：

```powershell
cd parametric-faithfulness-main
python unlearn.py --model_name meta-llama/Llama-3.2-3B-Instruct --strategy sentencize --stepwise --dataset sqa --lr 3e-05 --pos --ff2 --method npo_KL
```

首次运行会从 Hugging Face 下载模型和数据集，并在项目目录下生成缓存/结果目录，例如 `final_cot/`、`final_results/` 等。

注意：`unlearn.py` 会同时加载当前模型和 oracle model，显存需求较高。RTX 4060 Laptop 8GB 可以完成环境验证，但真实 unlearning 实验可能需要更大显存，或者改用更小的本地/远程模型。

## 7. 中文 C-Eval + Qwen3 复现实验

中文版本在 `parametric-faithfulness-main-cn/`，入口是 `unlearn.py`。该版本使用 Qwen chat template 并设置 `enable_thinking=False`，生成的是可见中文 CoT，不使用 Qwen3 的 hidden thinking 内容。

注意：我在 2026-05-19 检查 Hugging Face 时，官方 `Qwen/Qwen3-3B` 仓库不存在；如果你有本地或镜像中的 3B 权重，可以把 `--model_name` 换成本地路径。官方可用的近似规模模型用 `Qwen/Qwen3-4B`。

在单张 A100 80GB 上可以直接用默认 `--model_device auto --oracle_device auto` 做微调训练；如果要并行，中文入口已支持 `--num_shards` 和 `--shard_idx`，每个 shard 会写独立结果文件，避免并发写冲突。Hugging Face token 建议放在 `HF_TOKEN` 环境变量里，不要写进日志。

当前服务器显存被限制为约 10GB 时，建议先用 CPU oracle 做 smoke test：

```bash
conda activate faith
cd parametric-faithfulness-main-cn
python unlearn.py \
  --model_name Qwen/Qwen3-1.7B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 3e-5 --ff2 \
  --max_samples 8 --n_unlearn 1 --verify_samples 2 --epochs 1 \
  --cot_max_new_tokens 160 --eval_max_new_tokens 160 \
  --new_cot --oracle_device cpu
```

更大显存环境中跑完整 Qwen3-4B：

```bash
conda activate faith
cd parametric-faithfulness-main-cn
mkdir -p logs
nohup python -u unlearn.py \
  --model_name Qwen/Qwen3-4B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
  --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
  --new_cot \
  > logs/qwen3-4b-ceval-final.log 2>&1 &
```

完整 Qwen3-8B 只需把 `--model_name` 改成 `Qwen/Qwen3-8B`。运行完成后生成汇总和概率转移图：

```bash
python analyze_results.py \
  --result_file final_results/ceval/Qwen3-4B/npo_KL_sentencize_s=True_lr=1e-05_rs=1001_n=250_pos=False_ff2=True.out \
  --out_dir Qwen3-4B
```

本机已用缓存中的 `Qwen/Qwen3-1.7B` 跑通 `ceval-computer_network` 极小样本 smoke test，示例结果在 `parametric-faithfulness-main-cn/Qwen3-1.7B-smoke/`；也用 `Qwen/Qwen3-8B` 在 A100 默认 GPU device map 上跑通了 trainable+oracle 微调闭环，示例结果在 `parametric-faithfulness-main-cn/Qwen3-8B-smoke-a100/`。如果必须跑 3B，请提供本地模型路径或镜像仓库名；官方 HF 当前可直接替换为 `Qwen/Qwen3-4B` 或 `Qwen/Qwen3-8B`。
