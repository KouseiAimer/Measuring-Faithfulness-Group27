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
