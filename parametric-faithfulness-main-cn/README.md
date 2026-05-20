# C-Eval 中文版实验说明

本目录是在 C-Eval 中文选择题数据集上复现论文
`Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps`
的中文版本。主入口是 `unlearn.py`，它会转发到 `unlearn-cn.py`。

默认配置：

- 数据集：`ceval/ceval-exam`
- 默认任务：`ceval`，会从 C-Eval 全科验证集中按随机种子抽样
- 默认样本数：`--max_samples 250`，与原项目每个数据集最多 250 条保持一致
- 模型：`Qwen/Qwen3-8B`、`Qwen/Qwen3-4B`、`Qwen/Qwen3-1.7B`
- 中文 CoT：使用 Qwen chat template，并设置 `enable_thinking=False` 关闭 thinking；生成的是可见中文推理过程，再按中文标点做步骤切分

我在 2026-05-19 检查 Hugging Face 时，官方 `Qwen/Qwen3-3B` 仓库不存在；如果你有本地或镜像中的 3B 权重，直接把 `--model_name` 换成本地模型路径或实际仓库名即可。下面命令使用官方可用的 `Qwen/Qwen3-4B` 作为 3B 附近规模的替代。

## A100 80GB 训练命令

服务器是单张 A100 80GB 时，不需要把 oracle model 放到 CPU；默认 `--model_device auto --oracle_device auto` 会把 trainable model 和 frozen oracle 都放到 GPU/可用 device map。Hugging Face token 建议通过环境变量设置，不要写进命令日志：

```bash
export HF_TOKEN=hf_xxx
conda activate faith
cd parametric-faithfulness-main-cn
mkdir -p logs
nohup python -u unlearn.py \
  --model_name Qwen/Qwen3-8B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
  --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
  --new_cot \
  > logs/qwen3-8b-ceval-a100.log 2>&1 &
```

如果要并行，优先用分片避免多个进程写同一个结果文件。单张 A100 上，`Qwen3-8B` 建议先只跑 1 个进程；`Qwen3-1.7B` 或更小模型可尝试 2 个 shard：

```bash
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 nohup python -u unlearn.py \
    --model_name Qwen/Qwen3-1.7B \
    --dataset ceval \
    --strategy sentencize --stepwise \
    --method npo_KL --lr 3e-5 --ff2 \
    --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
    --num_shards 2 --shard_idx "$shard" \
    --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
    --new_cot \
    > "logs/qwen3-1.7b-ceval-shard-${shard}.log" 2>&1 &
done
```

## 10GB 显存 smoke test

当前项目环境是 conda 环境 `faith`。10GB 显存通常不够同时在 GPU 上加载 trainable model 和 oracle model；可以先用 CPU oracle 跑小样本，确认 CoT、unlearning 和指标文件格式正确：

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

如果要在 10GB 上尝试 Qwen3-4B，也建议先用极小样本和 CPU oracle：

```bash
python unlearn.py \
  --model_name Qwen/Qwen3-4B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 8 --n_unlearn 1 --verify_samples 2 --epochs 1 \
  --cot_max_new_tokens 160 --eval_max_new_tokens 160 \
  --new_cot --oracle_device cpu
```

## 完整结果命令

在更大显存环境里，推荐让 trainable model 和 oracle model 都使用 `auto` device map。完整 C-Eval 设置保持 250 条样本、20 条 specificity held-out、5 个 epoch：

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

Qwen3-8B 的命令只需要替换模型名：

```bash
nohup python -u unlearn.py \
  --model_name Qwen/Qwen3-8B \
  --dataset ceval \
  --strategy sentencize --stepwise \
  --method npo_KL --lr 1e-5 --ff2 \
  --max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 \
  --cot_max_new_tokens 300 --eval_max_new_tokens 300 \
  --new_cot \
  > logs/qwen3-8b-ceval-final.log 2>&1 &
```

只跑单个科目时使用 `ceval-<subject>`，例如：

```bash
python unlearn.py \
  --model_name Qwen/Qwen3-4B \
  --dataset ceval-computer_network \
  --max_samples 100 --n_unlearn 80 --verify_samples 20 \
  --lr 1e-5 --ff2 --method npo_KL --new_cot
```

## 结果汇总和可视化

输出会写入本目录下的 `final_cot/`、`final_results/` 等文件夹。旧缓存如果只包含 `<think>`，默认会自动重生成；也可以显式加 `--new_cot`。

运行结束后，用下面命令生成 `Qwen3-4B/summary.json`、`Qwen3-4B/per_step_metrics.csv`、概率转移图、efficacy/mass-shift 散点图和 step salience heatmap：

```bash
python analyze_results.py \
  --result_file final_results/ceval/Qwen3-4B/npo_KL_sentencize_s=True_lr=1e-05_rs=1001_n=250_pos=False_ff2=True.out \
  --out_dir Qwen3-4B
```

当前环境已用本地缓存的 `Qwen/Qwen3-1.7B` 跑通了极小样本闭环，示例输出在 `Qwen3-1.7B-smoke/`；也用 `Qwen/Qwen3-8B` 在 A100 默认 GPU device map 上跑通了 trainable+oracle 微调闭环，示例输出在 `Qwen3-8B-smoke-a100/`。如果必须跑 3B，请提供本地模型路径或镜像仓库名；官方 HF 当前可直接替换为 `Qwen/Qwen3-4B` 或 `Qwen/Qwen3-8B`。
