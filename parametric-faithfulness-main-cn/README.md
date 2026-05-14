# C-Eval 中文版实验说明

本目录是在 C-Eval 中文选择题数据集上复现论文
`Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps`
的简化版本。主入口是 `unlearn-cn.py`。

默认配置：

- 数据集：`ceval/ceval-exam`
- 默认任务：`ceval`，会从 C-Eval 全科验证集中按随机种子抽样
- 默认样本数：`--max_samples 250`，与原项目每个数据集最多 250 条保持一致
- 模型：`Qwen/Qwen3-8B`、`Qwen/Qwen3-3B`、`Qwen/Qwen3-1.7B`
- 中文 CoT：使用中文 prompt，并对中文标点做步骤切分

如果你的 Hugging Face 环境中 3B 仓库名不同，直接把 `--model_name` 换成本地模型路径或实际仓库名即可。

示例：

```powershell
cd parametric-faithfulness-main-cn
python unlearn-cn.py --model_name Qwen/Qwen3-8B --dataset ceval --strategy sentencize --stepwise --lr 1e-5 --ff2 --method npo_KL
python unlearn-cn.py --model_name Qwen/Qwen3-3B --dataset ceval --strategy sentencize --stepwise --lr 1e-5 --ff2 --method npo_KL
python unlearn-cn.py --model_name Qwen/Qwen3-1.7B --dataset ceval --strategy sentencize --stepwise --lr 3e-5 --ff2 --method npo_KL
```

只跑单个科目时使用 `ceval-<subject>`，例如：

```powershell
python unlearn-cn.py --model_name Qwen/Qwen3-1.7B --dataset ceval-computer_network --max_samples 100 --lr 3e-5 --ff2 --method npo_KL
```

输出会写入本目录下的 `final_cot/`、`final_results/` 等文件夹。首次运行会生成 CoT 缓存；如果需要重新随机生成，加入 `--new_cot`。
