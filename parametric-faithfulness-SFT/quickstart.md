# Quickstart：DeepSeek-Revision SFT 全量实验

本页给出从当前目录直接启动全量实验、恢复中断任务和读取最终结果的最短路径。
完整研究动机与指标解释见 [readme.md](readme.md)，精确协议见 [实验.md](实验.md)。

## 1. 本次正式运行配置

| 项目 | 设置 |
| --- | --- |
| 工作目录 | `parametric-faithfulness-SFT/` |
| GPU | NVIDIA A100-SXM4-80GB |
| student/base model | `local_models/Llama-3.2-3B-Instruct` |
| dataset | `local_datasets/openbookqa` |
| teacher | DeepSeek API `deepseek-v4-pro` |
| teacher thinking | `enabled`, `reasoning_effort=high`, `max_tokens=4096`（含 thinking token） |
| teacher 数据来源 | OpenBookQA `train` 中固定抽取的 `2000` 题 |
| SFT 类型 | 单次 `down_proj` LoRA-SFT |
| SFT 参数 | `r=32`, `alpha=64`, `lr=1e-4`, `epochs=1`, `bf16` |
| FUR 测试 | OpenBookQA `test`，`100 target + 20 retain` |
| FUR 参数 | `NPO+KL`, `lr=3e-05`, `epochs=5`, `--pos` 等价内容词过滤, FF2 only |
| seed | `1001` |

注意：`3e-05 / 5 epochs` 是原论文用于 FUR 参数干预的最佳设置；论文没有提供
本扩展 SFT 的最优训练参数，因此 SFT 的 `1e-4 / 1 epoch` 是预先固定的扩展设置。

## 2. 目录检查

本项目统一在 Conda 环境 `faith` 中执行。启动脚本已固定调用该环境的 Python：

```bash
conda activate faith
which python
# /inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python
```

进入目录：

```bash
cd parametric-faithfulness-SFT
```

正式实验需要以下资源已经存在：

```bash
test -d local_models/Llama-3.2-3B-Instruct
test -d local_datasets/openbookqa
test -f .env
```

`.env` 中保存 `DEEPSEEK_API_KEY`，并被 `.gitignore` 排除。日志不会记录 API key。

首次运行若提示缺失依赖，可在 `faith` 环境中安装后直接续跑，例如：

```bash
python -m pip install transformers datasets peft accelerate spacy requests tqdm
python -m spacy download en_core_web_sm
```

不要切换到另一个环境重跑中间阶段，以免依赖或 CUDA 配置变化影响对比。

如从另一份工作区重新建立资源副本，可运行：

```bash
python setup_resources.py
```

## 3. 一键启动全量实验

从本目录执行：

```bash
bash scripts/run_full_experiment.sh
```

该脚本顺序执行：

1. 原始 LLaMA-3B 对 `2000` 条 train 题以 greedy 批量解码生成 student drafts（A100 默认 batch `32`）。
2. `deepseek-v4-pro` 以 thinking 模式和有限并发修订每条 draft，过滤得到 SFT 数据，并重试临时失败请求。
3. 检查合格样本数并执行唯一一次 LoRA-SFT，保存合并模型。
4. Base 与 SFT 模型分别在固定 `test 100+20` 划分生成自己的 CoT。
5. 对 Base 与 SFT reasoning 分别执行完整 FUR。
6. 写出最终指标表。

主日志：

```text
logs/run_full_experiment.master.log
```

每阶段日志：

```text
logs/generate_drafts.log
logs/revise_drafts.log
logs/train_lora.log
logs/generate_eval_cots_base.log
logs/generate_eval_cots_sft.log
logs/run_base_fur.log
logs/run_sft_fur.log
```

## 4. 分阶段运行命令

需要手动控制阶段时，可使用以下命令。

### 4.1 生成 student drafts

```bash
bash scripts/generate_training_data.sh drafts
wc -l artifacts/data/train_student_drafts.jsonl
```

期望输出行数为 `2000`；正式脚本使用 `batch_size=32` 加快 A100 上的确定性 greedy 生成。

### 4.2 DeepSeek revisions

```bash
bash scripts/generate_training_data.sh revisions
wc -l artifacts/data/teacher_revisions.jsonl artifacts/data/sft_train.jsonl
```

`teacher_revisions.jsonl` 保存所有已处理请求及过滤状态；
`sft_train.jsonl` 仅保存通过过滤的训练样本。正式 SFT 默认要求至少 `500`
条合格数据，建议保留数达到 `800` 以上。正式脚本最多并发 `8` 个 API 请求，
并以最多 `4` 个并发请求重试第一次运行中记录为 API error 的条目。
正式 teacher revision 必须全部记录 `teacher_model=deepseek-v4-pro`；此前使用其他
teacher 的 smoke 产物位于 `artifacts/smoke/`，不进入正式数据。
V4-Pro 的 thinking token 与最终 JSON 共用 completion budget，因此正式请求将
`max_tokens` 固定为 `4096`，以免短预算截断 JSON。

### 4.3 一次 SFT

```bash
bash scripts/train_once.sh
test -f artifacts/models/deepseek_revision_sft/training_metadata.json
test -d artifacts/models/deepseek_revision_sft/merged/Llama-3.2-3B-Instruct
```

### 4.4 Base/SFT 测试 CoT

```bash
bash scripts/generate_eval_cots.sh base
bash scripts/generate_eval_cots.sh sft
wc -l artifacts/eval_cots/base_openbookqa_test.jsonl artifacts/eval_cots/sft_openbookqa_test.jsonl
```

两份文件均应为 `120` 行；两者使用相同题目 manifest，但 reasoning 来自各自模型。

### 4.5 FUR

整张 A100 80GB 可用时，两组 FUR 可同时运行，且写入不同结果文件：

```bash
bash scripts/run_base_fur.sh &
bash scripts/run_sft_fur.sh &
wait
```

FUR 输出按有效 reasoning step 写入 JSONL，行数取决于生成 CoT 的有效句子数量，
不会固定等于问题数。

### 4.6 汇总

```bash
bash scripts/analyze.sh
cat artifacts/analysis/metrics.md
```

## 5. 断点恢复

下列阶段原生支持重新执行：

| 阶段 | 恢复方式 |
| --- | --- |
| Student drafts | 已写入的 question `id` 自动跳过 |
| DeepSeek revisions | 已请求的 question `id` 自动跳过 |
| Base/SFT CoT 生成 | 已写入的 evaluation question 自动跳过 |
| Base/SFT FUR | 已完成的 `(question_id, step_idx)` 自动跳过 |

若任务中断，直接再次执行：

```bash
bash scripts/run_full_experiment.sh
```

SFT 本身不应无意重复训练。如果合并后的 SFT 模型目录已存在，全量脚本默认跳过
训练阶段；如明确需要重新训练，应先移动或删除
`artifacts/models/deepseek_revision_sft/`，并在实验记录中注明。

## 6. 进度检查

数据与推理进度：

```bash
wc -l artifacts/data/*.jsonl artifacts/eval_cots/*.jsonl artifacts/fur_results/*.jsonl 2>/dev/null
```

GPU 状态：

```bash
nvidia-smi
```

查看主日志尾部：

```bash
tail -f logs/run_full_experiment.master.log
```

## 7. 输出与判读

最终核心产物：

```text
artifacts/analysis/metrics.md
artifacts/analysis/metrics.json
```

表格将比较：

| 指标 | 作用 |
| --- | --- |
| `Direct Acc` / `CoT Acc` | 能力是否保持 |
| `D/C Agree` | FUR agree-only 指标覆盖度 |
| `Avg Steps` | SFT 是否导致 reasoning 膨胀 |
| `Eff` | 遗忘是否有效 |
| `Spec` | 遗忘是否保持局部性 |
| `FF-HARD All/Agree` | 参数干预是否引起答案翻转 |
| `FF-SOFT All/Agree` | 初始答案概率质量下降幅度 |

只有当 SFT 模型的 `FF-HARD`/`FF-SOFT` 提高，同时 `Specificity` 仍约为
`95%` 以上且准确率没有明显下降，才将结果解释为参数忠实性改善的证据。

## 8. 时间提示

数据生成与一次 LoRA-SFT 相对较快；完整 `100+20` FUR 是主要耗时来源，因为
每个可测 reasoning step 都需要独立进行五轮参数干预并评价 specificity。
draft 与 test CoT 解码使用 batch，FUR 的 specificity 推理也批量前向，且每个
arm 只加载一次模型并在每个 step 前恢复原始 `down_proj` 权重；API revision 使用有限网络并发。
单个 FUR arm 实测约占 21GB 显存，整张 A100 80GB 可用时 Base/SFT 可并行运行；
两者分别只读各自模型并写入各自 JSONL，不共享可变参数。输出支持续跑。
