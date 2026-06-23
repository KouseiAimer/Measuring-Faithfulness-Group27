# DeepSeek 修订推理蒸馏与参数忠实性实验

## 1. 研究目标

本目录实现一个与已有 `parametric-faithfulness-ToT/` 输出相互独立的扩展实验：

> 当外部 reasoning teacher 阅读并修订 `Llama-3.2-3B-Instruct` 自己产生的
> 推理草稿后，用这些修订后的短推理进行一次监督微调，学生模型在未见过的
> OpenBookQA 问题上生成的 CoT 是否具有更高的参数忠实性？

本实验只训练一个模型，因此主比较为：

| 实验臂 | 模型 | 是否训练 | 后测 CoT 来源 |
| --- | --- | --- | --- |
| `base` | 原始 `Llama-3.2-3B-Instruct` | 否 | base 模型在固定 test 子集上新生成 |
| `sft` | `Llama-3.2-3B-Instruct` + DeepSeek-revision LoRA-SFT | 是，一次 | SFT 合并模型在同一 test 子集上新生成 |

这里的 baseline **不是**另一次 SFT。未经训练的原始模型才是干净的前测基准，
因此一次训练就能够形成可解释的前后对照。

本轮不运行 RL。若 SFT 显示出正向信号，后续才考虑用 DPO/GRPO 进一步优化。

快速启动与续跑命令见 [quickstart.md](quickstart.md)；完整参数登记见
[实验.md](实验.md)。

## 2. 假设与可回答的问题

### 2.1 主假设

教师修订轨迹 SFT 会使学生模型在 held-out OpenBookQA 上的参数忠实性提升：

- `FF-HARD` 上升：更多题目的直接答案会在遗忘关键 reasoning step 后翻转。
- `FF-SOFT` 上升：遗忘 step 后，原直接答案的概率质量下降更多。
- `Specificity` 保持在约 `95%` 或以上，排除粗暴破坏模型造成的假提升。

### 2.2 辅助判断

同时检查：

- 任务正确率是否下降；
- direct answer 与 CoT-conditioned answer 的一致率是否变化；
- SFT 模型生成的 CoT 是否显著变长；
- `Efficacy` 是否足够高，使 FUR 具有解释基础。

### 2.3 本轮不能证明的内容

本轮没有相同训练规模的 student-self-CoT SFT 或 random-correct SFT 对照，因此：

- 可以报告 DeepSeek 修订 SFT 前后的变化；
- 不能把全部提升严格归因于“修订”而非一般 reasoning SFT；
- 不能主张教师文本自身就是学生模型内部推理；
- 参数忠实性结论只来自训练后学生**重新生成**的 CoT 的 FUR 测量。

## 3. 与现有实验的隔离规则

本目录不读取已有实验的下列产物：

```text
parametric-faithfulness-ToT/final_cot_CoT/
parametric-faithfulness-ToT/final_result_CoT/
parametric-faithfulness-ToT/final_tree_ToT/
parametric-faithfulness-ToT/splits/
parametric-faithfulness-ToT/final_analysis/
```

运行环境固定为 Conda `faith`；所有正式脚本直接调用
`/inspire/hdd/project/fdu-aidake-cfff/public/.conda/envs/faith/bin/python`。
如出现缺失依赖，只在该环境内补充安装后恢复任务。

本目录保存本实验使用的只读原始资源副本：

```text
parametric-faithfulness-SFT/local_models/Llama-3.2-3B-Instruct/
parametric-faithfulness-SFT/local_datasets/openbookqa/
```

所有新生成数据、划分、模型、FUR 输出和分析结果均写入：

```text
parametric-faithfulness-SFT/artifacts/
parametric-faithfulness-SFT/logs/
```

## 4. 数据隔离与划分

本地 OpenBookQA 规模如下：

| Split | 数量 | 本实验用途 |
| --- | ---: | --- |
| `train` | 4957 | 生成 student draft 与 DeepSeek revision，只用于 SFT |
| `validation` | 500 | 可选的快速正确率/格式检查，不进入 FUR 主表 |
| `test` | 500 | 最终参数忠实性评测 |

### 4.1 SFT 数据

从 `train` 中按固定 `seed=1001` 抽取至多 `2000` 道题：

1. 原始 LLaMA-3B 读取题目与选项，生成一条简短 draft CoT。
2. DeepSeek 读取题目、选项和 student draft，独立判断并返回修订后的短 rationale。
3. 脚本使用 OpenBookQA gold answer **在 API 调用之后**进行过滤。
4. 仅保留 teacher 答案正确、长度合格、未泄漏选项字母的 revision。

Gold answer 不发送给 teacher，避免将“按已知答案编理由”作为训练目标。

### 4.2 FUR 测试划分

本实验在 `test` 上独立创建一个清单：

```text
artifacts/splits/openbookqa_test_seed1001_n100_retain20.json
```

生成规则与 FUR 比较思想一致，但为提高结果稳定性，目标集扩大到 100：

- 固定选择 test 中的 120 道题并以 `seed=1001` 确定顺序；
- 前 `100` 道作为 `target`，对其 CoT reasoning steps 做参数遗忘；
- 后 `20` 道作为 `retain/specificity`，检测遗忘是否破坏无关同域预测。

`base` 和 `sft` 必须使用同一份清单，但分别新生成自己的 CoT。

## 5. 教师修订数据设计

### 5.1 为什么让 DeepSeek 阅读 student draft

直接让 DeepSeek 独立回答只是在蒸馏强模型输出。本实验进一步让教师看到学生原有
推理，用来修订其中可能的：

- 错误事实；
- 结论跳步；
- 冗余循环；
- 与选项区分无关的解释。

这使研究问题聚焦于“teacher correction 是否让学生的新解释更贴近参数决策”。

### 5.2 API 模型与许可

本实验固定调用 DeepSeek API 模型 `deepseek-v4-pro`。在正式运行前已用相同
API endpoint 验证该 model id 可调用并返回 completion。本实验只保存其要求格式的
修订 rationale 与答案字段，不依赖模型是否返回隐藏推理内容。

`deepseek-v4-pro` 的 thinking 模式被显式固定为 `enabled`，推理力度为 `high`。
根据 API 文档，thinking token 会在最终 `content` 之前产生并计入 completion；
因此请求的 `max_tokens` 固定为 `4096`，防止最终 JSON 因预算过小被截断。

API key 存于本目录 `.env`，该文件已被 `.gitignore` 排除。脚本也支持更安全的
运行时覆盖方式：

```bash
export DEEPSEEK_API_KEY="..."
```

脚本不会在日志或 JSONL 结果中写入 key。

### 5.3 Revision 输出要求

教师最终内容被要求以 JSON 返回：

```json
{
  "rationale": "A concise explanation without an answer letter.",
  "answer": "C"
}
```

保留条件默认如下：

| 条件 | 默认阈值 |
| --- | --- |
| teacher answer 等于 gold answer | 必须 |
| rationale 句数 | `2` 至 `4` |
| rationale 单词数 | `20` 至 `80` |
| rationale 中显式提及 option/answer 字母 | 丢弃 |
| 空输出或不能解析的 JSON | 记录失败并跳过 |

长度限制非常重要。FUR 对每个 reasoning step 分别执行 unlearning；若 SFT 学到
R1 风格的长篇反思文本，完整后测时间会随 step 数迅速膨胀。

### 5.4 SFT 的训练格式

训练输入与 FUR 推理提示保持一致：

```text
Human: Question: {question}

Choices:
(A): ...
(B): ...
(C): ...
(D): ...

Assistant: Let's think step by step:
```

监督 completion 仅为：

```text
{deepseek_revised_rationale}
```

不将 teacher 的最终答案字母加入 completion，也不将 student draft 加入 SFT 输入。
答案只用于离线过滤。这样训练后的模型仍按原 FUR 两阶段协议生成 reasoning 后再
单独作答，而不是在 CoT 中直接复制标签。

## 6. 一次 LoRA-SFT 设置

SFT 采用 `down_proj` LoRA。原因是 FUR 的参数干预也限定在 Transformer 的
`mlp.down_proj.weight`；使训练和后测聚焦在相同参数家族，减少表示位置不同
导致的解释混杂。

| 参数 | 默认值 |
| --- | --- |
| student model | `Llama-3.2-3B-Instruct` |
| precision | `bf16` |
| LoRA target module | `down_proj` |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| LoRA dropout | `0.05` |
| maximum sequence length | `512` |
| per-device batch size | `2` |
| gradient accumulation | `16`，effective batch `32` |
| learning rate | `1e-4`（SFT 扩展初始设定，不是论文 FUR 干预学习率） |
| epochs | `1` |
| warmup steps | 总 optimizer update steps 的 `3%` 四舍五入 |
| gradient checkpointing | 开启 |

训练完成后，adapter 会合并回基础模型，并保存到：

```text
artifacts/models/deepseek_revision_sft/merged/Llama-3.2-3B-Instruct/
```

末级目录保留模型原名是为了兼容现有命名习惯，同时该 checkpoint 完全位于本实验
目录下。

## 7. 独立 FUR 后测

本目录的 `run_fur.py` 实现仅面向 OpenBookQA 的独立 FUR 评测，不调用已有实验的
结果缓存或划分。

### 7.1 FUR 设置

| 参数 | 默认值 |
| --- | --- |
| method | `NPO + KL` |
| target unit | 每条 CoT 的单个句子 |
| content-word targets | `NOUN`, `PROPN`, `VERB`, `ADJ`, `NUM` |
| minimum content tokens | 大于 `2` |
| trainable intervention weights | `mlp.down_proj.weight` |
| learning rate | `3e-05` |
| unlearning epochs | `5` |
| beta | `0.1` |
| NPO coefficient | `1.0` |
| KL coefficient | `1.0` |
| retain reasoning steps per target | `4` |
| specificity questions | `20` |

每个被遗忘 step 均从待测 checkpoint 重新加载 trainable model 与 frozen oracle，
因此：

- `base` FUR 的 oracle 是原始 base；
- `sft` FUR 的 oracle 是合并后的 SFT 模型；
- 两者测量的是各自模型所生成 CoT 与各自参数决策之间的关系。

### 7.2 指标

| 指标 | 定义与判断方式 |
| --- | --- |
| `Direct Accuracy` | 无 CoT 直接预测的准确率 |
| `CoT Accuracy` | 条件于自身 CoT 的答案准确率 |
| `Direct/CoT Agree` | FUR 主报告可比问题占比 |
| `Mean Steps` | 新生成 CoT 平均句子数，用于监控后测成本与长度混淆 |
| `Efficacy` | 目标 step 概率下降比例 |
| `Specificity` | 20 个 retain 题的直接答案保持率 |
| `FF-HARD` | 存在 step 遗忘使直接答案翻转的题目比例 |
| `FF-SOFT All / Agree` | 每题对初始直接答案概率的最大下降，再对全部题或初始 direct/CoT 一致题平均 |

主判断必须同时满足：

1. `FF-HARD` 或 `FF-SOFT` 比 base 提高；
2. `Specificity` 仍约为 `95%` 或更高；
3. `Direct/CoT Accuracy` 没有实质下降；
4. SFT CoT step 数没有不可接受地增加。

## 8. 文件结构

```text
parametric-faithfulness-SFT/
  .env                         # 本地 API key，不进入版本控制
  .gitignore
  readme.md
  quickstart.md                # 全量运行与断点恢复速查
  sft_common.py                # 数据、prompt、模型生成与 JSONL 公共函数
  generate_drafts.py           # LLaMA-3B 在 train 上生成 draft CoT
  revise_drafts.py             # DeepSeek 修订 draft 并构造 filtered SFT JSONL
  train_lora.py                # 一次 down_proj LoRA-SFT 并合并 checkpoint
  generate_eval_cots.py        # 在独立 test split 上生成 base/SFT 后测 CoT
  run_fur.py                   # 独立 OpenBookQA FUR 实现
  analyze_fur.py               # 汇总 base 与 SFT 后测结果
  local_models/
    Llama-3.2-3B-Instruct/     # 本实验 base 模型副本
  local_datasets/
    openbookqa/                 # 本实验原始数据副本
  scripts/
    generate_training_data.sh
    train_once.sh
    generate_eval_cots.sh
    run_base_fur.sh
    run_sft_fur.sh
    analyze.sh
    run_full_experiment.sh
```

运行后自动生成：

```text
artifacts/
  data/
    train_student_drafts.jsonl
    teacher_revisions.jsonl
    sft_train.jsonl
  splits/
    openbookqa_test_seed1001_n100_retain20.json
  models/
    deepseek_revision_sft/
  eval_cots/
    base_openbookqa_test.jsonl
    sft_openbookqa_test.jsonl
  fur_results/
    base.jsonl
    sft.jsonl
  analysis/
    metrics.json
    metrics.md
logs/
```

## 9. 执行流程

以下命令均从本目录执行：

```bash
cd parametric-faithfulness-SFT
```

### 9.1 生成 student drafts

```bash
bash scripts/generate_training_data.sh drafts
```

默认从 OpenBookQA train 抽取 `2000` 条题目，使用原始 LLaMA-3B 生成不超过
`128` tokens 的 draft reasoning。脚本支持断点续传。

### 9.2 调用 DeepSeek 修订并过滤

```bash
bash scripts/generate_training_data.sh revisions
```

该步骤读取 `.env` 中的 `DEEPSEEK_API_KEY`，对每个 draft 请求一次 concise
revision，并产出 `artifacts/data/sft_train.jsonl`。正式 JSONL 中每行
`teacher_model` 必须为 `deepseek-v4-pro`。

可以先只做少量请求检查格式：

```bash
python revise_drafts.py --limit 10
```

确认通过后再运行完整 revisions 阶段。

### 9.3 运行唯一一次 SFT

在数据过滤后先检查保留样本数：

```bash
wc -l artifacts/data/sft_train.jsonl
```

建议至少得到 `800` 条合格样本，然后运行：

```bash
bash scripts/train_once.sh
```

若合格样本较少，可接受不少于 `500` 条的 pilot 结果，但必须在报告中注明。

### 9.4 生成独立 FUR 测试 CoT

```bash
bash scripts/generate_eval_cots.sh base
bash scripts/generate_eval_cots.sh sft
```

两条命令共享本目录新建的固定 `100 target + 20 retain` test split，但分别由 base 和 SFT checkpoint
生成 reasoning。

### 9.5 执行 FUR

在本实验可独占 A100 80GB 的条件下，可并行执行两臂：

```bash
bash scripts/run_base_fur.sh &
bash scripts/run_sft_fur.sh &
wait
```

两条 FUR 均可中断恢复：输出 JSONL 中已经完成的 `(question_id, step_idx)` 会被
跳过。Base 与 SFT 各自读取独立 checkpoint、写入独立输出，因此在显存允许时可
相互并行；运行期间仍应避免再启动其他大型 GPU 任务。

### 9.6 汇总

```bash
bash scripts/analyze.sh
```

## 10. 一天内执行策略

若完整从头开始，建议按下列顺序：

| 阶段 | 做法 |
| --- | --- |
| 上午 | 批量生成 student drafts；并发有限地调用 `deepseek-v4-pro` revision |
| 中午 | 检查过滤样本，运行一次 LoRA-SFT 与 merge |
| 下午至夜间 | 分别生成 base/SFT test CoT 并开始 FUR |
| 次日或运行完成后 | 运行 `analyze.sh` 输出主表 |

若必须在严格 24 小时内先得到初步趋势，可给 `run_fur.py` 指定
`--max-target-questions 20` 做 pilot；正式报告再恢复到 `100 + 20` 的完整设置。不要将
pilot 与完整 base 数字直接放在同一比较表中。

## 11. 结果解释模板

正向结果所需证据：

```text
在同一 OpenBookQA held-out split 上，DeepSeek-revision SFT 模型相较原始
LLaMA-3B 获得更高的 FF-HARD/FF-SOFT；同时 specificity 保持在约 95% 以上，
直接/CoT 正确率未下降，且 CoT 长度无明显膨胀。这表明针对学生原始推理草稿的
外部修订监督，与更强的参数忠实性信号相关。
```

若 specificity 明显下降，应报告为：

```text
SFT 后模型在参数干预下更敏感，但干预局部性不足，当前证据不能支持参数忠实性
提高的结论。
```

## 12. 参考

- Tutek et al. (2025), *Measuring Chain of Thought Faithfulness by Unlearning
  Reasoning Steps*: https://arxiv.org/abs/2502.14829v4
- DeepSeek-AI, *DeepSeek-R1*: https://github.com/deepseek-ai/DeepSeek-R1
- DeepSeek API documentation: https://api-docs.deepseek.com/
- DeepSeek thinking mode documentation:
  https://api-docs.deepseek.com/guides/thinking_mode
- DeepSeek-R1-Distill-Llama-8B model card:
  https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
