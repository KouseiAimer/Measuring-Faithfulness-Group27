**实验及训练设置**

| 项目 | 设置 |
| --- | --- |
| 基础模型 | Llama-3.2-3B-Instruct |
| Teacher | deepseek-v4-pro (thinking=high) |
| 训练数据 | OpenBookQA train; 2000 drafts; 1852 accepted revisions |
| SFT 方法 | LoRA, target=down_proj, rank=32, alpha=64, dropout=0.05 |
| SFT 超参数 | lr=1e-4; epoch=1; batch=2; grad_acc=16; max_len=512 |
| FUR 测试集 | OpenBookQA test: 100 target + 20 retain; seed=1001 |
| FUR 超参数 | NPO+KL; lr=3e-5; epoch=5; beta=0.1; KL=1.0 |
| FUR 可训练参数 | mlp.down_proj.weight only |
