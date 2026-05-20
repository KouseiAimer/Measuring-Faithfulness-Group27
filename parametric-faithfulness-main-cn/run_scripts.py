"""Print local A100 run commands for the Chinese C-Eval experiments."""

MODELS = {
    "Qwen/Qwen3-8B": {"lr": "1e-5", "shards": 1},
    "Qwen/Qwen3-4B": {"lr": "1e-5", "shards": 1},
    "Qwen/Qwen3-1.7B": {"lr": "3e-5", "shards": 2},
}

DATASETS = ["ceval"]


def short_name(model_id):
    return model_id.split("/")[-1].lower()


def command(model_id, dataset, lr, num_shards, shard_idx):
    shard_args = ""
    shard_label = ""
    if num_shards > 1:
        shard_args = f" --num_shards {num_shards} --shard_idx {shard_idx}"
        shard_label = f"-shard-{shard_idx}"
    log_name = f"logs/{short_name(model_id)}-{dataset}{shard_label}.log"
    return (
        "CUDA_VISIBLE_DEVICES=0 nohup python -u unlearn.py "
        f"--model_name {model_id} "
        f"--dataset {dataset} "
        "--strategy sentencize --stepwise "
        f"--method npo_KL --lr {lr} --ff2 "
        "--max_samples 250 --n_unlearn 250 --verify_samples 20 --epochs 5 "
        "--cot_max_new_tokens 300 --eval_max_new_tokens 300 "
        f"--new_cot{shard_args} "
        f"> {log_name} 2>&1 &"
    )


if __name__ == "__main__":
    print("mkdir -p logs")
    for dataset in DATASETS:
        for model_id, cfg in MODELS.items():
            for shard_idx in range(cfg["shards"]):
                print(command(model_id, dataset, cfg["lr"], cfg["shards"], shard_idx))
