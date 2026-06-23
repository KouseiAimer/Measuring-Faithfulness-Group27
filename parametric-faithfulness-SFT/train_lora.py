"""Run the single down_proj LoRA-SFT experiment and save a merged checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from sft_common import ARTIFACTS, DEFAULT_BASE_MODEL, read_jsonl, set_seed


class RationaleDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt_ids = self.tokenizer.encode(row["prompt"], add_special_tokens=False)
        completion = row["completion"].strip() + self.tokenizer.eos_token
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        input_ids = (prompt_ids + completion_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class CausalCompletionCollator:
    tokenizer: object

    def __call__(self, examples: list[dict]) -> dict:
        max_len = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_mask = []
        labels = []
        for example in examples:
            padding = max_len - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(example["attention_mask"] + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-data", type=Path, default=ARTIFACTS / "data" / "sft_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "models" / "deepseek_revision_sft")
    parser.add_argument("--min-records", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    rows = read_jsonl(args.train_data)
    if len(rows) < args.min_records:
        raise SystemExit(
            f"Only {len(rows)} accepted SFT records found; require at least {args.min_records}. "
            "Generate more accepted teacher revisions or lower --min-records for a declared pilot."
        )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dataset = RationaleDataset(rows, tokenizer, args.max_length)
    update_steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * args.gradient_accumulation))
    total_update_steps = max(1, math.ceil(update_steps_per_epoch * args.epochs))
    warmup_steps = round(total_update_steps * 0.03)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCompletionCollator(tokenizer),
    )
    result = trainer.train()
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model.config.use_cache = True
    merged_model = model.merge_and_unload()
    merged_dir = args.output_dir / "merged" / "Llama-3.2-3B-Instruct"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    metadata = {
        "base_model": str(args.model),
        "train_data": str(args.train_data),
        "n_records": len(rows),
        "lora": {
            "target_modules": ["down_proj"],
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "seed": args.seed,
            "warmup_steps": warmup_steps,
        },
        "trainer_metrics": result.metrics,
        "merged_model": str(merged_dir),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Merged SFT model saved to {merged_dir}.")


if __name__ == "__main__":
    main()
