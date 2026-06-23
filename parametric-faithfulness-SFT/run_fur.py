"""Run independent paper-style FUR on base or DeepSeek-revision SFT CoTs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_common import (
    ARTIFACTS,
    DEFAULT_BASE_MODEL,
    generate_rationale,
    option_probabilities,
    option_probabilities_batch,
    read_jsonl,
)


DEFAULT_SFT_MODEL = ARTIFACTS / "models" / "deepseek_revision_sft" / "merged" / "Llama-3.2-3B-Instruct"
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
IGNORE_INDEX = -100


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.clean_up_tokenization_spaces = False
    return tokenizer


def load_model(path: Path):
    return AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, dtype=torch.bfloat16, device_map="auto"
    )


def content_spans(text: str, nlp) -> list[tuple[int, int]]:
    return [(token.idx, token.idx + len(token.text)) for token in nlp(text) if token.pos_ in CONTENT_POS]


def encoded_pair(tokenizer, prompt: str, completion: str, device, nlp, content_only: bool) -> dict:
    full_text = prompt + completion
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    prefix_len = len(prompt)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    if content_only:
        spans = [(prefix_len + start, prefix_len + end) for start, end in content_spans(completion, nlp)]
        for index, (start, end) in enumerate(offsets):
            if any(start < target_end and end > target_start for target_start, target_end in spans):
                labels[0, index] = input_ids[0, index]
    else:
        for index, (start, end) in enumerate(offsets):
            if end > prefix_len and end > start:
                labels[0, index] = input_ids[0, index]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def target_count(batch: dict) -> int:
    return int((batch["labels"] != IGNORE_INDEX).sum().item())


def token_losses(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].transpose(1, 2),
        labels[:, 1:],
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )


def sequence_nll(model, batch: dict) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return token_losses(logits, batch["labels"]).sum(dim=-1)


@torch.no_grad()
def sequence_log_probability(model, batch: dict) -> float:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    losses = token_losses(logits, batch["labels"])
    count = (batch["labels"][:, 1:] != IGNORE_INDEX).sum().clamp_min(1)
    return float((-losses.sum() / count).float().cpu())


def npo_kl_loss(model, oracle, forget: dict, retain: dict, beta: float, kl_coeff: float) -> torch.Tensor:
    current_forget = sequence_nll(model, forget)
    with torch.no_grad():
        oracle_forget = sequence_nll(oracle, forget)
    neg_log_ratios = current_forget - oracle_forget
    forget_loss = -2.0 / beta * F.logsigmoid(beta * neg_log_ratios).mean()

    current_logits = model(
        input_ids=retain["input_ids"], attention_mask=retain["attention_mask"]
    ).logits.float()
    with torch.no_grad():
        oracle_logits = oracle(
            input_ids=retain["input_ids"], attention_mask=retain["attention_mask"]
        ).logits.float()
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    oracle_log_probs = F.log_softmax(oracle_logits, dim=-1)
    retain_loss = F.kl_div(current_log_probs, oracle_log_probs, log_target=True, reduction="batchmean")
    return forget_loss + kl_coeff * retain_loss


def step_prefix(row: dict, step_index: int) -> str:
    previous = row["segmented_cot"][:step_index]
    if not previous:
        return row["cot_prompt"]
    return row["cot_prompt"] + "\n".join(previous) + "\n"


def cot_prompt_from_eval(row: dict) -> str:
    choices = "\n".join(row["options"])
    return (
        f"Human: Question: {row['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Assistant: Let's think step by step:\n"
    )


def augment_eval_rows(rows: list[dict]) -> None:
    for row in rows:
        row["cot_prompt"] = cot_prompt_from_eval(row)


def valid_step_candidates(rows: list[dict], exclude_id: str, tokenizer, device, nlp) -> list[tuple[dict, int]]:
    candidates = []
    for row in rows:
        if row["id"] == exclude_id:
            continue
        for index, step in enumerate(row["segmented_cot"]):
            batch = encoded_pair(tokenizer, step_prefix(row, index), step, device, nlp, content_only=True)
            if target_count(batch) > 2:
                candidates.append((row, index))
    return candidates


@torch.no_grad()
def evaluate_state(model, tokenizer, target: dict, step_index: int, retain_rows: list[dict], nlp, new_cot_tokens: int) -> dict:
    device = next(model.parameters()).device
    step = target["segmented_cot"][step_index]
    probability_batch = encoded_pair(tokenizer, step_prefix(target, step_index), step, device, nlp, content_only=False)
    evaluation_instances = [target["raw_instance"]] + [row["raw_instance"] for row in retain_rows]
    evaluation_probs = option_probabilities_batch(
        model, tokenizer, evaluation_instances, [None] * len(evaluation_instances)
    )
    direct_probs = evaluation_probs[0]
    specificity_probs = evaluation_probs[1:]
    specificity_preds = [int(np.argmax(values)) for values in specificity_probs]
    new_cot = generate_rationale(
        model, tokenizer, target["raw_instance"], max_new_tokens=new_cot_tokens, temperature=0.0
    )
    new_cot_probs = option_probabilities(model, tokenizer, target["raw_instance"], rationale=new_cot)
    return {
        "probs": direct_probs,
        "prediction": int(np.argmax(direct_probs)),
        "specificity_probs": specificity_probs,
        "specificity_preds": specificity_preds,
        "new_cot": new_cot,
        "new_cot_probs": new_cot_probs,
        "cot_step_logprob": sequence_log_probability(model, probability_batch),
        "target_cot_step": step,
    }


def run_one_step(
    model,
    oracle,
    original_trainable: dict[str, torch.Tensor],
    tokenizer,
    target: dict,
    step_index: int,
    all_rows: list[dict],
    retain_rows: list[dict],
    nlp,
    args,
) -> dict | None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in original_trainable:
                parameter.copy_(original_trainable[name])
    model.eval()
    device = next(model.parameters()).device
    forget = encoded_pair(
        tokenizer,
        step_prefix(target, step_index),
        target["segmented_cot"][step_index],
        device,
        nlp,
        content_only=True,
    )
    if target_count(forget) <= args.min_content_tokens:
        return None

    candidates = valid_step_candidates(all_rows, target["id"], tokenizer, device, nlp)
    rng = random.Random(stable_seed(args.seed, target["id"], step_index))
    rng.shuffle(candidates)
    selected = candidates[: args.n_retain_steps]
    if not selected:
        raise ValueError("No valid retain reasoning steps found.")
    retain_batches = [
        encoded_pair(tokenizer, step_prefix(row, index), row["segmented_cot"][index], device, nlp, content_only=True)
        for row, index in selected
    ]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    epochs = {}
    epochs["0"] = evaluate_state(model, tokenizer, target, step_index, retain_rows, nlp, args.new_cot_tokens)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        retain = retain_batches[(epoch - 1) % len(retain_batches)]
        loss = npo_kl_loss(model, oracle, forget, retain, args.beta, args.kl_coeff)
        loss.backward()
        optimizer.step()
        model.eval()
        epochs[str(epoch)] = evaluate_state(
            model, tokenizer, target, step_index, retain_rows, nlp, args.new_cot_tokens
        )
        epochs[str(epoch)]["loss"] = float(loss.detach().float().cpu())
    del optimizer
    model.zero_grad(set_to_none=True)
    return {
        "id": target["id"],
        "arm": args.arm,
        "question": target["question"],
        "correct_letter": target["correct_letter"],
        "initial_direct_probs": target["direct_probs"],
        "initial_cot_probs": target["cot_probs"],
        "direct_answer": target["direct_answer"],
        "cot_answer": target["cot_answer"],
        "step_idx": step_index,
        "cot_step": target["segmented_cot"][step_index],
        "segmented_cot": target["segmented_cot"],
        "epochs": epochs,
    }


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["base", "sft"], required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--cots", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl-coeff", type=float, default=1.0)
    parser.add_argument("--n-retain-steps", type=int, default=4)
    parser.add_argument("--min-content-tokens", type=int, default=2)
    parser.add_argument("--new-cot-tokens", type=int, default=128)
    parser.add_argument("--max-steps-per-question", type=int, default=0)
    parser.add_argument("--max-target-questions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    model_path = args.model or (DEFAULT_BASE_MODEL if args.arm == "base" else DEFAULT_SFT_MODEL)
    cots_path = args.cots or ARTIFACTS / "eval_cots" / f"{args.arm}_openbookqa_test.jsonl"
    output = args.output or ARTIFACTS / "fur_results" / f"{args.arm}.jsonl"
    rows = read_jsonl(cots_path)
    if not rows:
        raise SystemExit(f"No evaluation CoTs found: {cots_path}")
    augment_eval_rows(rows)
    targets = [row for row in rows if row["role"] == "target"]
    if args.max_target_questions:
        targets = targets[: args.max_target_questions]
    retain_rows = [row for row in rows if row["role"] == "retain"]
    if not retain_rows:
        raise SystemExit("No specificity retain rows in evaluation CoT file.")
    done = {(row["id"], int(row["step_idx"])) for row in read_jsonl(output)}
    tokenizer = load_tokenizer(model_path)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    model = load_model(model_path)
    oracle = load_model(model_path)
    oracle.eval()
    for parameter in oracle.parameters():
        parameter.requires_grad = False
    original_trainable = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "mlp.down_proj.weight" in name
        if parameter.requires_grad:
            original_trainable[name] = parameter.detach().clone()
    print(f"{args.arm}: {len(targets)} target questions, {len(retain_rows)} specificity questions; output={output}")
    for target in tqdm(targets, desc=f"{args.arm} FUR questions"):
        steps = list(range(len(target["segmented_cot"])))
        if args.max_steps_per_question:
            steps = steps[: args.max_steps_per_question]
        for step_index in steps:
            if (target["id"], step_index) in done:
                continue
            result = run_one_step(
                model, oracle, original_trainable, tokenizer, target, step_index, rows, retain_rows, nlp, args
            )
            if result is not None:
                append_result(output, result)
    del model, oracle, original_trainable
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"FUR run complete or resumed through available targets: {output}.")


if __name__ == "__main__":
    main()
