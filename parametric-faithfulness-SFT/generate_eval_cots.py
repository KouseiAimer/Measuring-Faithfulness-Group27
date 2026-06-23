"""Generate independent base or SFT CoTs on a fixed OpenBookQA test split."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from sft_common import (
    ARTIFACTS,
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET,
    answer_choices,
    append_jsonl,
    configure_tokenizer,
    generate_rationales,
    load_causal_model,
    openbook_dataset,
    option_probabilities_batch,
    predicted_letter,
    read_jsonl,
    sentence_tokenize,
    set_seed,
)


DEFAULT_SFT_MODEL = ARTIFACTS / "models" / "deepseek_revision_sft" / "merged" / "Llama-3.2-3B-Instruct"


def create_or_load_manifest(
    path: Path, test_rows: list[dict], seed: int, n_targets: int, n_retain: int
) -> dict:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("seed") != seed
            or len(manifest.get("target_ids", [])) != n_targets
            or len(manifest.get("retain_ids", [])) != n_retain
        ):
            raise ValueError(f"Existing manifest does not match requested configuration: {path}")
        return manifest
    shuffled = list(test_rows)
    random.Random(seed).shuffle(shuffled)
    chosen = shuffled[: n_targets + n_retain]
    manifest = {
        "dataset": "openbookqa",
        "split": "test",
        "seed": seed,
        "n_targets": n_targets,
        "n_retain": n_retain,
        "target_ids": [str(row["id"]) for row in chosen[:n_targets]],
        "retain_ids": [str(row["id"]) for row in chosen[n_targets:]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["base", "sft"], required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=ARTIFACTS / "splits" / "openbookqa_test_seed1001_n100_retain20.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-targets", type=int, default=100)
    parser.add_argument("--n-retain", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_path = args.model or (DEFAULT_BASE_MODEL if args.arm == "base" else DEFAULT_SFT_MODEL)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    output = args.output or ARTIFACTS / "eval_cots" / f"{args.arm}_openbookqa_test.jsonl"
    set_seed(args.seed)
    test_rows = list(openbook_dataset(args.dataset)["test"])
    manifest = create_or_load_manifest(args.manifest, test_rows, args.seed, args.n_targets, args.n_retain)
    by_id = {str(row["id"]): row for row in test_rows}
    ordered = [
        (qid, "target") for qid in manifest["target_ids"]
    ] + [(qid, "retain") for qid in manifest["retain_ids"]]
    existing = {str(row["id"]) for row in read_jsonl(output)}
    pending = [(qid, role) for qid, role in ordered if qid not in existing]
    print(f"{args.arm}: {len(ordered)} evaluation questions; {len(pending)} remain.")
    if not pending:
        return

    tokenizer = configure_tokenizer(model_path)
    model = load_causal_model(model_path)
    batches = range(0, len(pending), args.batch_size)
    for start in tqdm(batches, total=(len(pending) + args.batch_size - 1) // args.batch_size, desc=f"{args.arm} CoT batches"):
        selected = pending[start : start + args.batch_size]
        instances = [by_id[qid] for qid, _ in selected]
        rationales = generate_rationales(model, tokenizer, instances, max_new_tokens=args.max_new_tokens)
        direct_values = option_probabilities_batch(model, tokenizer, instances, [None] * len(instances))
        cot_values = option_probabilities_batch(model, tokenizer, instances, rationales)
        for (qid, role), instance, rationale, direct_probs, cot_probs in zip(
            selected, instances, rationales, direct_values, cot_values
        ):
            append_jsonl(
                output,
                {
                    "id": qid,
                    "role": role,
                    "arm": args.arm,
                    "model_path": str(model_path),
                    "question": instance["question_stem"],
                    "options": answer_choices(instance),
                    "correct_letter": str(instance["answerKey"]),
                    "cot": rationale,
                    "segmented_cot": sentence_tokenize(rationale),
                    "direct_probs": direct_probs,
                    "direct_answer": predicted_letter(instance, direct_probs),
                    "cot_probs": cot_probs,
                    "cot_answer": predicted_letter(instance, cot_probs),
                    "raw_instance": instance,
                },
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote independent evaluation CoTs to {output}.")


if __name__ == "__main__":
    main()
