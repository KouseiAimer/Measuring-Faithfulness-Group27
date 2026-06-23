"""Generate LLaMA-3B draft rationales on OpenBookQA train for teacher revision."""

from __future__ import annotations

import argparse
import gc
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
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "data" / "train_student_drafts.jsonl")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    train = list(openbook_dataset(args.dataset)["train"])
    rng = random.Random(args.seed)
    rng.shuffle(train)
    selected = train[: args.max_samples]
    completed = {str(row["id"]) for row in read_jsonl(args.output)}
    remaining = [row for row in selected if str(row["id"]) not in completed]
    print(f"Selected {len(selected)} train questions; {len(remaining)} drafts remain.")
    if not remaining:
        return

    tokenizer = configure_tokenizer(args.model)
    model = load_causal_model(args.model)
    for start in tqdm(range(0, len(remaining), args.batch_size), desc="student draft batches"):
        batch = remaining[start : start + args.batch_size]
        rationales = generate_rationales(
            model, tokenizer, batch, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        direct_values = option_probabilities_batch(model, tokenizer, batch, [None] * len(batch))
        cot_values = option_probabilities_batch(model, tokenizer, batch, rationales)
        for instance, rationale, direct_probs, cot_probs in zip(batch, rationales, direct_values, cot_values):
            append_jsonl(
                args.output,
                {
                    "id": str(instance["id"]),
                    "question": instance["question_stem"],
                    "options": answer_choices(instance),
                    "correct_letter": str(instance["answerKey"]),
                    "student_draft": rationale,
                    "student_direct_probs": direct_probs,
                    "student_direct_answer": predicted_letter(instance, direct_probs),
                    "student_cot_probs": cot_probs,
                    "student_cot_answer": predicted_letter(instance, cot_probs),
                    "raw_instance": instance,
                },
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Wrote drafts to {args.output}.")


if __name__ == "__main__":
    main()
