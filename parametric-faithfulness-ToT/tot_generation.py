"""Generate and compare MCQA reasoning trees for the FUR extension."""

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataload import DATASETS
from evaluate import answer_probabilities, complete, generation_fixed_cot, safe_sent_tokenize
from util import set_random_seed


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def score_path(model, tokenizer, handler, instance, text):
    probs, prediction = generation_fixed_cot(model, tokenizer, handler, instance, text)
    return {
        "text": text.strip(),
        "answer_probs": probs.tolist(),
        "prediction": int(prediction),
        "score": float(np.max(probs)),
        "steps": safe_sent_tokenize(text.strip()),
    }


def deduplicate(paths):
    by_text = {}
    for path in paths:
        text = path["text"].strip()
        if text and (text not in by_text or path["score"] > by_text[text]["score"]):
            by_text[text] = path
    return list(by_text.values())


def sample_select(model, tokenizer, handler, instance, args):
    prompt = handler.make_cot_prompt(instance)
    paths = []
    for _ in range(args.num_paths):
        text = complete(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p
        )
        if text.strip():
            paths.append(score_path(model, tokenizer, handler, instance, text))
    paths = deduplicate(paths) or [score_path(model, tokenizer, handler, instance, "I need to choose the most plausible answer.")]
    return max(paths, key=lambda p: p["score"]), paths


def next_thought(text):
    steps = safe_sent_tokenize(text.strip())
    return steps[0].strip() if steps else text.strip()


def beam_prune(model, tokenizer, handler, instance, args):
    base_prompt = handler.make_cot_prompt(instance)
    beams = [{"text": "", "score": 0.0, "steps": []}]
    all_expansions = []
    for _depth in range(args.depth):
        expansions = []
        for beam in beams:
            prefix = base_prompt + beam["text"]
            if beam["text"]:
                prefix += "\n"
            for _ in range(args.branch_factor):
                generated = complete(
                    model, tokenizer, prefix, max_new_tokens=args.thought_max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p, split_newline=False
                )
                thought = next_thought(generated)
                if not thought:
                    continue
                text = "\n".join([part for part in (beam["text"], thought) if part])
                expansions.append(score_path(model, tokenizer, handler, instance, text))
        expansions = deduplicate(expansions)
        if not expansions:
            break
        all_expansions.extend(expansions)
        beams = sorted(expansions, key=lambda p: p["score"], reverse=True)[:args.beam_width]
    final_paths = deduplicate(beams or all_expansions)
    if not final_paths:
        return sample_select(model, tokenizer, handler, instance, args)
    return max(final_paths, key=lambda p: p["score"]), final_paths


def build_row(handler, instance, nocot_probs, selected, paths, mode):
    return {
        "id": instance[handler.id_key],
        "question": instance[handler.q_key],
        "correct_letter": handler.correct_answer_letter(instance),
        "cot_prompt": handler.make_cot_prompt(instance),
        "cot": selected["text"],
        "options": handler.get_answer_choices(instance),
        "nocot_probs": nocot_probs.tolist(),
        "cot_probs": selected["answer_probs"],
        "segmented_cot": selected["steps"] or [selected["text"]],
        "raw_instance": dict(instance),
        "tot_mode": mode,
        "paths": paths,
        "tree_metadata": {
            "num_distinct_paths": len(paths),
            "winning_score": selected["score"],
        },
    }


def generate_mode(model, tokenizer, handler, dataset, mode, args):
    rows = []
    builder = sample_select if mode == "sample_select" else beam_prune
    for idx, instance in tqdm(enumerate(dataset), desc=mode):
        if args.max_instances and idx >= args.max_instances:
            break
        _, nocot_probs, _ = answer_probabilities(model, tokenizer, handler, instance)
        selected, paths = builder(model, tokenizer, handler, instance, args)
        rows.append(build_row(handler, instance, nocot_probs, selected, paths, mode))
    return rows


def prediction_index(row):
    return int(np.argmax(row["cot_probs"]))


def summary(rows, handler):
    correct = []
    confidence = []
    diversity = []
    entropy = []
    for row in rows:
        letters = handler.get_answer_letters(row["raw_instance"])
        gold = letters.index(row["correct_letter"])
        correct.append(prediction_index(row) == gold)
        confidence.append(max(row["cot_probs"]))
        diversity.append(row["tree_metadata"]["num_distinct_paths"])
        votes = np.bincount([path["prediction"] for path in row["paths"]], minlength=len(letters)).astype(float)
        votes /= votes.sum()
        entropy.append(-sum(p * math.log(p) for p in votes if p > 0))
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if rows else 0.0,
        "mean_winning_confidence": float(np.mean(confidence)) if rows else 0.0,
        "mean_distinct_paths": float(np.mean(diversity)) if rows else 0.0,
        "mean_vote_entropy": float(np.mean(entropy)) if rows else 0.0,
    }


def store_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_complete_cache(path, expected_size):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as infile:
        rows = [json.loads(line) for line in infile if line.strip()]
    if len(rows) != expected_size:
        return None
    print(f"Reusing complete ToT cache: {path}")
    return rows


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", default="openbook")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--mode", choices=["sample_select", "beam_prune", "compare"], default="sample_select")
    parser.add_argument("--output_dir", default="final_tree_ToT")
    parser.add_argument("--max_instances", type=int, default=70)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--num_paths", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--thought_max_new_tokens", type=int, default=72)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branch_factor", type=int, default=2)
    parser.add_argument("--beam_width", type=int, default=3)
    return parser


def main():
    args = make_parser().parse_args()
    set_random_seed(args.seed)
    random.seed(args.seed)
    handler = DATASETS[args.dataset]
    _, validation, test = handler.get_dataset_splits()
    selected_split = validation if args.split == "validation" else test
    model, tokenizer = load_model(args.model_name)
    out_root = Path(args.output_dir) / args.dataset / args.model_name.split("/")[-1]
    modes = ["sample_select", "beam_prune"] if args.mode == "compare" else [args.mode]
    summaries = {}
    outputs = {}
    for mode in modes:
        suffix = f"{args.split}_n={args.max_instances}_s={args.seed}"
        output = out_root / f"{mode}_{suffix}.jsonl"
        rows = load_complete_cache(output, args.max_instances)
        if rows is None:
            set_random_seed(args.seed)
            rows = generate_mode(model, tokenizer, handler, selected_split, mode, args)
            store_jsonl(rows, output)
        outputs[mode] = str(output)
        summaries[mode] = summary(rows, handler)
    if args.mode == "compare":
        chosen = max(
            modes,
            key=lambda mode: (
                summaries[mode]["accuracy"],
                summaries[mode]["mean_winning_confidence"],
                summaries[mode]["mean_distinct_paths"],
            ),
        )
        report = {
            "selection_split": args.split,
            "selection_rule": "accuracy, then winning confidence, then distinct paths",
            "chosen_mode": chosen,
            "summaries": summaries,
            "outputs": outputs,
        }
        report_path = out_root / f"mode_selection_{args.split}_n={args.max_instances}_s={args.seed}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"mode": args.mode, "summary": summaries[args.mode], "output": outputs[args.mode]}, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
