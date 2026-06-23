"""Legacy two-file summary; use analysis_pipeline.py for paper-style reports."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def renorm(values):
    values = np.array(values, dtype=float)
    total = values.sum()
    return values / total if total else values


def metrics(rows):
    valid = [row for row in rows if row.get("unlearning_results")]
    if not valid:
        return {"n_steps": 0, "n_questions": 0}
    flip_questions = set()
    efficacies = []
    specificities = []
    soft = []
    for row in valid:
        epochs = row["unlearning_results"]
        before = epochs[str(min(map(int, epochs.keys())))]
        after = epochs[str(max(map(int, epochs.keys())))]
        before_step = np.exp(before["cot_step_prob"][0])
        after_step = np.exp(after["cot_step_prob"][0])
        if before_step:
            efficacies.append((1.0 - after_step / before_step) * 100.0)
        initial_pred = int(np.argmax(before["probs"]))
        if int(np.argmax(after["probs"])) != initial_pred:
            flip_questions.add(row["id"])
        before_probs = renorm(before["probs"])
        after_probs = renorm(after["probs"])
        soft.append(float(before_probs[initial_pred] - after_probs[initial_pred]))
        initial_spec = np.array(before["specificity_preds"])
        final_spec = np.array(after["specificity_preds"])
        specificities.append(float(np.mean(initial_spec == final_spec) * 100.0))
    question_ids = {row["id"] for row in valid}
    return {
        "n_steps": len(valid),
        "n_questions": len(question_ids),
        "efficacy": float(np.mean(efficacies)),
        "specificity": float(np.mean(specificities)),
        "ff_hard": float(len(flip_questions) / len(question_ids) * 100.0),
        "ff_soft": float(np.mean(soft)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", required=True)
    parser.add_argument("--tot", required=True)
    parser.add_argument("--tot_union", default="")
    parser.add_argument("--output", default="final_analysis/openbook/LLaMA-3-3B/summary.json")
    args = parser.parse_args()
    report = {
        "CoT": metrics(load_jsonl(args.cot)),
        "ToT-selected": metrics(load_jsonl(args.tot)),
    }
    if args.tot_union:
        report["ToT-union"] = metrics(load_jsonl(args.tot_union))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
