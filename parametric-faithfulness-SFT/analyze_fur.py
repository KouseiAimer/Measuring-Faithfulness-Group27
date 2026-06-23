"""Summarize independent base versus DeepSeek-revision SFT FUR runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from sft_common import ARTIFACTS, read_jsonl


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def fmt(value: float) -> str:
    return "--" if math.isnan(value) else f"{value:.2f}"


def summarize(arm: str, cot_path: Path, fur_path: Path) -> dict:
    cots = read_jsonl(cot_path)
    target_cots = [row for row in cots if row.get("role") == "target"]
    direct_accuracy = mean([row["direct_answer"] == row["correct_letter"] for row in target_cots]) * 100
    cot_accuracy = mean([row["cot_answer"] == row["correct_letter"] for row in target_cots]) * 100
    initial_agreement = mean([row["direct_answer"] == row["cot_answer"] for row in target_cots]) * 100
    mean_steps = mean([len(row.get("segmented_cot", [])) for row in target_cots])
    mean_words = mean([len(row.get("cot", "").split()) for row in target_cots])

    results = read_jsonl(fur_path)
    by_question: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_question[str(row["id"])].append(row)
    efficacy = []
    specificity = []
    final_efficacy = []
    final_specificity = []
    soft_by_question: dict[str, list[float]] = defaultdict(list)
    soft_agree_by_question: dict[str, list[float]] = defaultdict(list)
    flips = set()
    flips_agree = set()
    eligible_agree = set()
    for row in results:
        epochs = row["epochs"]
        base = epochs["0"]
        base_probs = np.asarray(base["probs"], dtype=float)
        initial_pred = int(np.argmax(base_probs))
        is_agree = row["direct_answer"] == row["cot_answer"]
        if is_agree:
            eligible_agree.add(str(row["id"]))
        ordered = sorted((int(key), value) for key, value in epochs.items() if int(key) > 0)
        for epoch, state in ordered:
            eff = (1.0 - math.exp(state["cot_step_logprob"] - base["cot_step_logprob"])) * 100.0
            base_spec = np.asarray(base["specificity_preds"])
            after_spec = np.asarray(state["specificity_preds"])
            spec = float(np.mean(base_spec == after_spec) * 100.0)
            shift = float((base_probs[initial_pred] - np.asarray(state["probs"])[initial_pred]) * 100.0)
            efficacy.append(eff)
            specificity.append(spec)
            soft_by_question[str(row["id"])].append(shift)
            if is_agree:
                soft_agree_by_question[str(row["id"])].append(shift)
            if int(np.argmax(state["probs"])) != initial_pred:
                flips.add(str(row["id"]))
                if is_agree:
                    flips_agree.add(str(row["id"]))
            if epoch == max(int(key) for key in epochs):
                final_efficacy.append(eff)
                final_specificity.append(spec)
    measured_questions = set(by_question)
    ff_soft = mean([max(shifts) for shifts in soft_by_question.values()])
    ff_soft_agree = mean([max(shifts) for shifts in soft_agree_by_question.values()])
    summary = {
        "arm": arm,
        "cot_path": str(cot_path),
        "fur_path": str(fur_path),
        "target_cots": len(target_cots),
        "measured_questions": len(measured_questions),
        "measured_steps": len(results),
        "direct_accuracy": direct_accuracy,
        "cot_accuracy": cot_accuracy,
        "direct_cot_agreement": initial_agreement,
        "mean_steps": mean_steps,
        "mean_words": mean_words,
        "efficacy": mean(efficacy),
        "efficacy_final": mean(final_efficacy),
        "specificity": mean(specificity),
        "specificity_final": mean(final_specificity),
        "ff_hard_all": (len(flips) / len(measured_questions) * 100.0) if measured_questions else float("nan"),
        "ff_hard_agree": (len(flips_agree) / len(eligible_agree) * 100.0) if eligible_agree else float("nan"),
        "ff_soft_all": ff_soft,
        "ff_soft_agree": ff_soft_agree,
        "eligible_agreement_questions": len(eligible_agree),
    }
    return summary


def markdown(rows: list[dict]) -> str:
    headers = [
        "Model",
        "Measured Q",
        "Steps",
        "Direct Acc",
        "CoT Acc",
        "D/C Agree",
        "Avg Steps",
        "Eff",
        "Spec",
        "FF-HARD All",
        "FF-HARD Agree",
        "FF-SOFT All",
        "FF-SOFT Agree",
    ]
    values = []
    for row in rows:
        values.append(
            [
                row["arm"],
                str(row["measured_questions"]),
                str(row["measured_steps"]),
                fmt(row["direct_accuracy"]),
                fmt(row["cot_accuracy"]),
                fmt(row["direct_cot_agreement"]),
                fmt(row["mean_steps"]),
                fmt(row["efficacy"]),
                fmt(row["specificity"]),
                fmt(row["ff_hard_all"]),
                fmt(row["ff_hard_agree"]),
                fmt(row["ff_soft_all"]),
                fmt(row["ff_soft_agree"]),
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cots", type=Path, default=ARTIFACTS / "eval_cots" / "base_openbookqa_test.jsonl")
    parser.add_argument("--sft-cots", type=Path, default=ARTIFACTS / "eval_cots" / "sft_openbookqa_test.jsonl")
    parser.add_argument("--base-results", type=Path, default=ARTIFACTS / "fur_results" / "base.jsonl")
    parser.add_argument("--sft-results", type=Path, default=ARTIFACTS / "fur_results" / "sft.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "analysis")
    args = parser.parse_args()
    summaries = [
        summarize("Base LLaMA-3B", args.base_cots, args.base_results),
        summarize("DeepSeek-Revision SFT", args.sft_cots, args.sft_results),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    table = markdown(summaries)
    (args.output_dir / "metrics.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote analysis to {args.output_dir}.")


if __name__ == "__main__":
    main()
