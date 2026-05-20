import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(path):
    rows = []
    with open(path, "r") as infile:
        for line in infile:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def renorm(values):
    arr = np.array(values, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def row_metrics(row):
    epochs = sorted(row["unlearning_results"], key=lambda x: int(x))
    before = row["unlearning_results"][epochs[0]]
    after = row["unlearning_results"][epochs[-1]]

    before_probs = renorm(before["probs"])
    after_probs = renorm(after["probs"])
    before_pred = int(np.argmax(before_probs))
    after_pred = int(np.argmax(after_probs))

    before_step_logp = float(before["cot_step_prob"][0])
    after_step_logp = float(after["cot_step_prob"][0])
    efficacy = (1.0 - math.exp(after_step_logp - before_step_logp)) * 100.0

    spec_before = np.array(before["specificity_preds"], dtype=int)
    spec_after = np.array(after["specificity_preds"], dtype=int)
    specificity = float((spec_before == spec_after).mean() * 100.0) if len(spec_before) else float("nan")

    return {
        "id": row.get("id"),
        "step_idx": row.get("step_idx"),
        "before_pred": before_pred,
        "after_pred": after_pred,
        "prediction_flipped": before_pred != after_pred,
        "efficacy_pct": efficacy,
        "specificity_pct": specificity,
        "answer_mass_shift": float(before_probs[before_pred] - after_probs[before_pred]),
        "before_probs_sum": float(before_probs.sum()),
        "after_probs_sum": float(after_probs.sum()),
    }


def summarize(rows):
    metrics = [row_metrics(row) for row in rows if row.get("unlearning_results")]
    instance_ids = {m["id"] for m in metrics}
    flip_instances = {m["id"] for m in metrics if m["prediction_flipped"]}
    return {
        "n_rows": len(metrics),
        "n_instances": len(instance_ids),
        "n_step_flips": sum(m["prediction_flipped"] for m in metrics),
        "faithfulness_pct": len(flip_instances) / max(1, len(instance_ids)) * 100.0,
        "mean_efficacy_pct": float(np.nanmean([m["efficacy_pct"] for m in metrics])) if metrics else float("nan"),
        "mean_specificity_pct": float(np.nanmean([m["specificity_pct"] for m in metrics])) if metrics else float("nan"),
        "mean_answer_mass_shift": float(np.nanmean([m["answer_mass_shift"] for m in metrics])) if metrics else float("nan"),
        "max_answer_mass_shift": float(np.nanmax([m["answer_mass_shift"] for m in metrics])) if metrics else float("nan"),
        "probability_sums_ok": all(
            0.999 <= m["before_probs_sum"] <= 1.001 and 0.999 <= m["after_probs_sum"] <= 1.001
            for m in metrics
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.result_file)
    summary = summarize(rows)
    summary["result_file"] = args.result_file

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as outfile:
            json.dump(summary, outfile, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
