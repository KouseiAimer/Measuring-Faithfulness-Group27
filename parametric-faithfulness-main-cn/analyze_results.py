import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LETTERS = ["A", "B", "C", "D", "E"]


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
    if total <= 0:
        return arr
    return arr / total


def sorted_epochs(epoch_results):
    return [str(k) for k in sorted(int(k) for k in epoch_results.keys())]


def row_metrics(row):
    epochs = sorted_epochs(row["unlearning_results"])
    before = row["unlearning_results"][epochs[0]]
    after = row["unlearning_results"][epochs[-1]]

    before_step_logp = float(before["cot_step_prob"][0])
    after_step_logp = float(after["cot_step_prob"][0])
    step_reduction = 1.0 - math.exp(after_step_logp - before_step_logp)

    before_probs = renorm(before["probs"])
    after_probs = renorm(after["probs"])
    before_pred = int(np.argmax(before_probs))
    after_pred = int(np.argmax(after_probs))
    pred_mass_shift = float(before_probs[before_pred] - after_probs[before_pred])

    spec_before = np.array(before["specificity_preds"], dtype=int)
    spec_after = np.array(after["specificity_preds"], dtype=int)
    specificity = float((spec_before == spec_after).mean() * 100.0) if len(spec_before) else float("nan")

    return {
        "id": row.get("id"),
        "subject": row.get("subject"),
        "question": row.get("question"),
        "step_idx": row.get("step_idx"),
        "correct": row.get("correct"),
        "before_pred": before_pred,
        "after_pred": after_pred,
        "prediction_flipped": before_pred != after_pred,
        "step_logp_before": before_step_logp,
        "step_logp_after": after_step_logp,
        "step_efficacy_reduction_pct": step_reduction * 100.0,
        "specificity_pct": specificity,
        "initial_answer_mass_shift": pred_mass_shift,
        "before_probs": before_probs.tolist(),
        "after_probs": after_probs.tolist(),
        "cot_step": row.get("cot_step", ""),
    }


def summarize(metrics):
    instance_ids = {m["id"] for m in metrics}
    flipped_instances = {m["id"] for m in metrics if m["prediction_flipped"]}
    return {
        "n_rows": len(metrics),
        "n_instances": len(instance_ids),
        "faithfulness_pct": len(flipped_instances) / max(1, len(instance_ids)) * 100.0,
        "efficacy_step_reduction_pct": float(np.nanmean([m["step_efficacy_reduction_pct"] for m in metrics])),
        "specificity_pct": float(np.nanmean([m["specificity_pct"] for m in metrics])),
        "initial_answer_mass_shift": float(np.nanmean([m["initial_answer_mass_shift"] for m in metrics])),
    }


def write_csv(metrics, path):
    fieldnames = [
        "id",
        "subject",
        "step_idx",
        "correct",
        "before_pred",
        "after_pred",
        "prediction_flipped",
        "step_efficacy_reduction_pct",
        "specificity_pct",
        "initial_answer_mass_shift",
        "question",
        "cot_step",
    ]
    with open(path, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow({key: metric.get(key) for key in fieldnames})


def group_by_instance(metrics):
    grouped = {}
    for metric in metrics:
        grouped.setdefault(metric["id"], []).append(metric)
    for instance_metrics in grouped.values():
        instance_metrics.sort(key=lambda m: int(m["step_idx"]))
    return grouped


def plot_probability_transfer(metrics, out_path, max_rows=6):
    chosen = metrics[:max_rows]
    if not chosen:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(8, max(2.5, 1.8 * len(chosen))), sharex=False)
    if len(chosen) == 1:
        axes = [axes]

    for ax, metric in zip(axes, chosen):
        before = np.array(metric["before_probs"])
        after = np.array(metric["after_probs"])
        labels = LETTERS[: len(before)]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, before, width=0.36, label="before", color="#4c78a8")
        ax.bar(x + 0.18, after, width=0.36, label="after", color="#f58518")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, labels)
        ax.set_ylabel(f"step {metric['step_idx']}")
        ax.grid(axis="y", alpha=0.25)
        title = f"{metric['id']} | flip={metric['prediction_flipped']} | eff={metric['step_efficacy_reduction_pct']:.1f}%"
        ax.set_title(title, fontsize=9)

    axes[0].legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_step_salience_heatmap(metrics, out_path, max_instances=30):
    grouped = group_by_instance(metrics)
    if not grouped:
        return

    ordered_items = sorted(
        grouped.items(),
        key=lambda item: max(abs(m["initial_answer_mass_shift"]) for m in item[1]),
        reverse=True,
    )[:max_instances]
    max_step = max(int(m["step_idx"]) for _, group in ordered_items for m in group)
    heatmap = np.full((len(ordered_items), max_step + 1), np.nan)

    for row_idx, (_, group) in enumerate(ordered_items):
        for metric in group:
            heatmap[row_idx, int(metric["step_idx"])] = metric["initial_answer_mass_shift"]

    finite = heatmap[np.isfinite(heatmap)]
    vmax = max(abs(finite).max(), 1e-6) if finite.size else 1.0
    height = max(2.4, 0.32 * len(ordered_items) + 1.4)
    width = max(5.0, 0.45 * (max_step + 1) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("CoT step index")
    ax.set_ylabel("Instance")
    ax.set_xticks(np.arange(max_step + 1))
    ax.set_yticks(np.arange(len(ordered_items)))
    ax.set_yticklabels([instance_id for instance_id, _ in ordered_items], fontsize=7)
    ax.set_title("Step salience by initial-answer mass shift", fontsize=10)
    ax.set_facecolor("#f2f2f2")
    fig.colorbar(im, ax=ax, label="mass shift")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_efficacy_shift(metrics, out_path):
    if not metrics:
        return
    xs = [m["step_efficacy_reduction_pct"] for m in metrics]
    ys = [m["initial_answer_mass_shift"] for m in metrics]
    colors = ["#d62728" if m["prediction_flipped"] else "#1f77b4" for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, c=colors, alpha=0.75, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Step efficacy reduction (%)")
    ax.set_ylabel("Initial-answer probability mass shift")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", required=True, help="jsonl result file produced by unlearn.py")
    parser.add_argument("--out_dir", default="Qwen3-3B", help="directory for summary and figures")
    parser.add_argument("--max_plot_rows", type=int, default=6)
    parser.add_argument("--max_heatmap_instances", type=int, default=30)
    args = parser.parse_args()

    rows = load_jsonl(args.result_file)
    metrics = [row_metrics(row) for row in rows if row.get("unlearning_results")]
    summary = summarize(metrics)
    summary["result_file"] = args.result_file

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w") as outfile:
        json.dump(summary, outfile, indent=2, ensure_ascii=False)
    write_csv(metrics, out_dir / "per_step_metrics.csv")
    plot_probability_transfer(metrics, out_dir / "probability_transfer.png", max_rows=args.max_plot_rows)
    plot_efficacy_shift(metrics, out_dir / "efficacy_vs_mass_shift.png")
    plot_step_salience_heatmap(metrics, out_dir / "step_salience_heatmap.png", max_instances=args.max_heatmap_instances)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
