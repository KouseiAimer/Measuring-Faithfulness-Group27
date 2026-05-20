import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "final_results" / "openbook" / "LLaMA-3-3B"
OUT_DIR = ROOT / "result_enhanced"

RUNS = {
    "full": {
        "label": "Full FUR",
        "pattern": "npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_shard={shard}-of-3.out",
        "selected": False,
    },
    "last_top1": {
        "label": "Last-step top1",
        "pattern": "npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=last_k=1_shard={shard}-of-3.out",
        "selected": True,
    },
    "random_top1": {
        "label": "Random top1",
        "pattern": "npo_KL_sentencize_s=True_lr=3e-05_rs=1001_n=100_pos=False_ff2=True_sel=random_k=1_shard={shard}-of-3.out",
        "selected": True,
    },
}

OPTION_LETTERS = list("ABCDE")


def load_jsonl(path):
    rows = []
    with open(path, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_run(run_name, cfg):
    rows = []
    merged_path = OUT_DIR / f"{run_name}_merged.jsonl"
    with open(merged_path, "w") as outfile:
        for shard in range(3):
            path = RESULT_DIR / cfg["pattern"].format(shard=shard)
            shard_rows = load_jsonl(path)
            rows.extend(shard_rows)
            for row in shard_rows:
                outfile.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, merged_path


def renorm(values):
    arr = np.array(values, dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def epoch_pair(row):
    keys = sorted(row["unlearning_results"], key=lambda k: int(k))
    return row["unlearning_results"][keys[0]], row["unlearning_results"][keys[-1]]


def get_probs(row):
    before, after = epoch_pair(row)
    before_probs = renorm(before["probs"])
    after_probs = renorm(after["probs"])
    return before, after, before_probs, after_probs


def safe_exp_delta(after_logp, before_logp):
    if before_logp is None or after_logp is None:
        return float("nan")
    delta = float(after_logp) - float(before_logp)
    try:
        return math.exp(delta)
    except OverflowError:
        return float("inf")


def row_to_metric(row, run_name, run_label):
    before, after, before_probs, after_probs = get_probs(row)
    before_pred = int(np.argmax(before_probs))
    after_pred = int(np.argmax(after_probs))
    correct_letter = row.get("correct")
    correct_idx = OPTION_LETTERS.index(correct_letter) if correct_letter in OPTION_LETTERS else None

    before_step_logp = float(before["cot_step_prob"][0])
    after_step_logp = float(after["cot_step_prob"][0])
    step_prob_ratio = safe_exp_delta(after_step_logp, before_step_logp)
    efficacy_pct = (1.0 - step_prob_ratio) * 100.0

    before_cot_logp = float(before["cot_prob"][0])
    after_cot_logp = float(after["cot_prob"][0])
    cot_prob_ratio = safe_exp_delta(after_cot_logp, before_cot_logp)
    cot_efficacy_pct = (1.0 - cot_prob_ratio) * 100.0

    spec_before = np.array(before.get("specificity_preds", []), dtype=int)
    spec_after = np.array(after.get("specificity_preds", []), dtype=int)
    specificity_pct = (
        float((spec_before == spec_after).mean() * 100.0)
        if len(spec_before) == len(spec_after) and len(spec_before) > 0
        else float("nan")
    )

    prob_cols = {}
    for idx, letter in enumerate(OPTION_LETTERS[: len(before_probs)]):
        prob_cols[f"before_{letter}"] = float(before_probs[idx])
        prob_cols[f"after_{letter}"] = float(after_probs[idx])
        prob_cols[f"delta_{letter}"] = float(after_probs[idx] - before_probs[idx])

    original_answer_mass_shift = float(before_probs[before_pred] - after_probs[before_pred])
    correct_mass_shift = (
        float(before_probs[correct_idx] - after_probs[correct_idx])
        if correct_idx is not None and correct_idx < len(before_probs)
        else float("nan")
    )
    total_variation = float(np.abs(after_probs - before_probs).sum() / 2.0)
    entropy_before = float(-(before_probs * np.log(np.clip(before_probs, 1e-12, 1.0))).sum())
    entropy_after = float(-(after_probs * np.log(np.clip(after_probs, 1e-12, 1.0))).sum())

    return {
        "run": run_name,
        "run_label": run_label,
        "id": row.get("id"),
        "question": row.get("question"),
        "step_idx": int(row.get("step_idx", 0)),
        "n_steps_in_cot": len(row.get("segmented_cot") or []),
        "cot_step": row.get("cot_step", ""),
        "options": " | ".join(row.get("options", [])),
        "correct": row.get("correct"),
        "before_pred_idx": before_pred,
        "after_pred_idx": after_pred,
        "before_pred": OPTION_LETTERS[before_pred],
        "after_pred": OPTION_LETTERS[after_pred],
        "before_correct": before_pred == correct_idx,
        "after_correct": after_pred == correct_idx,
        "prediction_flipped": before_pred != after_pred,
        "efficacy_pct": float(efficacy_pct),
        "cot_efficacy_pct": float(cot_efficacy_pct),
        "specificity_pct": specificity_pct,
        "original_answer_mass_shift": original_answer_mass_shift,
        "correct_mass_shift": correct_mass_shift,
        "total_variation": total_variation,
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "entropy_delta": entropy_after - entropy_before,
        "before_original_prob": float(before_probs[before_pred]),
        "after_original_prob": float(after_probs[before_pred]),
        "before_correct_prob": float(before_probs[correct_idx]) if correct_idx is not None else float("nan"),
        "after_correct_prob": float(after_probs[correct_idx]) if correct_idx is not None else float("nan"),
        "before_probs_sum": float(before_probs.sum()),
        "after_probs_sum": float(after_probs.sum()),
        **prob_cols,
    }


def summarize_run(metrics, run_name, run_label, full_row_count=None):
    df = metrics[metrics["run"] == run_name].copy()
    inst = df.groupby("id", dropna=False).agg(
        n_steps_evaluated=("step_idx", "count"),
        any_flip=("prediction_flipped", "max"),
        max_mass_shift=("original_answer_mass_shift", "max"),
        max_total_variation=("total_variation", "max"),
        mean_efficacy=("efficacy_pct", "mean"),
        max_efficacy=("efficacy_pct", "max"),
        mean_specificity=("specificity_pct", "mean"),
        before_correct=("before_correct", "max"),
        after_correct_any=("after_correct", "max"),
    )
    row_count = len(df)
    instance_count = len(inst)
    cost_reduction = (
        1.0 - row_count / full_row_count if full_row_count and run_name != "full" else 0.0
    )
    return {
        "run": run_name,
        "run_label": run_label,
        "rows_unlearned": int(row_count),
        "instances_covered": int(instance_count),
        "mean_steps_per_instance": float(inst["n_steps_evaluated"].mean()) if instance_count else float("nan"),
        "step_flip_count": int(df["prediction_flipped"].sum()),
        "step_flip_rate_pct": float(df["prediction_flipped"].mean() * 100.0) if row_count else 0.0,
        "faithful_instance_count": int(inst["any_flip"].sum()) if instance_count else 0,
        "faithful_instance_rate_pct": float(inst["any_flip"].mean() * 100.0) if instance_count else 0.0,
        "mean_efficacy_pct": float(df["efficacy_pct"].mean()) if row_count else float("nan"),
        "median_efficacy_pct": float(df["efficacy_pct"].median()) if row_count else float("nan"),
        "mean_specificity_pct": float(df["specificity_pct"].mean()) if row_count else float("nan"),
        "mean_original_answer_mass_shift": float(df["original_answer_mass_shift"].mean()) if row_count else float("nan"),
        "median_original_answer_mass_shift": float(df["original_answer_mass_shift"].median()) if row_count else float("nan"),
        "max_original_answer_mass_shift": float(df["original_answer_mass_shift"].max()) if row_count else float("nan"),
        "mean_total_variation": float(df["total_variation"].mean()) if row_count else float("nan"),
        "cost_reduction_vs_full_pct": float(cost_reduction * 100.0),
        "probability_sums_ok": bool(
            ((df["before_probs_sum"].between(0.999, 1.001)) & (df["after_probs_sum"].between(0.999, 1.001))).all()
        )
        if row_count
        else True,
    }


def summarize_recovery(row_metrics, instance_metrics, full_row_count):
    full_inst = instance_metrics[instance_metrics["run"] == "full"].set_index("id")
    full_flip_ids = set(full_inst.index[full_inst["any_flip"]])
    full_positive_steps = set(
        zip(
            row_metrics[(row_metrics["run"] == "full") & (row_metrics["prediction_flipped"])]["id"],
            row_metrics[(row_metrics["run"] == "full") & (row_metrics["prediction_flipped"])]["step_idx"],
        )
    )
    rows = []
    for run_name in ["last_top1", "random_top1"]:
        df = row_metrics[row_metrics["run"] == run_name].copy()
        inst = instance_metrics[instance_metrics["run"] == run_name].set_index("id")
        selected_ids = set(inst.index)
        found_ids = set(inst.index[inst["any_flip"]])
        overlap_full_flip = full_flip_ids & selected_ids
        selected_steps = set(zip(df["id"], df["step_idx"]))
        hit_steps = selected_steps & full_positive_steps
        rows.append(
            {
                "run": run_name,
                "run_label": RUNS[run_name]["label"],
                "selected_rows": int(len(df)),
                "selected_instances": int(len(selected_ids)),
                "full_rows": int(full_row_count),
                "full_faithful_instances": int(len(full_flip_ids)),
                "found_faithful_instances": int(len(found_ids & full_flip_ids)),
                "recovery_at_1_full_denominator_pct": float(len(found_ids & full_flip_ids) / max(1, len(full_flip_ids)) * 100.0),
                "recovery_at_1_overlap_denominator_pct": float(len(found_ids & overlap_full_flip) / max(1, len(overlap_full_flip)) * 100.0),
                "coverage_of_full_instances_pct": float(len(selected_ids) / max(1, len(full_inst)) * 100.0),
                "cost_reduction_pct": float((1.0 - len(df) / full_row_count) * 100.0),
                "step_hit_at_1_pct": float(len(hit_steps) / max(1, len(full_positive_steps)) * 100.0),
                "selected_step_precision_pct": float(len(hit_steps) / max(1, len(selected_steps)) * 100.0),
                "faithfulness_per_100_unlearns": float(len(found_ids & full_flip_ids) / max(1, len(df)) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def build_instance_metrics(row_metrics):
    rows = []
    for (run, instance_id), group in row_metrics.groupby(["run", "id"], dropna=False):
        idx_max = group["original_answer_mass_shift"].idxmax()
        best = group.loc[idx_max]
        flip_steps = group[group["prediction_flipped"]]["step_idx"].tolist()
        rows.append(
            {
                "run": run,
                "run_label": best["run_label"],
                "id": instance_id,
                "question": best["question"],
                "n_steps_evaluated": int(len(group)),
                "n_steps_in_cot": int(group["n_steps_in_cot"].max()),
                "any_flip": bool(group["prediction_flipped"].any()),
                "flip_steps": ",".join(map(str, flip_steps)),
                "n_flip_steps": int(group["prediction_flipped"].sum()),
                "best_step_idx_by_mass_shift": int(best["step_idx"]),
                "best_step_mass_shift": float(best["original_answer_mass_shift"]),
                "best_step_total_variation": float(best["total_variation"]),
                "best_step_efficacy_pct": float(best["efficacy_pct"]),
                "mean_efficacy_pct": float(group["efficacy_pct"].mean()),
                "mean_specificity_pct": float(group["specificity_pct"].mean()),
                "before_pred": best["before_pred"],
                "after_pred_at_best_shift": best["after_pred"],
                "correct": best["correct"],
                "before_correct": bool(group["before_correct"].iloc[0]),
                "after_correct_any": bool(group["after_correct"].any()),
                "options": best["options"],
            }
        )
    return pd.DataFrame(rows)


def save_markdown_table(df, path, max_rows=None):
    show = df if max_rows is None else df.head(max_rows)
    with open(path, "w") as outfile:
        outfile.write(show.to_markdown(index=False))
        outfile.write("\n")


def plot_overall(summary_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(summary_df))
    labels = summary_df["run_label"].tolist()

    axes[0, 0].bar(x, summary_df["rows_unlearned"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0, 0].set_title("Unlearning Cost")
    axes[0, 0].set_ylabel("Rows / Step-runs")
    axes[0, 0].set_xticks(x, labels, rotation=15, ha="right")

    axes[0, 1].bar(x, summary_df["faithful_instance_rate_pct"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0, 1].set_title("Faithful Instance Rate")
    axes[0, 1].set_ylabel("% instances with answer flip")
    axes[0, 1].set_xticks(x, labels, rotation=15, ha="right")

    axes[1, 0].bar(x, summary_df["mean_original_answer_mass_shift"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1, 0].set_title("Mean Original-answer Probability Drop")
    axes[1, 0].set_ylabel("Probability mass")
    axes[1, 0].set_xticks(x, labels, rotation=15, ha="right")

    axes[1, 1].bar(x, summary_df["mean_specificity_pct"], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1, 1].set_title("Mean Specificity")
    axes[1, 1].set_ylabel("% held-out predictions unchanged")
    axes[1, 1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 1].set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "overall_metrics_comparison.png", dpi=180)
    plt.close(fig)


def plot_recovery(recovery_df):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(recovery_df))
    width = 0.32
    ax1.bar(x - width / 2, recovery_df["recovery_at_1_full_denominator_pct"], width, label="Recovery@1", color="#f58518")
    ax1.bar(x + width / 2, recovery_df["cost_reduction_pct"], width, label="Cost reduction", color="#4c78a8")
    ax1.set_xticks(x, recovery_df["run_label"], rotation=10, ha="right")
    ax1.set_ylabel("%")
    ax1.set_ylim(0, 105)
    ax1.set_title("Selective FUR: Recovery vs Cost Reduction")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recovery_cost_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_efficacy_scatter(row_metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"full": "#4c78a8", "last_top1": "#f58518", "random_top1": "#54a24b"}
    for run, group in row_metrics.groupby("run"):
        ax.scatter(
            group["efficacy_pct"],
            group["original_answer_mass_shift"],
            s=18,
            alpha=0.55,
            label=RUNS[run]["label"],
            color=colors[run],
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Efficacy: target step probability reduction (%)")
    ax.set_ylabel("Original-answer probability drop")
    ax.set_title("Efficacy vs Answer Probability Change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "efficacy_vs_answer_mass_shift.png", dpi=180)
    plt.close(fig)


def plot_step_heatmap(row_metrics):
    full = row_metrics[row_metrics["run"] == "full"].copy()
    full_inst = (
        full.groupby("id")["original_answer_mass_shift"]
        .max()
        .sort_values(ascending=False)
        .head(18)
        .index.tolist()
    )
    if not full_inst:
        return
    pivot = full[full["id"].isin(full_inst)].pivot_table(
        index="id",
        columns="step_idx",
        values="original_answer_mass_shift",
        aggfunc="max",
    )
    pivot = pivot.loc[full_inst]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.fillna(np.nan).values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"step {c}" for c in pivot.columns])
    ax.set_title("Full FUR Step-wise Original-answer Probability Drop")
    fig.colorbar(im, ax=ax, label="Probability drop")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "full_fur_step_heatmap_top_instances.png", dpi=180)
    plt.close(fig)


def plot_example_probabilities(row_metrics):
    full = row_metrics[row_metrics["run"] == "full"].copy()
    candidates = full.sort_values(
        ["prediction_flipped", "original_answer_mass_shift", "total_variation"],
        ascending=[False, False, False],
    ).head(4)
    for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
        letters = [c for c in OPTION_LETTERS if f"before_{c}" in row and not pd.isna(row[f"before_{c}"])]
        before = [row[f"before_{c}"] for c in letters]
        after = [row[f"after_{c}"] for c in letters]
        x = np.arange(len(letters))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width / 2, before, width, label="Before unlearning", color="#4c78a8")
        ax.bar(x + width / 2, after, width, label="After unlearning", color="#f58518")
        ax.set_xticks(x, letters)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized answer probability")
        title = f"{row['id']} step {row['step_idx']}: {row['before_pred']} -> {row['after_pred']}"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"example_{rank}_{row['id']}_step{row['step_idx']}_probabilities.png", dpi=180)
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    merged_files = {}
    for run_name, cfg in RUNS.items():
        rows, merged_path = merge_run(run_name, cfg)
        merged_files[run_name] = str(merged_path.relative_to(ROOT))
        for row in rows:
            all_rows.append(row_to_metric(row, run_name, cfg["label"]))

    row_metrics = pd.DataFrame(all_rows)
    instance_metrics = build_instance_metrics(row_metrics)

    full_row_count = int((row_metrics["run"] == "full").sum())
    summary_df = pd.DataFrame(
        [
            summarize_run(row_metrics, run_name, cfg["label"], full_row_count=full_row_count)
            for run_name, cfg in RUNS.items()
        ]
    )
    recovery_df = summarize_recovery(row_metrics, instance_metrics, full_row_count)

    full_instance = instance_metrics[instance_metrics["run"] == "full"].copy()
    full_rows = row_metrics[row_metrics["run"] == "full"].copy()
    strongest_rows = full_rows.sort_values(
        ["prediction_flipped", "original_answer_mass_shift", "total_variation", "efficacy_pct"],
        ascending=[False, False, False, False],
    ).head(20)
    strongest_instances = full_instance.sort_values(
        ["any_flip", "best_step_mass_shift", "best_step_total_variation"],
        ascending=[False, False, False],
    ).head(20)

    joined_selected = []
    full_step = full_rows.set_index(["id", "step_idx"])
    for run_name in ["last_top1", "random_top1"]:
        selected = row_metrics[row_metrics["run"] == run_name].copy()
        for _, row in selected.iterrows():
            key = (row["id"], row["step_idx"])
            if key not in full_step.index:
                continue
            full_row = full_step.loc[key]
            joined_selected.append(
                {
                    "run": run_name,
                    "run_label": RUNS[run_name]["label"],
                    "id": row["id"],
                    "step_idx": row["step_idx"],
                    "full_flip_same_step": bool(full_row["prediction_flipped"]),
                    "selective_flip": bool(row["prediction_flipped"]),
                    "full_mass_shift_same_step": float(full_row["original_answer_mass_shift"]),
                    "selective_mass_shift": float(row["original_answer_mass_shift"]),
                    "mass_shift_gap_selective_minus_full": float(row["original_answer_mass_shift"] - full_row["original_answer_mass_shift"]),
                    "full_efficacy_same_step": float(full_row["efficacy_pct"]),
                    "selective_efficacy": float(row["efficacy_pct"]),
                    "question": row["question"],
                    "cot_step": row["cot_step"],
                }
            )
    selected_vs_full_df = pd.DataFrame(joined_selected)

    row_metrics.to_csv(OUT_DIR / "row_level_metrics.csv", index=False)
    instance_metrics.to_csv(OUT_DIR / "instance_level_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    recovery_df.to_csv(OUT_DIR / "selective_recovery_summary.csv", index=False)
    strongest_rows.to_csv(OUT_DIR / "strongest_step_examples.csv", index=False)
    strongest_instances.to_csv(OUT_DIR / "strongest_instance_examples.csv", index=False)
    selected_vs_full_df.to_csv(OUT_DIR / "selected_vs_full_same_step.csv", index=False)

    save_markdown_table(summary_df.round(4), OUT_DIR / "table_overall_summary.md")
    save_markdown_table(recovery_df.round(4), OUT_DIR / "table_selective_recovery.md")
    save_markdown_table(
        strongest_rows[
            [
                "id",
                "step_idx",
                "prediction_flipped",
                "before_pred",
                "after_pred",
                "correct",
                "original_answer_mass_shift",
                "total_variation",
                "efficacy_pct",
                "specificity_pct",
                "question",
                "cot_step",
            ]
        ].round(4),
        OUT_DIR / "table_strongest_step_examples.md",
    )
    save_markdown_table(
        strongest_instances[
            [
                "id",
                "any_flip",
                "flip_steps",
                "n_steps_evaluated",
                "best_step_idx_by_mass_shift",
                "best_step_mass_shift",
                "best_step_total_variation",
                "best_step_efficacy_pct",
                "question",
            ]
        ].round(4),
        OUT_DIR / "table_strongest_instance_examples.md",
    )
    if not selected_vs_full_df.empty:
        save_markdown_table(
            selected_vs_full_df.sort_values(
                ["selective_flip", "selective_mass_shift", "full_mass_shift_same_step"],
                ascending=[False, False, False],
            )
            .head(30)
            .round(4),
            OUT_DIR / "table_selected_vs_full_same_step.md",
        )

    plot_overall(summary_df)
    plot_recovery(recovery_df)
    plot_efficacy_scatter(row_metrics)
    plot_step_heatmap(row_metrics)
    plot_example_probabilities(row_metrics)

    extra = {
        "merged_files": merged_files,
        "notes": {
            "lr_filename_fix": "Original launcher looked for lr=3e-5 merged files, while unlearn.py wrote lr=3e-05 shard files. This analysis merges the actual lr=3e-05 shard outputs.",
            "run_scope": "openbook, Llama-3.2-3B-Instruct, max_samples=100, n_unlearn=80, verify_samples=20, epochs=2, 3 shards.",
        },
        "summary": summary_df.to_dict(orient="records"),
        "recovery": recovery_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as outfile:
        json.dump(extra, outfile, ensure_ascii=False, indent=2)

    print(json.dumps(extra, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
