"""Create paper-style tables and figures for CoT/ToT FUR experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATASET_NAMES = {
    "openbook": "OpenBookQA",
    "arc-challenge": "ARC-Challenge",
}
ARM_NAMES = {
    "CoT": "CoT",
    "ToT-selected": "ToT-selected",
    "ToT-union": "ToT-union",
}
ARM_COLORS = {
    "CoT": "#2166ac",
    "ToT-selected": "#b2182b",
    "ToT-union": "#4d9221",
}
ARM_MARKERS = {"CoT": "o", "ToT-selected": "s", "ToT-union": "^"}
EXPECTED_TARGETS = {"openbook": 50, "arc-challenge": 100}


@dataclass
class Run:
    dataset: str
    arm: str
    path: Path
    mmlu_path: Path | None = None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A concurrently running experiment may have one unfinished line.
                continue
    return rows


def epochs(result: dict) -> list[tuple[int, dict]]:
    values = result.get("unlearning_results", result) or {}
    return sorted(((int(k), v) for k, v in values.items()), key=lambda pair: pair[0])


def prediction(values) -> int:
    return int(np.argmax(np.asarray(values, dtype=float)))


def normalize(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    return array / total if total else array


def is_agreement_row(row: dict) -> bool:
    return row.get("prediction") == row.get("cot_prediction")


def mmlu_score(path: Path | None) -> float:
    if not path:
        return float("nan")
    scores = []
    for row in read_jsonl(path):
        raw = row.get("mmlu_results")
        if raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        scores.append(score * 100.0 if score <= 1.0 else score)
    return float(np.mean(scores)) if scores else float("nan")


def summarize(run: Run) -> tuple[dict, dict[int, dict]]:
    rows = [row for row in read_jsonl(run.path) if row.get("unlearning_results")]
    expected = EXPECTED_TARGETS.get(run.dataset)
    question_ids = {row.get("id", row.get("question")) for row in rows}
    agree_rows = [row for row in rows if is_agreement_row(row)]
    agree_ids = {row.get("id", row.get("question")) for row in agree_rows}

    eff_by_epoch = defaultdict(list)
    spec_by_epoch = defaultdict(list)
    generated_agreement_by_epoch = defaultdict(list)
    soft_by_question = defaultdict(list)
    soft_agree_by_question = defaultdict(list)
    flips_all = set()
    flips_agree = set()

    for row in rows:
        row_epochs = epochs(row["unlearning_results"])
        if len(row_epochs) < 2:
            continue
        base = row_epochs[0][1]
        qid = row.get("id", row.get("question"))
        initial_pred = prediction(base["probs"])
        base_logprob = float(base["cot_step_prob"][0])
        base_mass = normalize(base["probs"])[initial_pred]
        base_spec = np.asarray(base["specificity_preds"])
        for epoch, state in row_epochs[1:]:
            log_ratio = float(state["cot_step_prob"][0]) - base_logprob
            eff_by_epoch[epoch].append((1.0 - math.exp(min(log_ratio, 700))) * 100.0)
            spec_by_epoch[epoch].append(
                float(np.mean(base_spec == np.asarray(state["specificity_preds"])) * 100.0)
            )
            generated_agreement_by_epoch[epoch].append(
                float(prediction(state["probs"]) == prediction(state["new_cot_probs"])) * 100.0
            )
            shift = float(base_mass - normalize(state["probs"])[initial_pred]) * 100.0
            soft_by_question[qid].append(shift)
            if is_agreement_row(row):
                soft_agree_by_question[qid].append(shift)
            if prediction(state["probs"]) != initial_pred:
                flips_all.add(qid)
                if is_agreement_row(row):
                    flips_agree.add(qid)

    all_eff = [v for values in eff_by_epoch.values() for v in values]
    all_spec = [v for values in spec_by_epoch.values() for v in values]
    all_post_agree = [v for values in generated_agreement_by_epoch.values() for v in values]
    final_epoch = max(eff_by_epoch, default=None)
    summary = {
        "dataset": run.dataset,
        "dataset_name": DATASET_NAMES.get(run.dataset, run.dataset),
        "arm": run.arm,
        "path": str(run.path),
        "status": "complete" if expected and len(question_ids) >= expected else "running",
        "n_steps": len(rows),
        "n_questions": len(question_ids),
        "expected_questions": expected or "",
        "n_agreement_questions": len(agree_ids),
        "eff": float(np.mean(all_eff)) if all_eff else float("nan"),
        "eff_final": float(np.mean(eff_by_epoch[final_epoch])) if final_epoch else float("nan"),
        "spec": float(np.mean(all_spec)) if all_spec else float("nan"),
        "spec_final": float(np.mean(spec_by_epoch[final_epoch])) if final_epoch else float("nan"),
        "gen_mmlu": mmlu_score(run.mmlu_path),
        "post_cot_agree": float(np.mean(all_post_agree)) if all_post_agree else float("nan"),
        "faithfulness": float(len(flips_agree) / len(agree_ids) * 100.0) if agree_ids else float("nan"),
        "ff_hard_all": float(len(flips_all) / len(question_ids) * 100.0) if question_ids else float("nan"),
        "ff_soft": (
            float(np.mean([max(values) for values in soft_agree_by_question.values()]))
            if soft_agree_by_question else float("nan")
        ),
        "ff_soft_all": (
            float(np.mean([max(values) for values in soft_by_question.values()]))
            if soft_by_question else float("nan")
        ),
    }
    trajectory = {}
    for epoch in sorted(eff_by_epoch):
        trajectory[epoch] = {
            "eff": float(np.mean(eff_by_epoch[epoch])),
            "spec": float(np.mean(spec_by_epoch[epoch])),
            "post_cot_agree": float(np.mean(generated_agreement_by_epoch[epoch])),
        }
    return summary, trajectory


def newest(paths) -> Path | None:
    candidates = list(paths)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def discover_runs(base: Path) -> list[Run]:
    roots = {
        "CoT": "final_result_CoT",
        "ToT-selected": "final_result_ToT",
        "ToT-union": "final_result_ToT_union",
    }
    runs = []
    for dataset in DATASET_NAMES:
        for arm, root in roots.items():
            path = newest((base / root / dataset / "LLaMA-3-3B").glob("*.out"))
            if path is None:
                continue
            mmlu = newest((base / f"{root}_mmlu" / dataset / "LLaMA-3-3B").glob("*.out"))
            runs.append(Run(dataset, arm, path, mmlu))
    return runs


def parse_runs(specs: list[str]) -> list[Run]:
    runs = []
    for spec in specs:
        fields = spec.split("|")
        if len(fields) not in (3, 4):
            raise ValueError("--run requires dataset|arm|result_path[|mmlu_result_path]")
        runs.append(Run(fields[0], fields[1], Path(fields[2]), Path(fields[3]) if len(fields) == 4 else None))
    return runs


def fmt(value, digits=1) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(summaries: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    control_headers = ["Dataset", "Reasoning", "Eff", "Spec", "Gen (MMLU)", "Post-CoT Agree", "Progress"]
    faith_headers = ["Dataset", "Reasoning", "Faithfulness (FF-HARD)", "Max FF-SOFT", "FF-HARD (all)", "Questions"]
    control_rows = []
    faith_rows = []
    for row in summaries:
        progress = f"{row['n_questions']}/{row['expected_questions'] or '?'} ({row['status']})"
        control_rows.append([
            row["dataset_name"], row["arm"], fmt(row["eff"]), fmt(row["spec"]),
            fmt(row["gen_mmlu"]), fmt(row["post_cot_agree"]), progress,
        ])
        faith_rows.append([
            row["dataset_name"], row["arm"], fmt(row["faithfulness"]), fmt(row["ff_soft"]),
            fmt(row["ff_hard_all"]), f"{row['n_agreement_questions']}/{row['n_questions']}",
        ])

    def markdown(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        lines.extend("| " + " | ".join(values) + " |" for values in rows)
        return "\n".join(lines)

    def latex(headers, rows, caption, label):
        cols = "ll" + "r" * (len(headers) - 2)
        body = [" & ".join(headers) + r" \\", r"\midrule"]
        body.extend(" & ".join(values) + r" \\" for values in rows)
        return "\n".join([
            r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}",
            rf"\label{{{label}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
            *body, r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ])

    (out / "table_controls.md").write_text(markdown(control_headers, control_rows) + "\n", encoding="utf-8")
    (out / "table_faithfulness.md").write_text(markdown(faith_headers, faith_rows) + "\n", encoding="utf-8")
    (out / "table_controls.tex").write_text(
        latex(control_headers, control_rows, "Unlearning controls for CoT and ToT.", "tab:cot-tot-controls") + "\n",
        encoding="utf-8",
    )
    (out / "table_faithfulness.tex").write_text(
        latex(faith_headers, faith_rows, "Parametric faithfulness comparison.", "tab:cot-tot-faithfulness") + "\n",
        encoding="utf-8",
    )


def comparison_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for dataset in DATASET_NAMES:
        cot = next((row for row in summaries if row["dataset"] == dataset and row["arm"] == "CoT"), None)
        tot = next((row for row in summaries if row["dataset"] == dataset and row["arm"] == "ToT-selected"), None)
        if not cot or not tot:
            continue
        rows.append({
            "dataset": dataset,
            "dataset_name": DATASET_NAMES[dataset],
            "delta_eff": tot["eff"] - cot["eff"],
            "delta_spec": tot["spec"] - cot["spec"],
            "delta_faithfulness": tot["faithfulness"] - cot["faithfulness"],
            "delta_ff_soft": tot["ff_soft"] - cot["ff_soft"],
            "delta_post_cot_agree": tot["post_cot_agree"] - cot["post_cot_agree"],
        })
    return rows


def per_question_flip(run: Run) -> tuple[set, set]:
    agreed = set()
    flipped = set()
    for row in read_jsonl(run.path):
        row_epochs = epochs(row.get("unlearning_results", {}))
        if not row_epochs:
            continue
        qid = row.get("id", row.get("question"))
        if is_agreement_row(row):
            agreed.add(qid)
        initial_pred = prediction(row_epochs[0][1]["probs"])
        if any(prediction(state["probs"]) != initial_pred for _, state in row_epochs[1:]):
            flipped.add(qid)
    return agreed, flipped


def paired_faithfulness_rows(runs: list[Run]) -> list[dict]:
    rows = []
    for dataset in DATASET_NAMES:
        cot = next((run for run in runs if run.dataset == dataset and run.arm == "CoT"), None)
        tot = next((run for run in runs if run.dataset == dataset and run.arm == "ToT-selected"), None)
        if not cot or not tot:
            continue
        cot_agree, cot_flipped = per_question_flip(cot)
        tot_agree, tot_flipped = per_question_flip(tot)
        common = cot_agree & tot_agree
        if not common:
            continue
        cot_score = len(cot_flipped & common) / len(common) * 100.0
        tot_score = len(tot_flipped & common) / len(common) * 100.0
        rows.append({
            "dataset": dataset,
            "dataset_name": DATASET_NAMES[dataset],
            "n_common_agreement": len(common),
            "cot_ff_hard": cot_score,
            "tot_ff_hard": tot_score,
            "delta_ff_hard": tot_score - cot_score,
        })
    return rows


def write_comparison_table(rows: list[dict], out: Path):
    if not rows:
        return
    headers = ["Dataset", "$\\Delta$ Eff", "$\\Delta$ Spec", "$\\Delta$ FF-HARD",
               "$\\Delta$ FF-SOFT", "$\\Delta$ Post-CoT Agree"]
    formatted = [[
        row["dataset_name"], fmt(row["delta_eff"]), fmt(row["delta_spec"]),
        fmt(row["delta_faithfulness"]), fmt(row["delta_ff_soft"]), fmt(row["delta_post_cot_agree"]),
    ] for row in rows]
    markdown_headers = ["Dataset", "Delta Eff", "Delta Spec", "Delta FF-HARD", "Delta FF-SOFT", "Delta Post-CoT Agree"]
    lines = ["| " + " | ".join(markdown_headers) + " |",
             "| " + " | ".join(["---"] * len(markdown_headers)) + " |"]
    lines.extend("| " + " | ".join(values) + " |" for values in formatted)
    (out / "table_deltas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Difference of ToT-selected relative to CoT (percentage points).}",
        r"\label{tab:cot-tot-deltas}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    tex.extend(" & ".join(values) + r" \\" for values in formatted)
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "table_deltas.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_paired_table(rows: list[dict], out: Path):
    if not rows:
        return
    formatted = [[
        row["dataset_name"], str(row["n_common_agreement"]), fmt(row["cot_ff_hard"]),
        fmt(row["tot_ff_hard"]), fmt(row["delta_ff_hard"]),
    ] for row in rows]
    headers = ["Dataset", "Common agreement N", "CoT FF-HARD", "ToT FF-HARD", "Delta FF-HARD"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(values) + " |" for values in formatted)
    (out / "table_paired_faithfulness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex_headers = ["Dataset", "Common agreement $N$", "CoT FF-HARD", "ToT FF-HARD", "$\\Delta$ FF-HARD"]
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{FF-HARD on the common subset where both CoT and ToT initially agree with the direct prediction.}",
        r"\label{tab:paired-faithfulness}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        " & ".join(tex_headers) + r" \\",
        r"\midrule",
    ]
    tex.extend(" & ".join(values) + r" \\" for values in formatted)
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "table_paired_faithfulness.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def tot_selection_rows(base: Path) -> list[dict]:
    rows = []
    for dataset in DATASET_NAMES:
        path = base / "final_tree_ToT" / dataset / "Llama-3.2-3B-Instruct" / "mode_selection_validation_n=30_s=1001.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for mode, values in report["summaries"].items():
            rows.append({
                "dataset": dataset,
                "dataset_name": DATASET_NAMES[dataset],
                "mode": mode,
                "selected": mode == report["chosen_mode"],
                "accuracy": values["accuracy"] * 100.0,
                "winning_confidence": values["mean_winning_confidence"] * 100.0,
                "distinct_paths": values["mean_distinct_paths"],
                "vote_entropy": values["mean_vote_entropy"],
            })
    return rows


def write_selection_table(rows: list[dict], out: Path):
    if not rows:
        return
    headers = ["Dataset", "Mode", "Selected", "Accuracy", "Confidence", "Paths", "Vote entropy"]
    formatted = [[
        row["dataset_name"], row["mode"].replace("_", "-"), "Yes" if row["selected"] else "No",
        fmt(row["accuracy"]), fmt(row["winning_confidence"]), fmt(row["distinct_paths"], 2),
        fmt(row["vote_entropy"], 3),
    ] for row in rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(values) + " |" for values in formatted)
    (out / "table_tot_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Validation-based selection of the ToT reasoning procedure.}",
        r"\label{tab:tot-selection}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    tex.extend(" & ".join(values) + r" \\" for values in formatted)
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (out / "table_tot_selection.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def strict_json_values(value):
    if isinstance(value, dict):
        return {key: strict_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json_values(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def save_figure(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_eff_spec(summaries: list[dict], out: Path):
    datasets = [dataset for dataset in DATASET_NAMES if any(r["dataset"] == dataset for r in summaries)]
    arms = [arm for arm in ARM_NAMES if any(r["arm"] == arm for r in summaries)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.1 * len(datasets), 4.1), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for row in (r for r in summaries if r["dataset"] == dataset and not math.isnan(r["eff"])):
            size = 55 + (row["faithfulness"] if not math.isnan(row["faithfulness"]) else 0.0) * 2
            ax.scatter(row["eff"], row["spec"], s=size, marker=ARM_MARKERS[row["arm"]],
                       color=ARM_COLORS[row["arm"]], edgecolor="white", linewidth=0.8, alpha=0.9)
            ax.annotate(row["arm"], (row["eff"], row["spec"]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        ax.set_title(DATASET_NAMES[dataset], fontweight="bold")
        ax.set_xlabel("Efficacy (Eff, %)")
        ax.set_ylabel("Specificity (Spec, %)")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.axhline(95, color="#555555", linewidth=0.8, linestyle="--")
    handles = [Line2D([0], [0], marker=ARM_MARKERS[arm], color="w", markerfacecolor=ARM_COLORS[arm],
                      label=arm, markersize=8) for arm in arms]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save_figure(fig, out, "eff_spec_faithfulness")


def plot_faithfulness(summaries: list[dict], out: Path):
    datasets = [dataset for dataset in DATASET_NAMES if any(r["dataset"] == dataset for r in summaries)]
    arms = [arm for arm in ARM_NAMES if any(r["arm"] == arm for r in summaries)]
    if not datasets or not arms:
        return
    fig, ax = plt.subplots(figsize=(max(6.2, len(datasets) * 2.5), 4.2))
    x = np.arange(len(datasets))
    width = 0.72 / len(arms)
    for index, arm in enumerate(arms):
        values = []
        for dataset in datasets:
            item = next((r for r in summaries if r["dataset"] == dataset and r["arm"] == arm), None)
            values.append(item["faithfulness"] if item else float("nan"))
        offset = (index - (len(arms) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width * 0.88, label=arm, color=ARM_COLORS[arm])
        for bar, value in zip(bars, values):
            if not math.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, fmt(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [DATASET_NAMES[d] for d in datasets])
    ax.set_ylabel("Faithfulness (FF-HARD, %)")
    ax.set_ylim(0, 105)
    ax.set_title("Parametric Faithfulness: CoT vs ToT", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, "faithfulness_comparison")


def plot_trajectories(summaries: list[dict], trajectories: dict[tuple[str, str], dict], out: Path):
    datasets = [dataset for dataset in DATASET_NAMES if any(r["dataset"] == dataset for r in summaries)]
    arms = [arm for arm in ARM_NAMES if any(r["arm"] == arm for r in summaries)]
    if not datasets:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.0 * len(datasets), 6.1), squeeze=False, sharex="col")
    for col, dataset in enumerate(datasets):
        for row in (r for r in summaries if r["dataset"] == dataset):
            trajectory = trajectories.get((dataset, row["arm"]), {})
            if not trajectory:
                continue
            xs = sorted(trajectory)
            axes[0, col].plot(xs, [trajectory[e]["eff"] for e in xs], marker=ARM_MARKERS[row["arm"]],
                              color=ARM_COLORS[row["arm"]], label=row["arm"], linewidth=1.8)
            axes[1, col].plot(xs, [trajectory[e]["spec"] for e in xs], marker=ARM_MARKERS[row["arm"]],
                              color=ARM_COLORS[row["arm"]], label=row["arm"], linewidth=1.8)
        axes[0, col].set_title(DATASET_NAMES[dataset], fontweight="bold")
        axes[0, col].set_ylabel("Eff (%)")
        axes[1, col].set_ylabel("Spec (%)")
        axes[1, col].set_xlabel("Unlearning epoch")
        axes[0, col].set_ylim(0, 105)
        axes[1, col].set_ylim(0, 105)
        axes[1, col].axhline(95, color="#555555", linewidth=0.8, linestyle="--")
    handles = [Line2D([0], [0], color=ARM_COLORS[arm], marker=ARM_MARKERS[arm], label=arm) for arm in arms]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save_figure(fig, out, "unlearning_trajectories")


def plot_tot_selection(base: Path, out: Path):
    reports = []
    for dataset in DATASET_NAMES:
        path = base / "final_tree_ToT" / dataset / "Llama-3.2-3B-Instruct" / "mode_selection_validation_n=30_s=1001.json"
        if path.exists():
            reports.append((dataset, json.loads(path.read_text(encoding="utf-8"))))
    if not reports:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    modes = ["sample_select", "beam_prune"]
    colors = ["#b2182b", "#762a83"]
    x = np.arange(len(reports))
    width = 0.34
    for index, mode in enumerate(modes):
        axes[0].bar(x + (index - 0.5) * width, [report["summaries"][mode]["accuracy"] * 100 for _, report in reports],
                    width, color=colors[index], label=mode.replace("_", "-"))
        axes[1].bar(x + (index - 0.5) * width, [report["summaries"][mode]["mean_distinct_paths"] for _, report in reports],
                    width, color=colors[index], label=mode.replace("_", "-"))
    labels = [DATASET_NAMES[dataset] for dataset, _ in reports]
    axes[0].set_xticks(x, labels)
    axes[1].set_xticks(x, labels)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Mean distinct paths")
    axes[0].set_title("ToT Selector Quality", fontweight="bold")
    axes[1].set_title("Tree Diversity", fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "tot_mode_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=".", help="Experiment directory containing final_result_* folders.")
    parser.add_argument("--run", action="append", default=[],
                        help="Explicit dataset|arm|result_path[|mmlu_result_path], repeatable.")
    parser.add_argument("--output_dir", default="final_analysis/LLaMA-3-3B")
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.output_dir)
    data_dir = out / "data"
    table_dir = out / "tables"
    figure_dir = out / "figures"
    for directory in (data_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.run) if args.run else discover_runs(base)
    if not runs:
        raise SystemExit("No CoT/ToT result files found.")
    summaries = []
    trajectories = {}
    for run in runs:
        summary, trajectory = summarize(run)
        summaries.append(summary)
        trajectories[(run.dataset, run.arm)] = trajectory
    summaries.sort(key=lambda row: (row["dataset"], list(ARM_NAMES).index(row["arm"])))
    deltas = comparison_rows(summaries)
    paired = paired_faithfulness_rows(runs)
    selections = tot_selection_rows(base)
    write_csv(summaries, data_dir / "metrics.csv")
    write_csv(deltas, data_dir / "comparison_deltas.csv")
    write_csv(paired, data_dir / "paired_faithfulness.csv")
    write_csv(selections, data_dir / "tot_selection.csv")
    (data_dir / "metrics.json").write_text(
        json.dumps(strict_json_values(summaries), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tables(summaries, table_dir)
    write_comparison_table(deltas, table_dir)
    write_paired_table(paired, table_dir)
    write_selection_table(selections, table_dir)
    style()
    plot_eff_spec(summaries, figure_dir)
    plot_faithfulness(summaries, figure_dir)
    plot_trajectories(summaries, trajectories, figure_dir)
    plot_tot_selection(base, figure_dir)
    print(json.dumps(strict_json_values(summaries), ensure_ascii=False, indent=2))
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
