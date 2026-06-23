"""Generate tables and figures for the Qwen3 C-Eval FUR comparison report."""

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
import numpy as np


@dataclass(frozen=True)
class Run:
    key: str
    model: str
    cot_samples: int
    epochs: int
    path: Path

    @property
    def label(self) -> str:
        return f"{self.model}, n={self.cot_samples}, E={self.epochs}"


def default_runs(root: Path) -> list[Run]:
    result_root = root / "final_results" / "ceval"
    return [
        Run(
            "8b_n250_e5",
            "Qwen3-8B",
            250,
            5,
            result_root
            / "Qwen3-8B"
            / "npo_KL_sentencize_s=True_lr=1e-05_rs=1001_n=250_pos=False_ff2=True.out",
        ),
        Run(
            "4b_n250_e5",
            "Qwen3-4B",
            250,
            5,
            result_root
            / "Qwen3-4B"
            / "npo_KL_sentencize_s=True_lr=1e-05_rs=1001_n=250_pos=False_ff2=True.out",
        ),
        Run(
            "4b_n220_e10",
            "Qwen3-4B",
            220,
            10,
            result_root
            / "Qwen3-4B"
            / "npo_KL_sentencize_s=True_lr=1e-05_rs=1001_n=220_pos=False_ff2=True.out",
        ),
    ]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as infile:
        for lineno, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{lineno}") from exc
    return rows


def pred(values) -> int:
    return int(np.argmax(np.asarray(values, dtype=float)))


def normalized(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    return values / total if total else values


def question_id(row: dict) -> str:
    return str(row.get("id", row["question"]))


def eligible(row: dict) -> bool:
    return row["prediction"] == row["cot_prediction"]


def load_and_audit(run: Run) -> tuple[list[dict], dict]:
    rows = read_jsonl(run.path)
    keys = [(question_id(row), int(row["step_idx"])) for row in rows]
    expected_epochs = list(range(run.epochs + 1))
    bad_epochs = [
        key
        for key, row in zip(keys, rows)
        if sorted(int(e) for e in row["unlearning_results"]) != expected_epochs
    ]
    duplicate_steps = len(keys) - len(set(keys))
    questions = {question_id(row) for row in rows}
    eligible_questions = {question_id(row) for row in rows if eligible(row)}
    expected_targets = run.cot_samples - 20
    if bad_epochs or duplicate_steps or len(questions) != expected_targets:
        raise ValueError(
            f"Integrity failure for {run.label}: questions={len(questions)}/"
            f"{expected_targets}, duplicates={duplicate_steps}, bad_epochs={len(bad_epochs)}"
        )
    audit = {
        "key": run.key,
        "model": run.model,
        "cot_samples": run.cot_samples,
        "epochs": run.epochs,
        "target_questions": len(questions),
        "expected_targets": expected_targets,
        "step_rows": len(rows),
        "eligible_questions": len(eligible_questions),
        "duplicate_steps": duplicate_steps,
        "incomplete_epoch_rows": len(bad_epochs),
        "status": "complete",
        "path": str(run.path),
    }
    return rows, audit


def step_values(row: dict, epoch: int) -> dict:
    base = row["unlearning_results"]["0"]
    state = row["unlearning_results"][str(epoch)]
    base_probs = normalized(base["probs"])
    state_probs = normalized(state["probs"])
    initial_pred = pred(base_probs)
    base_logp = float(base["cot_step_prob"][0])
    state_logp = float(state["cot_step_prob"][0])
    return {
        "flip": initial_pred != pred(state_probs),
        "eff": (1.0 - math.exp(state_logp - base_logp)) * 100.0,
        "spec": float(
            np.mean(
                np.asarray(base["specificity_preds"], dtype=int)
                == np.asarray(state["specificity_preds"], dtype=int)
            )
            * 100.0
        ),
        "soft": float(base_probs[initial_pred] - state_probs[initial_pred]) * 100.0,
    }


def summary(rows: list[dict], epochs: list[int], agreement_only: bool = False) -> dict:
    chosen = [row for row in rows if not agreement_only or eligible(row)]
    questions = {question_id(row) for row in chosen}
    flipped_questions: set[str] = set()
    flipped_steps: set[tuple[str, int]] = set()
    eff, spec, soft = [], [], []
    maximum_soft: dict[str, list[float]] = defaultdict(list)
    for row in chosen:
        qid = question_id(row)
        step_key = (qid, int(row["step_idx"]))
        for epoch in epochs:
            metrics = step_values(row, epoch)
            eff.append(metrics["eff"])
            spec.append(metrics["spec"])
            soft.append(metrics["soft"])
            maximum_soft[qid].append(metrics["soft"])
            if metrics["flip"]:
                flipped_questions.add(qid)
                flipped_steps.add(step_key)
    return {
        "questions": len(questions),
        "steps": len(chosen),
        "eff": float(np.mean(eff)),
        "spec": float(np.mean(spec)),
        "ff_hard": len(flipped_questions) / len(questions) * 100.0,
        "step_ff_hard": len(flipped_steps) / len(chosen) * 100.0,
        "mean_ff_soft": float(np.mean(soft)),
        "max_ff_soft": float(np.mean([max(v) for v in maximum_soft.values()])),
        "flipped_questions": len(flipped_questions),
        "flipped_steps": len(flipped_steps),
    }


def analysis_rows(run: Run, rows: list[dict]) -> tuple[dict, dict, list[dict]]:
    standard_epochs = list(range(1, min(5, run.epochs) + 1))
    controls = summary(rows, standard_epochs)
    faithful = summary(rows, standard_epochs, agreement_only=True)
    standard = {
        "key": run.key,
        "model": run.model,
        "cot_samples": run.cot_samples,
        "evaluated_epochs": len(standard_epochs),
        "target_questions": controls["questions"],
        "step_rows": controls["steps"],
        "eff": controls["eff"],
        "spec": controls["spec"],
        "eligible_questions": faithful["questions"],
        "eligible_steps": faithful["steps"],
        "ff_hard": faithful["ff_hard"],
        "step_ff_hard": faithful["step_ff_hard"],
        "max_ff_soft": faithful["max_ff_soft"],
        "ff_hard_all": controls["ff_hard"],
    }

    endpoint_all = summary(rows, [run.epochs])
    endpoint_eligible = summary(rows, [run.epochs], agreement_only=True)
    endpoint = {
        "key": run.key,
        "model": run.model,
        "cot_samples": run.cot_samples,
        "endpoint_epoch": run.epochs,
        "eff": endpoint_all["eff"],
        "spec": endpoint_all["spec"],
        "ff_hard_all": endpoint_all["ff_hard"],
        "step_ff_hard_all": endpoint_all["step_ff_hard"],
        "mean_ff_soft_all": endpoint_all["mean_ff_soft"],
        "eligible_questions": endpoint_eligible["questions"],
        "ff_hard_eligible": endpoint_eligible["ff_hard"],
    }

    trajectory = []
    for epoch in range(1, run.epochs + 1):
        point = summary(rows, [epoch])
        point_eligible = summary(rows, [epoch], agreement_only=True)
        trajectory.append(
            {
                "key": run.key,
                "model": run.model,
                "cot_samples": run.cot_samples,
                "epoch": epoch,
                "eff": point["eff"],
                "spec": point["spec"],
                "ff_hard_all": point["ff_hard"],
                "step_ff_hard_all": point["step_ff_hard"],
                "mean_ff_soft_all": point["mean_ff_soft"],
                "ff_hard_eligible": point_eligible["ff_hard"],
            }
        )
    return standard, endpoint, trajectory


def paired_model_comparison(rows_by_key: dict[str, list[dict]]) -> list[dict]:
    left = rows_by_key["8b_n250_e5"]
    right = rows_by_key["4b_n250_e5"]
    left_eligible = {question_id(row) for row in left if eligible(row)}
    right_eligible = {question_id(row) for row in right if eligible(row)}
    common = left_eligible & right_eligible
    output = []
    for key, rows in (("8b_n250_e5", left), ("4b_n250_e5", right)):
        selected = [row for row in rows if question_id(row) in common]
        through = summary(selected, list(range(1, 6)))
        endpoint = summary(selected, [5])
        output.append(
            {
                "key": key,
                "model": key.split("_")[0].upper().replace("B", "B"),
                "common_eligible_questions": len(common),
                "ff_hard_through_e5": through["ff_hard"],
                "ff_hard_endpoint_e5": endpoint["ff_hard"],
                "max_ff_soft_through_e5": through["max_ff_soft"],
            }
        )
    return output


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def latex_table(headers: list[str], rows: list[list[str]], cols: str, caption: str, label: str) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write_tables(
    audits: list[dict], standards: list[dict], endpoints: list[dict], paired: list[dict], out: Path
):
    out.mkdir(parents=True, exist_ok=True)
    config_rows = [
        [
            row["model"],
            str(row["cot_samples"]),
            str(row["target_questions"]),
            str(row["step_rows"]),
            str(row["epochs"]),
            str(row["eligible_questions"]),
        ]
        for row in audits
    ]
    (out / "table_config.tex").write_text(
        latex_table(
            ["模型", "CoT 数", "目标题", "步骤数", "Epoch", "可比题"],
            config_rows,
            "lrrrrr",
            "三组输出的完整性审计。可比题指初始 direct 与 CoT 预测一致的题目。",
            "tab:config",
        ),
        encoding="utf-8",
    )

    standard_rows = [
        [
            row["model"] + (r" ($n=220$, 截至 E5)" if row["cot_samples"] == 220 else ""),
            fmt(row["eff"]),
            fmt(row["spec"]),
            f"{fmt(row['ff_hard'])}\\%",
            f"{fmt(row['step_ff_hard'])}\\%",
            fmt(row["max_ff_soft"]),
        ]
        for row in standards
    ]
    (out / "table_standard.tex").write_text(
        latex_table(
            ["实验", "Eff", "Spec", "FF-HARD", "Step FF", "Max FF-SOFT"],
            standard_rows,
            "lrrrrr",
            "统一 5 epoch 预算下的论文式比较。Eff/Spec 使用全部步骤；忠实度只使用初始预测一致的题目。",
            "tab:standard",
        ),
        encoding="utf-8",
    )

    endpoint_rows = [
        [
            row["model"] + f" ($n={row['cot_samples']}$)",
            str(row["endpoint_epoch"]),
            fmt(row["eff"]),
            fmt(row["spec"]),
            f"{fmt(row['ff_hard_all'])}\\%",
            f"{fmt(row['ff_hard_eligible'])}\\%",
            fmt(row["mean_ff_soft_all"]),
        ]
        for row in endpoints
    ]
    (out / "table_endpoint.tex").write_text(
        latex_table(
            ["实验", "终点 E", "Eff", "Spec", "FF-HARD(all)", "FF-HARD(agree)", "Mean soft"],
            endpoint_rows,
            "lrrrrrr",
            "各结果文件最终 checkpoint 的 endpoint 汇总；此口径与既有 8B 摘要的最终态统计一致。",
            "tab:endpoint",
        ),
        encoding="utf-8",
    )

    paired_rows = [
        [
            "Qwen3-" + row["model"],
            str(row["common_eligible_questions"]),
            f"{fmt(row['ff_hard_through_e5'])}\\%",
            f"{fmt(row['ff_hard_endpoint_e5'])}\\%",
            fmt(row["max_ff_soft_through_e5"]),
        ]
        for row in paired
    ]
    (out / "table_paired.tex").write_text(
        latex_table(
            ["模型", "共同可比题", "FF-HARD(1--5)", "FF-HARD(E5)", "Max FF-SOFT"],
            paired_rows,
            "lrrrr",
            "两种模型在共同初始可比题目上的比较。CoT 文本由各自模型生成，故不是相同步骤的逐步配对。",
            "tab:paired",
        ),
        encoding="utf-8",
    )


COLORS = {
    "8b_n250_e5": "#2166ac",
    "4b_n250_e5": "#b2182b",
    "4b_n220_e10": "#1b7837",
}
NAMES = {
    "8b_n250_e5": "8B n=250, E=5",
    "4b_n250_e5": "4B n=250, E=5",
    "4b_n220_e10": "4B n=220, E=10",
}


def plot_standard(standards: list[dict], out: Path):
    labels = ["8B\nn=250", "4B\nn=250", "4B n=220\ntruncated E=5"]
    metrics = [("eff", "Efficacy"), ("spec", "Specificity"), ("ff_hard", "FF-HARD (agreement set)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    colors = [COLORS[row["key"]] for row in standards]
    for ax, (metric, title) in zip(axes, metrics):
        vals = [row[metric] for row in standards]
        bars = ax.bar(range(3), vals, color=colors, alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105 if metric != "ff_hard" else max(vals) + 5)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=8)
    fig.suptitle("Matched intervention budget: epochs 1--5", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "standard_five_epoch_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "standard_five_epoch_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(trajectories: list[dict], out: Path):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for point in trajectories:
        grouped[point["key"]].append(point)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    panels = [("eff", "Endpoint efficacy"), ("spec", "Endpoint specificity"), ("ff_hard_all", "Endpoint FF-HARD (all)")]
    for ax, (field, title) in zip(axes, panels):
        for key, points in grouped.items():
            points = sorted(points, key=lambda row: row["epoch"])
            ax.plot(
                [row["epoch"] for row in points],
                [row[field] for row in points],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=COLORS[key],
                label=NAMES[key],
            )
        ax.axvline(5, linestyle="--", color="black", alpha=0.35, linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Unlearning epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Percent")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "epoch_trajectories.pdf", bbox_inches="tight")
    fig.savefig(out / "epoch_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_soft_histograms(runs: list[Run], rows_by_key: dict[str, list[dict]], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.15), sharey=True)
    bins = [-100, -50, -25, 0, 25, 50, 100]
    for ax, run in zip(axes, runs):
        shifts = [step_values(row, run.epochs)["soft"] for row in rows_by_key[run.key]]
        ax.hist(shifts, bins=bins, color=COLORS[run.key], alpha=0.88, edgecolor="white")
        ax.axvline(0, linewidth=0.8, color="black")
        ax.set_title(NAMES[run.key], fontsize=9)
        ax.set_xlabel("FF-SOFT: removed mass (%)", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Reasoning steps")
    fig.suptitle("Endpoint probability-mass shifts", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "soft_shift_histograms.pdf", bbox_inches="tight")
    fig.savefig(out / "soft_shift_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    data_dir = output / "data"
    table_dir = output / "tables"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = default_runs(root)
    rows_by_key, audits, standards, endpoints, trajectories = {}, [], [], [], []
    for run in runs:
        rows, audit = load_and_audit(run)
        standard, endpoint, trajectory = analysis_rows(run, rows)
        rows_by_key[run.key] = rows
        audits.append(audit)
        standards.append(standard)
        endpoints.append(endpoint)
        trajectories.extend(trajectory)
    paired = paired_model_comparison(rows_by_key)

    write_csv(audits, data_dir / "run_audit.csv")
    write_csv(standards, data_dir / "standard_five_epoch.csv")
    write_csv(endpoints, data_dir / "endpoint_metrics.csv")
    write_csv(trajectories, data_dir / "epoch_trajectories.csv")
    write_csv(paired, data_dir / "paired_models.csv")
    write_tables(audits, standards, endpoints, paired, table_dir)
    plot_standard(standards, figure_dir)
    plot_trajectories(trajectories, figure_dir)
    plot_soft_histograms(runs, rows_by_key, figure_dir)

    payload = {"audit": audits, "standard_five_epoch": standards, "endpoints": endpoints, "paired": paired}
    (data_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
