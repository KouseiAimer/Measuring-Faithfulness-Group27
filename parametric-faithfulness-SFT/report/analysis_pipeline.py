#!/usr/bin/env python3
"""Generate statistical tables and figures for the SFT faithfulness report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = HERE / "data"
TABLE_DIR = HERE / "tables"
FIGURE_DIR = HERE / "figures"
SEED = 1001
BOOTSTRAP_REPS = 10000

ARMS = {
    "base": {
        "label": "Base LLaMA-3B",
        "short": "Base",
        "color": "#2166AC",
        "marker": "o",
    },
    "sft": {
        "label": "DeepSeek-Revision SFT",
        "short": "SFT",
        "color": "#B2182B",
        "marker": "s",
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(old, new)
    return text


def write_table(
    name: str,
    caption: str,
    label: str,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    align: str | None = None,
) -> None:
    row_list = [[str(cell) for cell in row] for row in rows]
    md_lines = [
        f"**{caption}**",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in row_list)
    (TABLE_DIR / f"{name}.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    tex_lines.extend(
        " & ".join(latex_escape(cell) for cell in row) + r" \\" for row in row_list
    )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLE_DIR / f"{name}.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def percent(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def bootstrap_mean_ci(
    differences: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    if differences.size == 0:
        return float("nan"), float("nan")
    samples = rng.choice(
        differences, size=(BOOTSTRAP_REPS, differences.size), replace=True
    ).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def mcnemar_exact(before: np.ndarray, after: np.ndarray) -> dict[str, float | int]:
    improved = int(np.sum((before == 0) & (after == 1)))
    degraded = int(np.sum((before == 1) & (after == 0)))
    discordant = improved + degraded
    p_value = (
        float(binomtest(improved, discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "improved": improved,
        "degraded": degraded,
        "discordant": discordant,
        "p_value": p_value,
    }


def analyze_arm(arm: str) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    cot_path = ARTIFACTS / "eval_cots" / f"{arm}_openbookqa_test.jsonl"
    fur_path = ARTIFACTS / "fur_results" / f"{arm}.jsonl"
    cots = [row for row in load_jsonl(cot_path) if row.get("role") == "target"]
    fur_rows = load_jsonl(fur_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fur_rows:
        grouped[str(row["id"])].append(row)

    question_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for item in cots:
        qid = str(item["id"])
        rows = sorted(grouped.get(qid, []), key=lambda row: int(row["step_idx"]))
        shifts: list[float] = []
        hard_flip = False
        for row in rows:
            epoch_zero = row["epochs"]["0"]
            initial_probs = np.asarray(epoch_zero["probs"], dtype=float)
            initial_pred = int(np.argmax(initial_probs))
            initial_logp = float(epoch_zero["cot_step_logprob"])
            for epoch in range(1, 6):
                state = row["epochs"][str(epoch)]
                probs = np.asarray(state["probs"], dtype=float)
                prediction = int(np.argmax(probs))
                efficacy = (1.0 - math.exp(float(state["cot_step_logprob"]) - initial_logp)) * 100.0
                base_specificity = np.asarray(epoch_zero["specificity_preds"])
                current_specificity = np.asarray(state["specificity_preds"])
                specificity = float(np.mean(base_specificity == current_specificity) * 100.0)
                shift = float((initial_probs[initial_pred] - probs[initial_pred]) * 100.0)
                hard_flip = hard_flip or prediction != initial_pred
                shifts.append(shift)
                state_rows.append(
                    {
                        "arm": arm,
                        "model": ARMS[arm]["label"],
                        "id": qid,
                        "step_idx": int(row["step_idx"]),
                        "epoch": epoch,
                        "efficacy": efficacy,
                        "specificity": specificity,
                        "soft_shift": shift,
                        "prediction_changed": int(prediction != initial_pred),
                    }
                )
        direct_probs = np.asarray(item["direct_probs"], dtype=float)
        sorted_probs = np.sort(direct_probs)
        question_rows.append(
            {
                "arm": arm,
                "model": ARMS[arm]["label"],
                "id": qid,
                "direct_correct": int(item["direct_answer"] == item["correct_letter"]),
                "cot_correct": int(item["cot_answer"] == item["correct_letter"]),
                "agreement": int(item["direct_answer"] == item["cot_answer"]),
                "generated_steps": len(item["segmented_cot"]),
                "valid_steps": len(rows),
                "cot_words": len(item["cot"].split()),
                "initial_margin": float((sorted_probs[-1] - sorted_probs[-2]) * 100.0),
                "hard_flip": int(hard_flip),
                "max_soft_shift": float(max(shifts)) if shifts else float("nan"),
            }
        )

    qdf = pd.DataFrame(question_rows)
    sdf = pd.DataFrame(state_rows)
    agreed = qdf[qdf["agreement"] == 1]
    final_states = sdf[sdf["epoch"] == 5]
    summary: dict[str, float | int] = {
        "target_cots": int(len(cots)),
        "measured_questions": int(qdf["id"].nunique()),
        "measured_steps": int(qdf["valid_steps"].sum()),
        "direct_accuracy": float(qdf["direct_correct"].mean() * 100.0),
        "cot_accuracy": float(qdf["cot_correct"].mean() * 100.0),
        "direct_cot_agreement": float(qdf["agreement"].mean() * 100.0),
        "mean_steps": float(qdf["generated_steps"].mean()),
        "mean_words": float(qdf["cot_words"].mean()),
        "efficacy": float(sdf["efficacy"].mean()),
        "efficacy_final": float(final_states["efficacy"].mean()),
        "specificity": float(sdf["specificity"].mean()),
        "specificity_final": float(final_states["specificity"].mean()),
        "ff_hard_all": float(qdf["hard_flip"].mean() * 100.0),
        "ff_hard_agree": float(agreed["hard_flip"].mean() * 100.0),
        "ff_soft_all": float(qdf["max_soft_shift"].mean()),
        "ff_soft_agree": float(agreed["max_soft_shift"].mean()),
        "eligible_agreement_questions": int(len(agreed)),
        "mean_initial_margin": float(qdf["initial_margin"].mean()),
        "mean_initial_margin_agree": float(agreed["initial_margin"].mean()),
    }
    return summary, qdf, sdf


def build_integrity_table() -> list[list[str]]:
    paths = [
        ("Student drafts", ARTIFACTS / "data" / "train_student_drafts.jsonl", 2000),
        ("Teacher revisions", ARTIFACTS / "data" / "teacher_revisions.jsonl", 2000),
        ("Accepted SFT records", ARTIFACTS / "data" / "sft_train.jsonl", 1852),
        ("Base evaluation CoTs", ARTIFACTS / "eval_cots" / "base_openbookqa_test.jsonl", 120),
        ("SFT evaluation CoTs", ARTIFACTS / "eval_cots" / "sft_openbookqa_test.jsonl", 120),
        ("Base valid FUR steps", ARTIFACTS / "fur_results" / "base.jsonl", 495),
        ("SFT valid FUR steps", ARTIFACTS / "fur_results" / "sft.jsonl", 311),
    ]
    output = []
    for name, path, expected in paths:
        observed = len(load_jsonl(path))
        output.append([name, str(expected), str(observed), "PASS" if observed == expected else "CHECK"])
    return output


def teacher_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    records = load_jsonl(ARTIFACTS / "data" / "teacher_revisions.jsonl")
    statuses: Counter[str] = Counter()
    teachers = Counter()
    for row in records:
        teachers[str(row.get("teacher_model", row.get("model", "unknown")))] += 1
        if row.get("accepted", False):
            statuses["accepted"] += 1
        else:
            reason = str(row.get("reject_reason", row.get("reason", "rejected")))
            statuses[reason] += 1
    # Older generated outputs record rejection details in validation_errors.
    if statuses.get("rejected") and len(statuses) == 2:
        statuses = Counter()
        for row in records:
            if row.get("accepted", False):
                statuses["accepted"] += 1
                continue
            errors = row.get("validation_errors", [])
            statuses[str(errors[0]) if errors else "rejected"] += 1
    preferred_order = [
        "accepted",
        "wrong_teacher_answer",
        "invalid_json",
        "answer_leak",
        "sentence_count",
        "rationale_length",
    ]
    total = len(records)
    rows = []
    for status in preferred_order + sorted(set(statuses) - set(preferred_order)):
        if statuses[status]:
            rows.append(
                {
                    "status": status,
                    "count": int(statuses[status]),
                    "percentage": float(statuses[status] / total * 100.0),
                }
            )
    return pd.DataFrame(rows), {"total": total, "teacher_models": dict(teachers)}


def paired_statistics(qdf: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    base = qdf[qdf["arm"] == "base"].set_index("id")
    sft = qdf[qdf["arm"] == "sft"].set_index("id")
    paired = base.join(sft, lsuffix="_base", rsuffix="_sft", how="inner")
    common = paired[(paired["agreement_base"] == 1) & (paired["agreement_sft"] == 1)]
    rng = np.random.default_rng(SEED)
    output: list[dict[str, Any]] = []

    def append_row(
        metric: str,
        frame: pd.DataFrame,
        base_col: str,
        sft_col: str,
        scale: float,
        test_binary: bool = False,
    ) -> None:
        before = frame[base_col].to_numpy(dtype=float) * scale
        after = frame[sft_col].to_numpy(dtype=float) * scale
        differences = after - before
        low, high = bootstrap_mean_ci(differences, rng)
        row: dict[str, Any] = {
            "metric": metric,
            "n": int(len(frame)),
            "base": float(before.mean()),
            "sft": float(after.mean()),
            "delta": float(differences.mean()),
            "ci_low": low,
            "ci_high": high,
            "p_value": float("nan"),
            "improved": "",
            "degraded": "",
        }
        if test_binary:
            test = mcnemar_exact(
                frame[base_col].to_numpy(dtype=int), frame[sft_col].to_numpy(dtype=int)
            )
            row.update(
                {
                    "p_value": test["p_value"],
                    "improved": test["improved"],
                    "degraded": test["degraded"],
                }
            )
        output.append(row)

    append_row("Direct accuracy", paired, "direct_correct_base", "direct_correct_sft", 100.0, True)
    append_row("CoT accuracy", paired, "cot_correct_base", "cot_correct_sft", 100.0, True)
    append_row("Direct-CoT agreement", paired, "agreement_base", "agreement_sft", 100.0, True)
    append_row("CoT steps", paired, "generated_steps_base", "generated_steps_sft", 1.0)
    append_row("CoT words", paired, "cot_words_base", "cot_words_sft", 1.0)
    append_row("FF-HARD (common agree)", common, "hard_flip_base", "hard_flip_sft", 100.0, True)
    append_row("FF-SOFT (common agree)", common, "max_soft_shift_base", "max_soft_shift_sft", 1.0)
    append_row("Initial margin (common agree)", common, "initial_margin_base", "initial_margin_sft", 1.0)

    result = pd.DataFrame(output)
    details = {
        "paired_questions": int(len(paired)),
        "common_agreement_questions": int(len(common)),
        "base_agreement_only": int(
            ((paired["agreement_base"] == 1) & (paired["agreement_sft"] == 0)).sum()
        ),
        "sft_agreement_only": int(
            ((paired["agreement_base"] == 0) & (paired["agreement_sft"] == 1)).sum()
        ),
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": SEED,
    }
    return result, details, paired.reset_index()


def trajectories(sdf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        rows.append(
            {
                "arm": arm,
                "model": ARMS[arm]["label"],
                "epoch": 0,
                "efficacy": 0.0,
                "specificity": 100.0,
                "soft_shift": 0.0,
            }
        )
        part = sdf[sdf["arm"] == arm]
        for epoch, epoch_df in part.groupby("epoch"):
            rows.append(
                {
                    "arm": arm,
                    "model": ARMS[arm]["label"],
                    "epoch": int(epoch),
                    "efficacy": float(epoch_df["efficacy"].mean()),
                    "specificity": float(epoch_df["specificity"].mean()),
                    "soft_shift": float(epoch_df["soft_shift"].mean()),
                }
            )
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def save_plot(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_eff_spec(summary: dict[str, dict[str, float | int]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for arm, values in summary.items():
        size = 45 + float(values["ff_hard_agree"]) * 8
        ax.scatter(
            values["efficacy"],
            values["specificity"],
            s=size,
            marker=ARMS[arm]["marker"],
            color=ARMS[arm]["color"],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.88,
            label=f'{ARMS[arm]["label"]} (FF-HARD={values["ff_hard_agree"]:.1f})',
        )
        offset = (-45, 12) if arm == "sft" else (12, -18)
        ax.annotate(
            ARMS[arm]["short"],
            (values["efficacy"], values["specificity"]),
            xytext=offset,
            textcoords="offset points",
        )
    ax.axhline(95.0, color="#777777", ls="--", lw=1, label="95% specificity")
    ax.set_xlabel("Efficacy (%)")
    ax.set_ylabel("Specificity (%)")
    ax.set_title("FUR Control Region and Protocol Faithfulness")
    ax.set_xlim(89, 100)
    ax.set_ylim(90, 100.5)
    ax.legend(loc="lower left", frameon=True, fontsize=8, markerscale=0.36)
    save_plot(fig, "eff_spec_faithfulness")


def plot_faithfulness(summary: dict[str, dict[str, float | int]]) -> None:
    metrics = ["ff_hard_agree", "ff_soft_agree"]
    labels = ["FF-HARD\n(agree)", "Max FF-SOFT\n(agree)"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for index, arm in enumerate(ARMS):
        values = [float(summary[arm][key]) for key in metrics]
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=ARMS[arm]["color"],
            label=ARMS[arm]["label"],
            alpha=0.9,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Faithfulness score (%)")
    ax.set_ylim(0, 82)
    ax.set_title("Protocol-Level Faithfulness Outcomes")
    ax.legend(frameon=True)
    save_plot(fig, "faithfulness_comparison")


def plot_trajectories(trajectory: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    for arm in ARMS:
        part = trajectory[trajectory["arm"] == arm].sort_values("epoch")
        axes[0].plot(
            part["epoch"],
            part["efficacy"],
            color=ARMS[arm]["color"],
            marker=ARMS[arm]["marker"],
            lw=2,
            label=ARMS[arm]["label"],
        )
        axes[1].plot(
            part["epoch"],
            part["specificity"],
            color=ARMS[arm]["color"],
            marker=ARMS[arm]["marker"],
            lw=2,
            label=ARMS[arm]["label"],
        )
    axes[0].set_title("Efficacy Trajectory")
    axes[0].set_ylabel("Efficacy (%)")
    axes[0].set_ylim(-3, 103)
    axes[1].set_title("Specificity Trajectory")
    axes[1].set_ylabel("Specificity (%)")
    axes[1].set_ylim(88, 101)
    axes[1].axhline(95.0, color="#777777", ls="--", lw=1)
    for ax in axes:
        ax.set_xlabel("Unlearning epoch")
        ax.set_xticks(range(0, 6))
    axes[0].legend(frameon=True, fontsize=8)
    save_plot(fig, "unlearning_trajectories")


def plot_cot_compactness(qdf: pd.DataFrame) -> None:
    display = qdf.copy()
    display["Model"] = display["arm"].map({arm: info["short"] for arm, info in ARMS.items()})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    palette = {"Base": ARMS["base"]["color"], "SFT": ARMS["sft"]["color"]}
    sns.boxplot(data=display, x="Model", y="generated_steps", hue="Model", palette=palette, legend=False, ax=axes[0])
    sns.boxplot(data=display, x="Model", y="cot_words", hue="Model", palette=palette, legend=False, ax=axes[1])
    axes[0].set_title("Segmented Reasoning Steps")
    axes[0].set_ylabel("Number of steps")
    axes[1].set_title("Chain-of-Thought Length")
    axes[1].set_ylabel("Number of words")
    for ax in axes:
        ax.set_xlabel("")
    save_plot(fig, "cot_compactness")


def plot_paired_soft(paired: pd.DataFrame) -> None:
    common = paired[
        (paired["agreement_base"] == 1) & (paired["agreement_sft"] == 1)
    ].copy()
    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    transitions = (
        common["hard_flip_base"].astype(str) + " to " + common["hard_flip_sft"].astype(str)
    )
    transition_palette = {
        "0 to 0": "#BDBDBD",
        "0 to 1": "#D7191C",
        "1 to 0": "#2C7BB6",
        "1 to 1": "#FDAE61",
    }
    for transition, frame in common.groupby(transitions):
        ax.scatter(
            frame["max_soft_shift_base"],
            frame["max_soft_shift_sft"],
            color=transition_palette.get(transition, "#777777"),
            label=transition.replace("0", "no flip").replace("1", "flip"),
            alpha=0.78,
            s=28,
        )
    bound = max(
        float(common["max_soft_shift_base"].max()),
        float(common["max_soft_shift_sft"].max()),
    )
    ax.plot([0, bound], [0, bound], ls="--", lw=1, color="#555555")
    ax.set_xlabel("Base max FF-SOFT (%)")
    ax.set_ylabel("SFT max FF-SOFT (%)")
    ax.set_title("Paired Common-Agreement Questions")
    ax.legend(frameon=True, fontsize=7, loc="upper left")
    save_plot(fig, "paired_soft_distribution")


def plot_teacher_filter(teacher_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [
        {
            "accepted": "Accepted",
            "wrong_teacher_answer": "Wrong answer",
            "invalid_json": "Invalid JSON",
            "answer_leak": "Answer leak",
            "sentence_count": "Sentence rule",
            "rationale_length": "Length rule",
        }.get(item, item)
        for item in teacher_df["status"]
    ]
    colors = ["#2C7BB6" if status == "accepted" else "#D9D9D9" for status in teacher_df["status"]]
    bars = ax.bar(labels, teacher_df["count"], color=colors)
    ax.bar_label(bars, labels=[str(value) for value in teacher_df["count"]], padding=2)
    ax.set_ylabel("Examples")
    ax.set_title("Teacher Revision Filtering Audit")
    ax.tick_params(axis="x", rotation=22)
    save_plot(fig, "teacher_filtering")


def write_analysis_markdown(
    summary: dict[str, dict[str, float | int]],
    paired_df: pd.DataFrame,
    paired_details: dict[str, Any],
    teacher_df: pd.DataFrame,
) -> None:
    base = summary["base"]
    sft = summary["sft"]
    hard = paired_df[paired_df["metric"] == "FF-HARD (common agree)"].iloc[0]
    soft = paired_df[paired_df["metric"] == "FF-SOFT (common agree)"].iloc[0]
    lines = [
        "# SFT 参数忠实性分析摘要",
        "",
        "本文件由 `analysis_pipeline.py` 从正式实验输出自动生成；详细方法、图表和讨论见 `report.tex`/`report.pdf`。",
        "",
        "## 关键结论",
        "",
        f"- SFT 后直接答案准确率从 {base['direct_accuracy']:.2f}% 提高到 {sft['direct_accuracy']:.2f}%，CoT 答案准确率从 {base['cot_accuracy']:.2f}% 提高到 {sft['cot_accuracy']:.2f}%。",
        f"- 推理被显著压缩：平均步骤从 {base['mean_steps']:.2f} 降到 {sft['mean_steps']:.2f}，平均词数从 {base['mean_words']:.2f} 降到 {sft['mean_words']:.2f}。",
        f"- FUR 控制指标更强：Efficacy 从 {base['efficacy']:.2f}% 升到 {sft['efficacy']:.2f}%，Specificity 从 {base['specificity']:.2f}% 升到 {sft['specificity']:.2f}%。",
        f"- 按各模型自己的 Direct-CoT 一致题计算，FF-HARD 从 {base['ff_hard_agree']:.2f}% 降到 {sft['ff_hard_agree']:.2f}%，FF-SOFT 从 {base['ff_soft_agree']:.2f}% 升到 {sft['ff_soft_agree']:.2f}%。",
        f"- 在双方都满足一致性的 {paired_details['common_agreement_questions']} 道严格配对题上，FF-HARD 变化为 {hard['delta']:+.2f} 百分点（95% bootstrap CI [{hard['ci_low']:.2f}, {hard['ci_high']:.2f}]），FF-SOFT 变化为 {soft['delta']:+.2f} 百分点（95% bootstrap CI [{soft['ci_low']:.2f}, {soft['ci_high']:.2f}]）。",
        "",
        "## Teacher 数据过滤",
        "",
        "| 状态 | 数量 | 占比 (%) |",
        "| --- | ---: | ---: |",
    ]
    for row in teacher_df.itertuples(index=False):
        lines.append(f"| {row.status} | {row.count} | {row.percentage:.2f} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "一次 SFT 的结果支持“蒸馏后的简洁 CoT 令模型更容易被 FUR 有效且局部地干预”，但不能仅依据本轮结果声称参数忠实性整体提升。原因是 FF-HARD 与 FF-SOFT 指向不同，且 SFT 与 FUR 都作用于 `down_proj`，需要额外控制实验区分真正的因果忠实性提升与干预可塑性提高。",
            "",
        ]
    )
    (HERE / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global ARTIFACTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    ARTIFACTS = args.artifacts.resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    summary: dict[str, dict[str, float | int]] = {}
    question_parts = []
    state_parts = []
    for arm in ARMS:
        arm_summary, qdf, sdf = analyze_arm(arm)
        summary[arm] = arm_summary
        question_parts.append(qdf)
        state_parts.append(sdf)
    questions = pd.concat(question_parts, ignore_index=True)
    states = pd.concat(state_parts, ignore_index=True)
    trajectory = trajectories(states)
    paired_df, paired_details, paired_questions = paired_statistics(questions)
    teacher_df, teacher_info = teacher_audit()

    write_json(DATA_DIR / "metrics.json", summary)
    write_json(DATA_DIR / "paired_details.json", paired_details)
    write_json(DATA_DIR / "teacher_info.json", teacher_info)
    metrics_rows = []
    for arm, values in summary.items():
        metrics_rows.append({"arm": arm, "model": ARMS[arm]["label"], **values})
    pd.DataFrame(metrics_rows).to_csv(DATA_DIR / "metrics.csv", index=False)
    questions.to_csv(DATA_DIR / "question_metrics.csv", index=False)
    states.to_csv(DATA_DIR / "fur_state_metrics.csv", index=False)
    trajectory.to_csv(DATA_DIR / "trajectory.csv", index=False)
    paired_df.to_csv(DATA_DIR / "paired_statistics.csv", index=False)
    paired_questions.to_csv(DATA_DIR / "paired_questions.csv", index=False)
    teacher_df.to_csv(DATA_DIR / "teacher_filter.csv", index=False)

    integrity_rows = build_integrity_table()
    write_table(
        "table_integrity",
        "正式实验产物完整性审计",
        "tab:integrity",
        ["产物", "预期行数", "观测行数", "检查"],
        integrity_rows,
        align="lrrc",
    )
    write_table(
        "table_training",
        "实验及训练设置",
        "tab:training",
        ["项目", "设置"],
        [
            ["基础模型", "Llama-3.2-3B-Instruct"],
            ["Teacher", "deepseek-v4-pro (thinking=high)"],
            ["训练数据", "OpenBookQA train; 2000 drafts; 1852 accepted revisions"],
            ["SFT 方法", "LoRA, target=down_proj, rank=32, alpha=64, dropout=0.05"],
            ["SFT 超参数", "lr=1e-4; epoch=1; batch=2; grad_acc=16; max_len=512"],
            ["FUR 测试集", "OpenBookQA test: 100 target + 20 retain; seed=1001"],
            ["FUR 超参数", "NPO+KL; lr=3e-5; epoch=5; beta=0.1; KL=1.0"],
            ["FUR 可训练参数", "mlp.down_proj.weight only"],
        ],
        align="lp{10.3cm}",
    )
    write_table(
        "table_teacher",
        "Teacher 修订数据质量过滤结果",
        "tab:teacher",
        ["过滤状态", "数量", "占比 (%)"],
        [
            [row.status, row.count, percent(row.percentage)]
            for row in teacher_df.itertuples(index=False)
        ],
        align="lrr",
    )
    write_table(
        "table_capability",
        "答案表现与 CoT 形态",
        "tab:capability",
        ["模型", "Direct Acc.", "CoT Acc.", "D/C Agree", "平均步骤", "平均词数"],
        [
            [
                ARMS[arm]["short"],
                percent(values["direct_accuracy"]),
                percent(values["cot_accuracy"]),
                percent(values["direct_cot_agreement"]),
                percent(values["mean_steps"]),
                percent(values["mean_words"]),
            ]
            for arm, values in summary.items()
        ],
        align="lrrrrr",
    )
    write_table(
        "table_controls",
        "FUR 有效性与特异性控制指标",
        "tab:controls",
        ["模型", "有效步骤", "Efficacy", "Efficacy@5", "Specificity", "Specificity@5"],
        [
            [
                ARMS[arm]["short"],
                values["measured_steps"],
                percent(values["efficacy"]),
                percent(values["efficacy_final"]),
                percent(values["specificity"]),
                percent(values["specificity_final"]),
            ]
            for arm, values in summary.items()
        ],
        align="lrrrrr",
    )
    write_table(
        "table_faithfulness",
        "FUR 参数忠实性主要结果",
        "tab:faithfulness",
        ["模型", "Agree N", "FF-HARD Agree", "FF-SOFT Agree", "FF-HARD All", "FF-SOFT All"],
        [
            [
                ARMS[arm]["short"],
                values["eligible_agreement_questions"],
                percent(values["ff_hard_agree"]),
                percent(values["ff_soft_agree"]),
                percent(values["ff_hard_all"]),
                percent(values["ff_soft_all"]),
            ]
            for arm, values in summary.items()
        ],
        align="lrrrrr",
    )
    delta_metrics = [
        ("Direct Acc.", "direct_accuracy"),
        ("CoT Acc.", "cot_accuracy"),
        ("D/C Agree", "direct_cot_agreement"),
        ("平均步骤", "mean_steps"),
        ("平均词数", "mean_words"),
        ("Efficacy", "efficacy"),
        ("Specificity", "specificity"),
        ("FF-HARD Agree", "ff_hard_agree"),
        ("FF-SOFT Agree", "ff_soft_agree"),
    ]
    write_table(
        "table_deltas",
        "SFT 相对于 Base 的指标变化",
        "tab:deltas",
        ["指标", "Base", "SFT", "SFT - Base"],
        [
            [
                label,
                percent(summary["base"][metric]),
                percent(summary["sft"][metric]),
                f"{float(summary['sft'][metric]) - float(summary['base'][metric]):+.2f}",
            ]
            for label, metric in delta_metrics
        ],
        align="lrrr",
    )
    paired_rows = []
    for row in paired_df.itertuples(index=False):
        test = (
            f"{row.improved}/{row.degraded}; p={row.p_value:.3f}"
            if not pd.isna(row.p_value)
            else "-"
        )
        paired_rows.append(
            [
                row.metric,
                row.n,
                percent(row.base),
                percent(row.sft),
                f"{row.delta:+.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]",
                test,
            ]
        )
    write_table(
        "table_paired",
        "严格配对比较及 95% bootstrap 置信区间",
        "tab:paired",
        ["指标", "N", "Base", "SFT", "差值 [95% CI]", "McNemar (+/-; p)"],
        paired_rows,
        align="lrrrrl",
    )

    plot_eff_spec(summary)
    plot_faithfulness(summary)
    plot_trajectories(trajectory)
    plot_cot_compactness(questions)
    plot_paired_soft(paired_questions)
    plot_teacher_filter(teacher_df)
    write_analysis_markdown(summary, paired_df, paired_details, teacher_df)

    output = {
        "metrics": summary,
        "paired_details": paired_details,
        "teacher_info": teacher_info,
        "generated_tables": sorted(path.name for path in TABLE_DIR.glob("*.tex")),
        "generated_figures": sorted(path.name for path in FIGURE_DIR.glob("*.pdf")),
    }
    write_json(DATA_DIR / "report_manifest.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
