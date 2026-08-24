#!/usr/bin/env python
"""Patient-level uncertainty and calibration analysis for formal 5-fold OOF results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = {
    "dice_3d": ("Dice", "higher"),
    "iou_3d": ("IoU", "higher"),
    "hd95_3d": ("HD95 (mm)", "lower"),
    "asd_3d": ("ASD (mm)", "lower"),
    "nsd_3d": ("NSD", "higher"),
    "dice_3d_gt_nonempty": ("Dice (GT nonempty)", "higher"),
    "iou_3d_gt_nonempty": ("IoU (GT nonempty)", "higher"),
    "hd95_3d_gt_nonempty": ("HD95 (GT nonempty, mm)", "lower"),
    "asd_3d_gt_nonempty": ("ASD (GT nonempty, mm)", "lower"),
    "nsd_3d_gt_nonempty": ("NSD (GT nonempty)", "higher"),
}

CALIBRATION_METRICS = {
    "brier": ("Brier", "lower"),
    "balanced_brier": ("Balanced Brier", "lower"),
    "nll": ("NLL", "lower"),
    "balanced_nll": ("Balanced NLL", "lower"),
    "ece": ("ECE", "lower"),
}


def percentile_bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, iterations: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    p_values_array = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values_array)
    adjusted = np.empty_like(p_values_array)
    running_max = 0.0
    count = len(p_values_array)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values_array[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def summarize_branch(
    frame: pd.DataFrame,
    metrics: dict[str, tuple[str, str]],
    rng: np.random.Generator,
    iterations: int,
) -> list[dict]:
    rows = []
    for column, (label, direction) in metrics.items():
        values = frame[column].to_numpy(dtype=np.float64)
        low, high = percentile_bootstrap_mean(values, rng, iterations)
        rows.append(
            {
                "metric": column,
                "label": label,
                "direction": direction,
                "n": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)),
                "sample_sd": float(np.nanstd(values, ddof=1)),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
            }
        )
    return rows


def paired_comparison(
    paired: pd.DataFrame,
    metrics: dict[str, tuple[str, str]],
    rng: np.random.Generator,
    iterations: int,
) -> list[dict]:
    rows = []
    raw_p_values = []
    for column, (label, direction) in metrics.items():
        student = paired[f"{column}_student"].to_numpy(dtype=np.float64)
        teacher = paired[f"{column}_teacher"].to_numpy(dtype=np.float64)
        valid = np.isfinite(student) & np.isfinite(teacher)
        student = student[valid]
        teacher = teacher[valid]

        # Positive improvement always favors the student.
        improvement = student - teacher if direction == "higher" else teacher - student
        low, high = percentile_bootstrap_mean(improvement, rng, iterations)
        if np.allclose(improvement, 0):
            p_value = 1.0
        else:
            p_value = float(
                stats.wilcoxon(
                    improvement,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        raw_p_values.append(p_value)
        rows.append(
            {
                "metric": column,
                "label": label,
                "direction": direction,
                "n": int(len(improvement)),
                "student_mean": float(student.mean()),
                "teacher_mean": float(teacher.mean()),
                "student_improvement_mean": float(improvement.mean()),
                "improvement_bootstrap_95ci_low": low,
                "improvement_bootstrap_95ci_high": high,
                "wilcoxon_p_raw": p_value,
            }
        )

    for row, adjusted in zip(rows, holm_adjust(raw_p_values)):
        row["wilcoxon_p_holm"] = float(adjusted)
        row["significant_after_holm_0.05"] = bool(adjusted < 0.05)
    return rows


def counterfactual_fold_analysis(root: Path) -> list[dict]:
    fields = [
        "matched_dice_case_mean",
        "permuted_dice_case_mean",
        "zero_dice_case_mean",
        "permuted_drop_case_mean",
        "zero_drop_case_mean",
    ]
    fold_rows = []
    for fold in range(5):
        path = (
            root
            / f"fold_{fold}"
            / "teacher"
            / "counterfactual_full_best3d_v2"
            / "teacher_counterfactual_summary.json"
        )
        summary = json.loads(path.read_text(encoding="utf-8"))
        fold_rows.append({"fold": fold, **{field: summary[field] for field in fields}})

    output = []
    critical = stats.t.ppf(0.975, df=len(fold_rows) - 1)
    for field in fields:
        values = np.asarray([row[field] for row in fold_rows], dtype=np.float64)
        mean = values.mean()
        sd = values.std(ddof=1)
        half_width = critical * sd / np.sqrt(len(values))
        output.append(
            {
                "metric": field,
                "n_folds": len(values),
                "mean": float(mean),
                "sample_sd": float(sd),
                "t_95ci_low": float(mean - half_width),
                "t_95ci_high": float(mean + half_width),
                "fold_values": values.tolist(),
            }
        )
    return output


def format_interval(row: dict, low_key: str, high_key: str) -> str:
    return f"{row['mean']:.4f} [{row[low_key]:.4f}, {row[high_key]:.4f}]"


def write_markdown(result: dict, output_path: Path) -> None:
    student = result["student_performance"]
    calibration = result["student_calibration"]
    paired = result["student_vs_teacher"]
    counterfactual = result["teacher_counterfactual_fold_level"]

    lines = [
        "# Offline formal 5-fold confidence analysis",
        "",
        f"- OOF patients: {result['protocol']['num_oof_patients']}",
        f"- Bootstrap iterations: {result['protocol']['bootstrap_iterations']}",
        "- Intervals: patient-level percentile bootstrap unless stated otherwise",
        "",
        "## Student performance",
        "",
        "| Metric | Mean [95% CI] | SD |",
        "|---|---:|---:|",
    ]
    for row in student:
        interval = format_interval(row, "bootstrap_95ci_low", "bootstrap_95ci_high")
        lines.append(f"| {row['label']} | {interval} | {row['sample_sd']:.4f} |")

    lines.extend(
        [
            "",
            "## Student probability calibration",
            "",
            "| Metric | Mean [95% CI] | SD |",
            "|---|---:|---:|",
        ]
    )
    for row in calibration:
        interval = format_interval(row, "bootstrap_95ci_low", "bootstrap_95ci_high")
        lines.append(f"| {row['label']} | {interval} | {row['sample_sd']:.4f} |")

    lines.extend(
        [
            "",
            "## Paired student versus teacher",
            "",
            "Positive improvement favors the student. Holm correction covers all 10 segmentation metrics.",
            "",
            "| Metric | Student | Teacher | Improvement [95% CI] | Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        interval = (
            f"{row['student_improvement_mean']:.4f} "
            f"[{row['improvement_bootstrap_95ci_low']:.4f}, "
            f"{row['improvement_bootstrap_95ci_high']:.4f}]"
        )
        lines.append(
            f"| {row['label']} | {row['student_mean']:.4f} | "
            f"{row['teacher_mean']:.4f} | {interval} | {row['wilcoxon_p_holm']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Teacher counterfactual sensitivity",
            "",
            "These are t intervals across five fold-level estimates, not patient-population intervals.",
            "",
            "| Metric | Fold mean [95% CI] | SD |",
            "|---|---:|---:|",
        ]
    )
    for row in counterfactual:
        interval = format_interval(row, "t_95ci_low", "t_95ci_high")
        lines.append(f"| {row['metric']} | {interval} | {row['sample_sd']:.4f} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work_dir/prism_offline_distillation_formal"),
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    oof_path = args.root / "offline_formal_5fold_oof_cases.csv"
    frame = pd.read_csv(oof_path)
    student = frame.loc[frame["branch"] == "student"].copy()
    teacher = frame.loc[frame["branch"] == "teacher"].copy()
    keys = ["fold", "case_id"]
    if len(student) != 73 or len(teacher) != 73:
        raise ValueError(f"Expected 73 rows per branch, got {len(student)} and {len(teacher)}")
    if student[keys].duplicated().any() or teacher[keys].duplicated().any():
        raise ValueError("Duplicate fold/case pairs found")

    paired = student.merge(teacher, on=keys, suffixes=("_student", "_teacher"), validate="one_to_one")
    if len(paired) != 73:
        raise ValueError(f"Expected 73 paired patients, got {len(paired)}")

    rng = np.random.default_rng(args.seed)
    result = {
        "protocol": {
            "num_oof_patients": len(paired),
            "bootstrap_iterations": args.iterations,
            "bootstrap_seed": args.seed,
            "confidence_level": 0.95,
            "bootstrap_method": "patient-level nonparametric percentile bootstrap",
            "paired_test": "two-sided Wilcoxon signed-rank with Holm correction",
            "counterfactual_interval": "two-sided t interval over five fold-level means",
        },
        "student_performance": summarize_branch(
            student, PRIMARY_METRICS, rng, args.iterations
        ),
        "student_calibration": summarize_branch(
            student, CALIBRATION_METRICS, rng, args.iterations
        ),
        "student_vs_teacher": paired_comparison(
            paired, PRIMARY_METRICS, rng, args.iterations
        ),
        "teacher_counterfactual_fold_level": counterfactual_fold_analysis(args.root),
    }

    json_path = args.root / "offline_formal_5fold_confidence_analysis.json"
    markdown_path = args.root / "offline_formal_5fold_confidence_analysis.md"
    comparison_path = args.root / "offline_formal_5fold_paired_comparison.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["student_vs_teacher"]).to_csv(comparison_path, index=False)
    write_markdown(result, markdown_path)
    print(json.dumps(result, indent=2))
    print(f"[OK] wrote {json_path.resolve()}")
    print(f"[OK] wrote {markdown_path.resolve()}")
    print(f"[OK] wrote {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
