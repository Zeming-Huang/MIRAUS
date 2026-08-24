"""Aggregate patient-level OOF metrics for the old formal offline LUPI experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "dice_3d",
    "iou_3d",
    "hd95_3d",
    "asd_3d",
    "nsd_3d",
    "dice_3d_gt_nonempty",
    "iou_3d_gt_nonempty",
    "hd95_3d_gt_nonempty",
    "asd_3d_gt_nonempty",
    "nsd_3d_gt_nonempty",
)


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def read_metrics(path, fold, branch):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fold"] = fold
        row["branch"] = branch
    return rows


def summarize_rows(rows):
    return {metric: stats([finite_float(row.get(metric)) for row in rows]) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work_dir/prism_offline_distillation_formal"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    student_rows = []
    teacher_rows = []
    counterfactual = []
    fold_summary = []
    for fold in range(5):
        fold_root = root / f"fold_{fold}"
        student_path = (
            fold_root
            / "student"
            / "heldout_oof_native08_full_image_v2"
            / "cv_test_metrics.csv"
        )
        teacher_path = (
            fold_root
            / "teacher"
            / "heldout_teacher_matched_native08_full_image_v2"
            / "cv_test_metrics.csv"
        )
        counterfactual_path = (
            fold_root
            / "teacher"
            / "counterfactual_full_best3d_v2"
            / "teacher_counterfactual_summary.json"
        )
        if not student_path.exists():
            raise FileNotFoundError(student_path)
        if not teacher_path.exists():
            raise FileNotFoundError(teacher_path)
        if not counterfactual_path.exists():
            raise FileNotFoundError(counterfactual_path)

        fold_student = read_metrics(student_path, fold, "student")
        fold_teacher = read_metrics(teacher_path, fold, "teacher")
        student_rows.extend(fold_student)
        teacher_rows.extend(fold_teacher)
        fold_summary.append(
            {
                "fold": fold,
                "num_cases": len(fold_student),
                "student": summarize_rows(fold_student),
                "teacher": summarize_rows(fold_teacher),
            }
        )
        with counterfactual_path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        item["fold"] = fold
        counterfactual.append(item)

    student_cases = [row["case_id"] for row in student_rows]
    if len(student_cases) != len(set(student_cases)):
        raise RuntimeError("Held-out OOF cases are not unique across folds.")

    cf_metrics = (
        "matched_dice_case_mean",
        "permuted_dice_case_mean",
        "zero_dice_case_mean",
        "permuted_drop_case_mean",
        "zero_drop_case_mean",
    )
    output = {
        "protocol": {
            "num_folds": 5,
            "split_seed": 2026,
            "prompt": "full_image",
            "student_input": "single_slice_trus",
            "teacher_privileged_input": "trus_plus_five_slice_mri",
            "spacing_mm": [0.8, 0.8, 0.8],
            "nsd_tolerance_mm": 1.6,
            "statistics": "patient-level macro mean and sample standard deviation",
        },
        "num_oof_cases": len(student_rows),
        "student_oof": summarize_rows(student_rows),
        "teacher_oof_matched": summarize_rows(teacher_rows),
        "counterfactual_fold_macro": {
            metric: stats([finite_float(item.get(metric)) for item in counterfactual])
            for metric in cf_metrics
        },
        "folds": fold_summary,
    }

    output_path = root / "offline_formal_5fold_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)

    rows_path = root / "offline_formal_5fold_oof_cases.csv"
    fieldnames = ["branch", "fold", "case_id"] + [
        key for key in student_rows[0] if key not in {"branch", "fold", "case_id"}
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(student_rows)
        writer.writerows(teacher_rows)

    print(json.dumps(output, indent=2, allow_nan=False))
    print(f"[OK] wrote {output_path}")
    print(f"[OK] wrote {rows_path}")


if __name__ == "__main__":
    main()
