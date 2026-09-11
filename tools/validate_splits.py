"""Validate the public patient-disjoint muRegPro fold manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SPLIT = Path(__file__).parents[1] / "configs" / "muregpro_5fold_splits.json"


def validate_split_manifest(path: Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    patients = payload.get("patients", [])
    folds = payload.get("folds", [])
    if len(folds) != int(payload.get("num_folds", -1)):
        raise ValueError("num_folds does not match the number of fold records")
    if len(patients) != len(set(patients)):
        raise ValueError("patients contains duplicate identifiers")

    held_out = []
    for expected_index, fold in enumerate(folds):
        if int(fold.get("fold", -1)) != expected_index:
            raise ValueError("fold indices must be consecutive and zero-based")
        test_cases = fold.get("test", [])
        validation_cases = fold.get("validation", [])
        if len(test_cases) != len(set(test_cases)):
            raise ValueError(f"fold {expected_index} contains duplicate test cases")
        if len(validation_cases) != len(set(validation_cases)):
            raise ValueError(f"fold {expected_index} contains duplicate validation cases")
        if set(test_cases) & set(validation_cases):
            raise ValueError(f"fold {expected_index} validation and test sets overlap")
        if not set(test_cases).issubset(patients) or not set(validation_cases).issubset(patients):
            raise ValueError(f"fold {expected_index} contains an unknown patient")
        held_out.extend(test_cases)

    if len(held_out) != len(set(held_out)):
        raise ValueError("held-out patient lists overlap across folds")
    if set(held_out) != set(patients):
        raise ValueError("held-out folds do not cover the declared patient list")
    return {"patients": len(patients), "folds": len(folds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLIT)
    args = parser.parse_args()
    summary = validate_split_manifest(args.splits)
    print(f"Validated {summary['patients']} patients across {summary['folds']} folds")


if __name__ == "__main__":
    main()
