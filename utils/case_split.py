import os
import random
import re
from glob import glob
from os.path import basename, join, normpath


def split_root_spec(root_spec):
    """Return one or more dataset roots from a semicolon-separated root spec."""
    if root_spec is None:
        return []
    if isinstance(root_spec, (list, tuple)):
        raw_roots = root_spec
    else:
        raw_roots = str(root_spec).split(";")
    roots = []
    for root in raw_roots:
        root = str(root).strip().strip('"')
        if not root:
            continue
        if not os.path.isabs(root):
            root = os.path.abspath(root)
        roots.append(normpath(root))
    return roots


def extract_case_id(path_or_name):
    match = re.search(r"(case\d+)", basename(str(path_or_name)))
    return match.group(1) if match else None


def normalize_case_ids(case_ids):
    if case_ids is None:
        return None
    if isinstance(case_ids, str):
        raw = re.split(r"[,;\s]+", case_ids.strip())
    else:
        raw = case_ids
    normalized = {str(case_id).strip() for case_id in raw if str(case_id).strip()}
    return normalized or None


def root_spec_exists(root_spec):
    roots = split_root_spec(root_spec)
    return bool(roots) and all(os.path.isdir(root) for root in roots)


def list_case_ids_from_roots(root_spec):
    case_ids = set()
    for root in split_root_spec(root_spec):
        for pattern in (join(root, "gts", "*.npy"), join(root, "*.npz")):
            for path in glob(pattern):
                case_id = extract_case_id(path)
                if case_id:
                    case_ids.add(case_id)
    return sorted(case_ids)


def make_kfold_case_split(case_ids, num_folds, fold, seed=2026, inner_val_fraction=0.1):
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}], got {fold}")

    shuffled = sorted(normalize_case_ids(case_ids) or [])
    if len(shuffled) < num_folds:
        raise ValueError(
            f"Need at least {num_folds} cases for {num_folds}-fold CV, got {len(shuffled)}"
        )
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    fold_sizes = [len(shuffled) // num_folds] * num_folds
    for idx in range(len(shuffled) % num_folds):
        fold_sizes[idx] += 1
    start = sum(fold_sizes[:fold])
    stop = start + fold_sizes[fold]

    test_cases = sorted(shuffled[start:stop])
    remaining = [case_id for case_id in shuffled if case_id not in set(test_cases)]

    val_count = 0
    if inner_val_fraction > 0:
        val_count = max(1, int(round(len(remaining) * inner_val_fraction)))
        val_count = min(val_count, max(len(remaining) - 1, 1))
    val_rng = random.Random(seed + 1009 * (fold + 1))
    val_cases = sorted(val_rng.sample(remaining, val_count)) if val_count else []
    train_cases = sorted(case_id for case_id in remaining if case_id not in set(val_cases))

    return {
        "num_folds": num_folds,
        "fold": fold,
        "seed": seed,
        "inner_val_fraction": inner_val_fraction,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
    }
