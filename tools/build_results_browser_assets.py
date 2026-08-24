"""Build anonymized static assets for the public segmentation result browser."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "docs" / "results-browser"
ASSET_ROOT = PUBLIC_ROOT / "assets" / "samples"
PRIVATE_MANIFEST = ROOT / "results" / "results_browser_source_manifest.json"
MIRAUS_PATTERN = (
    "work_dir/prism_offline_distillation_formal/fold_*/student/"
    "heldout_oof_predictions_native08_full_image/predictions/case*.npz"
)
BASELINE_PATTERN = (
    "work_dir/prism_a0_strict_5fold_fixed60/fold_*/student/"
    "heldout_oof_native08_full_image_v2/predictions/case*.npz"
)
DEFAULT_SOURCE_ROOTS = (
    ROOT / "data" / "muregpro" / "train" / "us_images",
    ROOT / "data" / "muregpro" / "val" / "us_images",
)
REGIONS = ("apex", "mid-gland", "base")
DISPLAY_SIZE = (486, 708)

COLORS = {
    "ground_truth": (83, 153, 104),
    "miraus": (214, 79, 73),
    "baseline": (73, 118, 171),
    "probability_gain": (216, 42, 48),
    "probability_suppression": (43, 103, 178),
}


def index_predictions(pattern: str) -> dict[str, Path]:
    paths = sorted(ROOT.glob(pattern))
    indexed = {path.stem: path for path in paths}
    if len(indexed) != 73:
        raise RuntimeError(f"Expected 73 OOF predictions for {pattern}, found {len(indexed)}")
    return indexed


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(prediction.sum() + target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    union = float(np.logical_or(prediction, target).sum())
    return 1.0 if union == 0 else float(np.logical_and(prediction, target).sum()) / union


def hd95_mm(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float]) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(np.hypot(*(np.asarray(prediction.shape) * np.asarray(spacing))))

    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((target_distance[pred_surface], pred_distance[target_surface]))
    return float(np.percentile(distances, 95))


def contour(mask: np.ndarray, width: int = 1) -> np.ndarray:
    edge = mask.astype(bool) ^ binary_erosion(mask.astype(bool))
    expanded = edge.copy()
    for _ in range(max(0, width - 1)):
        neighbors = np.zeros_like(expanded)
        neighbors[1:] |= expanded[:-1]
        neighbors[:-1] |= expanded[1:]
        neighbors[:, 1:] |= expanded[:, :-1]
        neighbors[:, :-1] |= expanded[:, 1:]
        expanded |= neighbors
    return expanded


def rgba_layer(
    mask: np.ndarray,
    color: tuple[int, int, int],
    fill_alpha: int = 78,
    contour_alpha: int = 255,
    contour_width: int = 1,
) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = color
    rgba[mask, 3] = fill_alpha
    edge = contour(mask, contour_width)
    rgba[edge, :3] = color
    rgba[edge, 3] = contour_alpha
    return rgba


def contour_comparison(gt: np.ndarray, miraus: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*gt.shape, 4), dtype=np.uint8)
    for mask, key in ((baseline, "baseline"), (gt, "ground_truth"), (miraus, "miraus")):
        edge = contour(mask, 1)
        rgba[edge, :3] = COLORS[key]
        rgba[edge, 3] = 255
    return rgba


def signed_probability_layer(
    miraus_probability: np.ndarray,
    baseline_probability: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Render the signed probability difference used in the paper figure."""
    difference = miraus_probability.astype(np.float32) - baseline_probability.astype(np.float32)
    difference = np.clip(difference, -1.0, 1.0)
    rgba = colormaps["RdBu_r"]((difference + 1.0) / 2.0)
    rgba[..., 3] = 0.88 * np.clip(np.abs(difference) / 0.55, 0.0, 1.0) ** 0.72
    rgba = np.round(rgba * 255.0).astype(np.uint8)

    reference_edge = contour(target, 1)
    rgba[reference_edge, :3] = (255, 255, 255)
    rgba[reference_edge, 3] = 255
    return rgba


def resize_and_save(array: np.ndarray, path: Path, *, is_overlay: bool = False) -> None:
    image = Image.fromarray(array, mode="RGBA" if is_overlay else "L")
    image = image.resize(DISPLAY_SIZE, Image.Resampling.NEAREST if is_overlay else Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_overlay:
        image.save(path, optimize=True)
    else:
        image.save(path, format="WEBP", quality=90, method=6)


def resize_difference_and_save(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(array, mode="RGBA")
    image = image.resize(DISPLAY_SIZE, Image.Resampling.BILINEAR)
    image.save(path, optimize=True)


def source_image_path(case_id: str, source_roots: tuple[Path, ...]) -> Path:
    for root in source_roots:
        candidate = root / f"{case_id}.nii.gz"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No source TRUS image found for {case_id}")


def normalized_volume(path: Path) -> np.ndarray:
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    finite = volume[np.isfinite(volume)]
    low, high = np.percentile(finite, (1.0, 99.5))
    volume = np.clip((volume - low) / max(high - low, 1e-6), 0.0, 1.0)
    return np.round(volume * 255.0).astype(np.uint8)


def select_cases(miraus_paths: dict[str, Path], baseline_paths: dict[str, Path]) -> list[str]:
    scored = []
    for case_id, path in miraus_paths.items():
        with np.load(path) as archive:
            target = archive["ground_truth"].astype(bool)
            miraus = archive["prediction"].astype(bool)
        with np.load(baseline_paths[case_id]) as archive:
            baseline = archive["prediction"].astype(bool)

        slices = selected_slices(target)
        differences = [
            dice_score(miraus[index], target[index]) - dice_score(baseline[index], target[index])
            for index in slices.values()
        ]
        scored.append((float(np.mean(differences)), sum(value > 0 for value in differences), case_id))

    ranked = sorted(scored, reverse=True)
    successes = [record for record in ranked if record[1] >= 2]
    chosen = successes[:9]

    median = float(np.median([item[0] for item in scored]))
    representative = sorted(scored, key=lambda record: abs(record[0] - median))
    failures = sorted(scored, key=lambda record: record[0])
    for pool, count in ((representative, 4), (failures, 2)):
        added = 0
        for record in pool:
            if record[2] in {item[2] for item in chosen}:
                continue
            chosen.append(record)
            added += 1
            if added == count:
                break

    for record in ranked:
        if len(chosen) >= 15:
            break
        if record[2] not in {item[2] for item in chosen}:
            chosen.append(record)
    return [record[2] for record in chosen[:15]]


def selected_slices(ground_truth: np.ndarray) -> dict[str, int]:
    positive = np.flatnonzero(ground_truth.reshape(ground_truth.shape[0], -1).any(axis=1))
    if len(positive) < 3:
        raise RuntimeError("A public example must contain at least three positive slices")
    groups = np.array_split(positive, 3)
    return {region: int(group[len(group) // 2]) for region, group in zip(REGIONS, groups)}


def metric_record(prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float]) -> dict:
    return {
        "dice": round(dice_score(prediction, target) * 100.0, 2),
        "iou": round(iou_score(prediction, target) * 100.0, 2),
        "hd95": round(hd95_mm(prediction, target, spacing), 2),
    }


def build(source_roots: tuple[Path, ...]) -> None:
    miraus_paths = index_predictions(MIRAUS_PATTERN)
    baseline_paths = index_predictions(BASELINE_PATTERN)
    case_ids = select_cases(miraus_paths, baseline_paths)
    if ASSET_ROOT.exists():
        shutil.rmtree(ASSET_ROOT)
    public_samples = []
    private_samples = []

    for sample_index, case_id in enumerate(case_ids, start=1):
        sample_id = f"sample-{sample_index:02d}"
        with np.load(miraus_paths[case_id]) as archive:
            gt = archive["ground_truth"].astype(bool)
            miraus = archive["prediction"].astype(bool)
            miraus_probability = archive["probability"].astype(np.float32)
            spacing = tuple(float(v) for v in archive["spacing_mm"][1:])
        with np.load(baseline_paths[case_id]) as archive:
            baseline = archive["prediction"].astype(bool)
            baseline_probability = archive["probability"].astype(np.float32)
        image_path = source_image_path(case_id, source_roots)
        image_volume = normalized_volume(image_path)
        if image_volume.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch for {case_id}: image {image_volume.shape}, GT {gt.shape}")

        region_slices = selected_slices(gt)
        positive_slices = np.flatnonzero(gt.reshape(gt.shape[0], -1).any(axis=1))
        first_positive = int(positive_slices[0])
        last_positive = int(positive_slices[-1])
        region_records = {}
        for region, slice_index in region_slices.items():
            relative = 0.0 if last_positive == first_positive else (
                100.0 * (slice_index - first_positive) / (last_positive - first_positive)
            )
            output = ASSET_ROOT / sample_id / region
            resize_and_save(image_volume[slice_index], output / "original.webp")
            resize_and_save(
                rgba_layer(gt[slice_index], COLORS["ground_truth"]),
                output / "ground-truth.png",
                is_overlay=True,
            )
            resize_and_save(
                rgba_layer(miraus[slice_index], COLORS["miraus"]),
                output / "miraus.png",
                is_overlay=True,
            )
            resize_and_save(
                rgba_layer(baseline[slice_index], COLORS["baseline"]),
                output / "baseline.png",
                is_overlay=True,
            )
            resize_and_save(
                contour_comparison(gt[slice_index], miraus[slice_index], baseline[slice_index]),
                output / "contours.png",
                is_overlay=True,
            )
            resize_difference_and_save(
                signed_probability_layer(
                    miraus_probability[slice_index],
                    baseline_probability[slice_index],
                    gt[slice_index],
                ),
                output / "miraus-error.png",
            )
            region_records[region] = {
                "relative_depth": round(relative),
                "miraus": metric_record(miraus[slice_index], gt[slice_index], spacing),
                "baseline": metric_record(baseline[slice_index], gt[slice_index], spacing),
            }

        public_samples.append(
            {
                "id": sample_id,
                "label": f"Sample {sample_index:02d}",
                "regions": region_records,
            }
        )
        private_samples.append(
            {
                "sample_id": sample_id,
                "case_id": case_id,
                "miraus_prediction": str(miraus_paths[case_id]),
                "baseline_prediction": str(baseline_paths[case_id]),
                "source_image": str(image_path),
                "selected_slices": region_slices,
            }
        )

    manifest = {
        "version": 1,
        "evaluation": "Patient-disjoint five-fold out-of-fold predictions",
        "spacing_mm": 0.8,
        "samples": public_samples,
    }
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_ROOT / "samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    PRIVATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_MANIFEST.write_text(json.dumps(private_samples, indent=2), encoding="utf-8")
    print(f"Generated {len(public_samples)} anonymized samples in {PUBLIC_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        dest="source_roots",
        help="Directory containing case*.nii.gz TRUS volumes; may be repeated.",
    )
    arguments = parser.parse_args()
    build(tuple(arguments.source_roots or DEFAULT_SOURCE_ROOTS))
