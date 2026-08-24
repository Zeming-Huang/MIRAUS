#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive segmentation evaluation.

Metrics:
- 3D Dice / IoU on the whole volume
- 2D Dice / IoU / Recall / Specificity / Precision averaged over non-empty slices
- HD95 averaged over non-empty slices, reported in both px and mm
- Volume error metrics using voxel spacing
"""

import argparse
import os
from glob import glob
from os.path import basename, join

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def compute_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_sensitivity(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def compute_specificity(pred, gt):
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))
    if tn + fp == 0:
        return 1.0
    return tn / (tn + fp)


def compute_precision(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def normalize_spacing(spacing):
    if isinstance(spacing, np.ndarray):
        spacing = spacing.flatten()
    else:
        spacing = np.array([spacing] if np.isscalar(spacing) else spacing)

    if len(spacing) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if len(spacing) == 1:
        return np.array([spacing[0], spacing[0], 1.0], dtype=np.float32)
    if len(spacing) == 2:
        return np.array([spacing[0], spacing[1], 1.0], dtype=np.float32)
    return spacing.astype(np.float32)


def get_inplane_spacing_hw(spacing):
    """
    Return the 2D slice spacing in array axis order (H, W).

    For the current dataset all spacing values are equal (0.8, 0.8, 0.8),
    so this mapping is stable. We keep the function explicit so the unit
    handling stays visible and auditable.
    """
    spacing = normalize_spacing(spacing)
    return np.array([float(spacing[0]), float(spacing[1])], dtype=np.float32)


def compute_hausdorff_distance(pred, gt, percentile=95, spacing_hw=None):
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    pred_points = pred_points.astype(np.float32)
    gt_points = gt_points.astype(np.float32)

    if spacing_hw is not None:
        spacing_hw = np.asarray(spacing_hw, dtype=np.float32)
        pred_points = pred_points * spacing_hw
        gt_points = gt_points * spacing_hw

    distances_pred_to_gt = []
    for p in pred_points:
        dists = np.linalg.norm(gt_points - p, axis=-1)
        distances_pred_to_gt.append(np.min(dists))

    distances_gt_to_pred = []
    for g in gt_points:
        dists = np.linalg.norm(pred_points - g, axis=-1)
        distances_gt_to_pred.append(np.min(dists))

    all_distances = distances_pred_to_gt + distances_gt_to_pred
    if len(all_distances) == 0:
        return float("inf")

    if percentile == 100:
        return float(np.max(all_distances))
    return float(np.percentile(all_distances, percentile))


def mask_to_surface(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded


def compute_surface_distance_metrics(pred, gt, spacing_hw=None):
    """
    Compute symmetric surface-based distances on a 2D slice.

    Returns
    -------
    tuple[float, float]
        (hd95, asd)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf"), float("inf")

    pred_surface = mask_to_surface(pred)
    gt_surface = mask_to_surface(gt)
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("inf"), float("inf")

    sampling = None if spacing_hw is None else np.asarray(spacing_hw, dtype=np.float32)
    dt_gt = distance_transform_edt(~gt_surface, sampling=sampling)
    dt_pred = distance_transform_edt(~pred_surface, sampling=sampling)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    if all_distances.size == 0:
        return float("inf"), float("inf")

    hd95 = float(np.percentile(all_distances, 95))
    asd = float(np.mean(all_distances))
    return hd95, asd


def evaluate_predictions(pred_dir, gt_dir):
    results = []
    pred_files = sorted(glob(join(pred_dir, "**", "*.npz"), recursive=True))
    print(f"找到 {len(pred_files)} 个预测文件")

    for pred_file in tqdm(pred_files, desc="评估中"):
        pred_data = np.load(pred_file, allow_pickle=True)
        pred_segs = pred_data["segs"]
        spacing = normalize_spacing(pred_data["spacing"]) if "spacing" in pred_data else normalize_spacing([1.0, 1.0, 1.0])
        spacing_hw = get_inplane_spacing_hw(spacing)

        file_name = basename(pred_file)
        gt_file = join(gt_dir, file_name)
        if not os.path.exists(gt_file):
            print(f"警告: 找不到对应真值文件 {gt_file}")
            continue

        gt_data = np.load(gt_file, allow_pickle=True)
        gt_segs = gt_data["gts"]

        if pred_segs.shape != gt_segs.shape:
            print(f"警告: 形状不匹配 {pred_segs.shape} vs {gt_segs.shape} for {file_name}")
            continue

        slice_metrics = {
            "dice": [],
            "iou": [],
            "sensitivity": [],
            "specificity": [],
            "precision": [],
        }

        for i in range(pred_segs.shape[0]):
            pred_binary = (pred_segs[i] > 0).astype(np.float32)
            gt_binary = (gt_segs[i] > 0).astype(np.float32)

            is_empty_slice = np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0
            if is_empty_slice:
                continue

            slice_metrics["dice"].append(compute_dice_score(pred_binary, gt_binary))
            slice_metrics["iou"].append(compute_iou(pred_binary, gt_binary))
            slice_metrics["sensitivity"].append(compute_sensitivity(pred_binary, gt_binary))
            slice_metrics["specificity"].append(compute_specificity(pred_binary, gt_binary))
            slice_metrics["precision"].append(compute_precision(pred_binary, gt_binary))


        pred_3d_binary = (pred_segs > 0).astype(np.float32)
        gt_3d_binary = (gt_segs > 0).astype(np.float32)

        dice_3d = compute_dice_score(pred_3d_binary, gt_3d_binary)
        iou_3d = compute_iou(pred_3d_binary, gt_3d_binary)

        # 3D surface distances: operate on the full volume, not per-slice averages.
        # Array shape is (Num_slices, H, W); sampling order must match: (slice, H, W).
        # spacing = [sx, sy, sz] where sx/sy = in-plane, sz = slice thickness.
        spacing_3d_ordered = np.array(
            [float(spacing[2]), float(spacing[0]), float(spacing[1])], dtype=np.float32
        )
        hd95_3d_px, asd_3d_px = compute_surface_distance_metrics(
            pred_3d_binary.astype(bool), gt_3d_binary.astype(bool), spacing_hw=None
        )
        hd95_3d_mm, asd_3d_mm = compute_surface_distance_metrics(
            pred_3d_binary.astype(bool), gt_3d_binary.astype(bool), spacing_hw=spacing_3d_ordered
        )
        # inf means one side is empty (total miss or empty GT); report as NaN
        hd95_3d_px = float("nan") if hd95_3d_px == float("inf") else hd95_3d_px
        hd95_3d_mm = float("nan") if hd95_3d_mm == float("inf") else hd95_3d_mm
        asd_3d_px  = float("nan") if asd_3d_px  == float("inf") else asd_3d_px
        asd_3d_mm  = float("nan") if asd_3d_mm  == float("inf") else asd_3d_mm

        voxel_volume = float(np.prod(spacing[:3]))
        volume_pred = float(np.sum(pred_3d_binary) * voxel_volume)
        volume_gt = float(np.sum(gt_3d_binary) * voxel_volume)

        if volume_gt == 0:
            volume_error = 0.0 if volume_pred == 0 else float("nan")
            rvd = 0.0 if volume_pred == 0 else float("nan")
        else:
            volume_error = abs(volume_pred - volume_gt) / volume_gt * 100.0
            rvd = (volume_pred - volume_gt) / volume_gt * 100.0

        if len(slice_metrics["dice"]) > 0:
            dice_2d_mean = float(np.mean(slice_metrics["dice"]))
            dice_2d_std = float(np.std(slice_metrics["dice"]))
            iou_2d_mean = float(np.mean(slice_metrics["iou"]))
            iou_2d_std = float(np.std(slice_metrics["iou"]))
            sensitivity_2d_mean = float(np.mean(slice_metrics["sensitivity"]))
            sensitivity_2d_std = float(np.std(slice_metrics["sensitivity"]))
            specificity_2d_mean = float(np.mean(slice_metrics["specificity"]))
            specificity_2d_std = float(np.std(slice_metrics["specificity"]))
            precision_2d_mean = float(np.mean(slice_metrics["precision"]))
            precision_2d_std = float(np.std(slice_metrics["precision"]))
        else:
            dice_2d_mean = float("nan")
            dice_2d_std = float("nan")
            iou_2d_mean = float("nan")
            iou_2d_std = float("nan")
            sensitivity_2d_mean = float("nan")
            sensitivity_2d_std = float("nan")
            specificity_2d_mean = float("nan")
            specificity_2d_std = float("nan")
            precision_2d_mean = float("nan")
            precision_2d_std = float("nan")

        results.append(
            {
                "case": file_name,
                "Dice_3D_Overall": dice_3d,
                "IoU_3D_Overall": iou_3d,
                "Dice_2D_Mean": dice_2d_mean,
                "Dice_2D_Std": dice_2d_std,
                "IoU_2D_Mean": iou_2d_mean,
                "IoU_2D_Std": iou_2d_std,
                "Sensitivity_2D_Mean": sensitivity_2d_mean,
                "Sensitivity_2D_Std": sensitivity_2d_std,
                "Specificity_2D_Mean": specificity_2d_mean,
                "Specificity_2D_Std": specificity_2d_std,
                "Precision_2D_Mean": precision_2d_mean,
                "Precision_2D_Std": precision_2d_std,
                "HD95_3D_px": hd95_3d_px,
                "HD95_3D_mm": hd95_3d_mm,
                "ASD_3D_px": asd_3d_px,
                "ASD_3D_mm": asd_3d_mm,
                "Volume_Error_%": volume_error,
                "Relative_Volume_Diff_%": rvd,
                "Volume_Pred_mm3": volume_pred,
                "Volume_GT_mm3": volume_gt,
                "Num_Slices": pred_segs.shape[0],
                "Num_NonEmpty_Slices": len(slice_metrics["dice"]),
                "Spacing": f"{spacing[0]:.4f},{spacing[1]:.4f},{spacing[2]:.4f}",
            }
        )

    return results


def print_summary(df):
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(f"病例数: {len(df)}")
    print(f"总切片数: {df['Num_Slices'].sum()}")
    print(f"非空切片数: {df['Num_NonEmpty_Slices'].sum()}")

    print("\n3D 指标")
    print(f"Dice (3D): {df['Dice_3D_Overall'].mean():.4f} ± {df['Dice_3D_Overall'].std():.4f}")
    print(f"IoU  (3D): {df['IoU_3D_Overall'].mean():.4f} ± {df['IoU_3D_Overall'].std():.4f}")

    print("\n2D 指标（非空切片平均）")
    print(f"Dice:      {df['Dice_2D_Mean'].mean():.4f} ± {df['Dice_2D_Mean'].std():.4f}")
    print(f"IoU:       {df['IoU_2D_Mean'].mean():.4f} ± {df['IoU_2D_Mean'].std():.4f}")
    print(f"Recall:    {df['Sensitivity_2D_Mean'].mean():.4f} ± {df['Sensitivity_2D_Mean'].std():.4f}")
    print(f"Specificity:{df['Specificity_2D_Mean'].mean():.4f} ± {df['Specificity_2D_Mean'].std():.4f}")
    print(f"Precision: {df['Precision_2D_Mean'].mean():.4f} ± {df['Precision_2D_Mean'].std():.4f}")

    print("\nHD95（3D 体积）")
    print(f"HD95 px:   {df['HD95_3D_px'].mean():.4f} ± {df['HD95_3D_px'].std():.4f}")
    print(f"HD95 mm:   {df['HD95_3D_mm'].mean():.4f} ± {df['HD95_3D_mm'].std():.4f}")

    print("\nASD（3D 体积）")
    print(f"ASD px:    {df['ASD_3D_px'].mean():.4f} ± {df['ASD_3D_px'].std():.4f}")
    print(f"ASD mm:    {df['ASD_3D_mm'].mean():.4f} ± {df['ASD_3D_mm'].std():.4f}")

    print("\nVolume")
    print(f"Volume Error: {df['Volume_Error_%'].mean():.4f} ± {df['Volume_Error_%'].std():.4f} %")
    print(f"RVD:          {df['Relative_Volume_Diff_%'].mean():.4f} ± {df['Relative_Volume_Diff_%'].std():.4f} %")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of prediction npz files.")
    parser.add_argument("-pred_dir", type=str, required=True, help="prediction directory")
    parser.add_argument("-gt_dir", type=str, required=True, help="ground-truth directory")
    parser.add_argument("-output_csv", type=str, default="evaluation_results_comprehensive.csv", help="output csv path")
    args = parser.parse_args()

    results = evaluate_predictions(args.pred_dir, args.gt_dir)
    if not results:
        print("没有找到有效的评估结果")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print_summary(df)
    print(f"\n详细结果已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()
