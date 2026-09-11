"""Evaluate MIRAUS checkpoints on held-out CV cases without copying fold data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zlib
from glob import glob
from os.path import basename, join, normpath

import cv2
import numpy as np
import torch
from scipy import ndimage
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from enhanced_dual_modal import EnhancedDualModalMedSAM_Lite, EnhancedMaskDecoder
from prism_checkpoint_utils import normalize_prism_checkpoint_state_dict
from prism_volume_protocol import context_window_indices, volume_scope_masks
from tiny_vit_sam import TinyViT
from utils.case_split import extract_case_id, normalize_case_ids, split_root_spec


def infer_student_architecture_from_state_dict(state_dict):
    adapter_weight = state_dict.get("student_adapter.0.weight")
    if adapter_weight is None:
        return {"use_privileged_distillation": False}
    return {
        "use_privileged_distillation": True,
        "student_adapter_hidden": int(adapter_weight.shape[0]),
        "use_student_refinement_adapter": any(
            key.startswith("student_refinement_adapter.") for key in state_dict
        ),
        "use_gated_student_adapter": any(
            key.startswith("student_gate.") for key in state_dict
        ),
    }


def build_model_from_checkpoint(
    checkpoint_path,
    device,
    dual_modal=True,
    no_fusion_ablation=False,
    model_kwargs=None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_prism_checkpoint_state_dict(checkpoint)

    image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = EnhancedMaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_src_enhancement=True,
    )
    if not dual_modal:
        raise NotImplementedError("This evaluator targets MIRAUS privileged-learning checkpoints.")
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.update(infer_student_architecture_from_state_dict(state_dict))
    model = EnhancedDualModalMedSAM_Lite(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
        use_cross_modal=True,
        use_src_enhancement=True,
        use_fusion=not no_fusion_ablation,
        **model_kwargs,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _load_test_cases(work_dir, fallback_cases=None):
    if fallback_cases:
        return sorted(normalize_case_ids(fallback_cases) or [])
    config_path = join(work_dir, "training_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing training_config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    split = config.get("case_split") or {}
    test_cases = split.get("test_cases") or []
    if not test_cases:
        raise ValueError(f"No test_cases recorded in {config_path}")
    return sorted(test_cases)


def _collect_case_slices(trus_roots, case_ids):
    case_ids = set(case_ids)
    grouped = {case_id: [] for case_id in case_ids}
    for root in split_root_spec(trus_roots):
        img_dir = join(root, "imgs")
        gt_dir = join(root, "gts")
        for gt_path in sorted(glob(join(gt_dir, "*.npy"))):
            case_id = extract_case_id(gt_path)
            if case_id not in case_ids:
                continue
            img_path = join(img_dir, basename(gt_path))
            if not os.path.isfile(img_path):
                continue
            slice_part = basename(gt_path).rsplit("-", 1)[-1].split(".")[0]
            try:
                slice_idx = int(slice_part)
            except ValueError:
                slice_idx = len(grouped[case_id])
            grouped[case_id].append((slice_idx, normpath(img_path), normpath(gt_path)))
    return {case_id: sorted(entries) for case_id, entries in grouped.items() if entries}


def _context_window_indices(center, length, radius, stride=1):
    return context_window_indices(center, length, radius, stride=stride)


def _training_args(work_dir):
    config_path = join(work_dir, "training_config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return (json.load(f).get("args") or {})


def _box_from_mask(mask, bbox_shift=5):
    y, x = np.where(mask > 0)
    if len(x) == 0:
        h, w = mask.shape[:2]
        return np.array([0, 0, w - 1, h - 1], dtype=np.int32)
    h, w = mask.shape[:2]
    return np.array(
        [
            max(0, int(x.min()) - bbox_shift),
            max(0, int(y.min()) - bbox_shift),
            min(w - 1, int(x.max()) + bbox_shift),
            min(h - 1, int(y.max()) + bbox_shift),
        ],
        dtype=np.int32,
    )


def _prompt_box(mask, bbox_shift=5, box_mode="gt"):
    if box_mode == "no_prompt":
        return None
    if box_mode == "full_image":
        height, width = mask.shape[:2]
        return np.array([0, 0, width - 1, height - 1], dtype=np.int32)
    if box_mode == "center_90":
        height, width = mask.shape[:2]
        margin_x = int(round(width * 0.05))
        margin_y = int(round(height * 0.05))
        return np.array(
            [margin_x, margin_y, width - 1 - margin_x, height - 1 - margin_y],
            dtype=np.int32,
        )
    if box_mode != "gt":
        raise ValueError(f"Unknown box_mode: {box_mode}")
    return _box_from_mask(mask, bbox_shift=bbox_shift)


def _jitter_box(box, image_shape, fraction=0.0, seed=2026):
    if box is None or fraction <= 0:
        return box
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = np.asarray(box, dtype=np.float64)
    box_width = max(x_max - x_min, 1.0)
    box_height = max(y_max - y_min, 1.0)
    rng = np.random.default_rng(int(seed))
    offsets = rng.uniform(-float(fraction), float(fraction), size=4)
    x_min += offsets[0] * box_width
    y_min += offsets[1] * box_height
    x_max += offsets[2] * box_width
    y_max += offsets[3] * box_height
    x_min = int(np.clip(round(x_min), 0, width - 2))
    y_min = int(np.clip(round(y_min), 0, height - 2))
    x_max = int(np.clip(round(x_max), x_min + 1, width - 1))
    y_max = int(np.clip(round(y_max), y_min + 1, height - 1))
    return np.asarray([x_min, y_min, x_max, y_max], dtype=np.int32)


@torch.no_grad()
def _infer_slice(
    model,
    image_3c,
    gt_mask,
    device,
    bbox_shift=5,
    mask_threshold=0.5,
    box_mode="gt",
    box_jitter_fraction=0.0,
    box_jitter_seed=2026,
    trus_valid_mask=None,
    return_probability=False,
):
    h, w = gt_mask.shape[:2]
    if image_3c.ndim == 4:
        image_256 = np.stack([
            cv2.resize(image_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
            for image_slice in image_3c
        ], axis=0)
    else:
        image_256 = cv2.resize(image_3c, (256, 256), interpolation=cv2.INTER_LINEAR)
    image_256 = image_256.astype(np.float32)
    if image_256.max() > 1.0:
        image_256 = image_256 / 255.0
    if image_256.ndim == 4:
        tensor = torch.from_numpy(image_256).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    else:
        tensor = torch.from_numpy(image_256).permute(2, 0, 1).unsqueeze(0).float().to(device)

    gt_256 = cv2.resize((gt_mask > 0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    box = _prompt_box(gt_256, bbox_shift=bbox_shift, box_mode=box_mode)
    box = _jitter_box(
        box,
        gt_256.shape,
        fraction=box_jitter_fraction,
        seed=box_jitter_seed,
    )
    box_t = None
    if box is not None:
        box_t = torch.as_tensor(box[None, None, :], dtype=torch.float32, device=device)
    valid_t = None
    if image_3c.ndim == 4 and trus_valid_mask is not None:
        valid_t = torch.as_tensor(
            trus_valid_mask, dtype=torch.bool, device=device
        ).unsqueeze(0)
    if valid_t is None:
        logits, _ = model(tensor, None, box_t, training=False)
    else:
        logits, _ = model(
            tensor,
            None,
            box_t,
            training=False,
            trus_valid_mask=valid_t,
        )
    prob = torch.sigmoid(model.postprocess_masks(logits, (256, 256), (h, w))).squeeze().cpu().numpy()
    pred = (prob > mask_threshold).astype(np.uint8)
    if return_probability:
        return pred, prob.astype(np.float32)
    return pred


@torch.no_grad()
def _infer_teacher_sample(
    model,
    sample,
    device,
    mask_threshold=0.5,
    mri_mode="matched",
    bbox_shift=3,
    box_mode="gt",
    box_jitter_fraction=0.0,
    box_jitter_seed=2026,
):
    trus = sample["trus_image"].unsqueeze(0).to(device)
    mri = sample["mri_image"].unsqueeze(0).to(device)
    if mri_mode == "zero":
        mri = torch.zeros_like(mri)
    elif mri_mode not in ("matched", "mismatched"):
        raise ValueError(f"Unknown teacher MRI mode: {mri_mode}")

    gt = sample["gt2D"].squeeze().cpu().numpy()
    box = _prompt_box(gt, bbox_shift=bbox_shift, box_mode=box_mode)
    box = _jitter_box(
        box,
        gt.shape,
        fraction=box_jitter_fraction,
        seed=box_jitter_seed,
    )
    boxes = None if box is None else torch.as_tensor(
        box[None, None, :], dtype=torch.float32, device=device
    )
    trus_valid_mask = sample.get("trus_valid_mask")
    if trus_valid_mask is not None:
        trus_valid_mask = trus_valid_mask.unsqueeze(0).to(device)
    mri_valid_mask = sample.get("mri_valid_mask")
    if mri_valid_mask is not None:
        mri_valid_mask = mri_valid_mask.unsqueeze(0).to(device)
    relative_depth = sample.get("relative_depth")
    if relative_depth is not None:
        relative_depth = relative_depth.unsqueeze(0).to(device)

    if hasattr(model, "forward_teacher"):
        output = model.forward_teacher(
            trus,
            mri,
            boxes=boxes,
            mri_valid_mask=mri_valid_mask,
            relative_depth=relative_depth,
            trus_valid_mask=trus_valid_mask,
        )
    else:
        output = model(
            trus,
            mri,
            boxes,
            training=True,
            mri_valid_mask=mri_valid_mask,
            relative_depth=relative_depth,
        )
    height, width = gt.shape[-2:]
    logits = model.postprocess_masks(
        output["logits_teacher"], (256, 256), (height, width)
    )
    probability = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
    return (probability > mask_threshold).astype(np.uint8), probability


def _dice_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union_dice = pred_sum + gt_sum
    dice = 1.0 if union_dice == 0 else 2.0 * inter / union_dice
    union_iou = np.logical_or(pred, gt).sum()
    iou = 1.0 if union_iou == 0 else inter / union_iou
    return float(dice), float(iou)


def _restore_native_metric_mask(mask, native_hw=(118, 81)):
    """Restore a square preprocessed mask to the native mu-RegPro TRUS plane."""
    native_height, native_width = (int(native_hw[0]), int(native_hw[1]))
    if native_height <= 0 or native_width <= 0:
        raise ValueError(f"Invalid native metric shape: {native_hw}")
    binary = (mask > 0).astype(np.uint8)
    if binary.shape == (native_height, native_width):
        return binary
    return cv2.resize(
        binary,
        (native_width, native_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def _restore_native_metric_probability(probability, native_hw=(118, 81)):
    native_height, native_width = (int(native_hw[0]), int(native_hw[1]))
    if native_height <= 0 or native_width <= 0:
        raise ValueError(f"Invalid native metric shape: {native_hw}")
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != (native_height, native_width):
        probability = cv2.resize(
            probability,
            (native_width, native_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for surface distances: {pred.shape} vs {gt.shape}")
    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float64)
    if not pred.any() or not gt.any():
        return np.array([np.inf], dtype=np.float64)

    structure = ndimage.generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(pred, ndimage.binary_erosion(pred, structure=structure, border_value=0))
    gt_surface = np.logical_xor(gt, ndimage.binary_erosion(gt, structure=structure, border_value=0))
    if not pred_surface.any() or not gt_surface.any():
        return np.array([np.inf], dtype=np.float64)

    spacing = tuple(float(x) for x in spacing)
    if len(spacing) != pred.ndim:
        raise ValueError(f"Spacing length {len(spacing)} does not match ndim {pred.ndim}")
    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return np.concatenate([dist_to_gt[pred_surface], dist_to_pred[gt_surface]]).astype(np.float64)


def _hd95_asd(pred, gt, spacing=(1.0, 1.0, 1.0)):
    dists = _surface_distances(pred, gt, spacing=spacing)
    if np.isinf(dists).any():
        return float("inf"), float("inf")
    return float(np.percentile(dists, 95)), float(np.mean(dists))


def _normalized_surface_dice(pred, gt, spacing=(1.0, 1.0, 1.0), tolerance_mm=1.6):
    if float(tolerance_mm) < 0:
        raise ValueError("NSD tolerance must be non-negative")
    dists = _surface_distances(pred, gt, spacing=spacing)
    if np.isinf(dists).any():
        return 0.0
    return float(np.mean(dists <= float(tolerance_mm)))


def _binary_calibration(probability, target, n_bins=15):
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    target = (np.asarray(target).reshape(-1) > 0).astype(np.float64)
    if probability.shape != target.shape:
        raise ValueError("Probability and target must have matching shapes")
    probability = np.clip(probability, 0.0, 1.0)
    error_sq = (probability - target) ** 2
    eps = 1e-7
    clipped = np.clip(probability, eps, 1.0 - eps)
    nll_values = -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))

    class_brier = []
    class_nll = []
    for label in (0.0, 1.0):
        selected = target == label
        if selected.any():
            class_brier.append(float(error_sq[selected].mean()))
            class_nll.append(float(nll_values[selected].mean()))

    n_bins = max(1, int(n_bins))
    bin_ids = np.minimum((probability * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        selected = bin_ids == bin_idx
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return {
        "brier": float(error_sq.mean()),
        "balanced_brier": float(np.mean(class_brier)),
        "nll": float(nll_values.mean()),
        "balanced_nll": float(np.mean(class_nll)),
        "ece": float(ece),
    }


def _prostate_depth_thirds(gt_volume):
    gt_volume = np.asarray(gt_volume)
    positive = np.flatnonzero(np.any(gt_volume > 0, axis=tuple(range(1, gt_volume.ndim))))
    if positive.size == 0:
        return {"low_index": [], "middle": [], "high_index": []}
    extent = np.arange(int(positive[0]), int(positive[-1]) + 1)
    groups = np.array_split(extent, 3)
    return {
        name: [int(index) for index in group]
        for name, group in zip(("low_index", "middle", "high_index"), groups)
    }


def _apply_image_perturbation(image, mode="none", severity=0.0, seed=0):
    image = np.asarray(image, dtype=np.float32)
    mode = str(mode)
    severity = max(0.0, float(severity))
    if mode == "none" or severity == 0.0:
        return image.copy()
    if image.ndim == 4:
        return np.stack(
            [_apply_image_perturbation(item, mode, severity, seed + idx) for idx, item in enumerate(image)]
        )
    if image.ndim != 3:
        raise ValueError(f"Expected HWC or SHWC image, got shape {image.shape}")

    support = np.any(image != 0.0, axis=-1, keepdims=True)
    rng = np.random.default_rng(int(seed))
    if mode == "speckle":
        noise = rng.normal(0.0, severity, size=(*image.shape[:-1], 1))
        changed = image * (1.0 + noise)
    elif mode == "gaussian_noise":
        supported_values = image[support.repeat(image.shape[-1], axis=-1)]
        image_scale = float(supported_values.std()) if supported_values.size else 0.0
        noise = rng.normal(0.0, severity * image_scale, size=(*image.shape[:-1], 1))
        changed = image + noise
    elif mode == "gaussian_blur":
        changed = cv2.GaussianBlur(image, (0, 0), sigmaX=max(severity, 1e-6))
    elif mode == "contrast":
        values = image[support.repeat(image.shape[-1], axis=-1)]
        center = float(values.mean()) if values.size else 0.0
        changed = center + (image - center) * max(0.0, 1.0 - severity)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    return (np.clip(changed, 0.0, 1.0) * support).astype(np.float32)


def _depth_bin(relative_depth):
    if relative_depth < 0.2:
        return "apex"
    if relative_depth > 0.8:
        return "base"
    return "mid"


def _finite_mean_and_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), 0.0
    return float(values[finite].mean()), float(finite.mean())


def evaluate(args):
    device = torch.device(args.device)
    saved_args = _training_args(args.work_dir)
    from prism_interpretability import _model_kwargs_from_training_config

    model_kwargs = _model_kwargs_from_training_config(args.work_dir)
    if "trus_context_mix" in saved_args:
        model_kwargs["trus_context_mix"] = float(saved_args["trus_context_mix"])
    model = build_model_from_checkpoint(
        args.checkpoint,
        device,
        no_fusion_ablation=args.no_fusion_ablation,
        model_kwargs=model_kwargs,
    )
    trus_window_radius = (
        int(args.trus_window_radius)
        if int(args.trus_window_radius) >= 0
        else int(saved_args.get("trus_window_radius", 0))
    )
    test_cases = _load_test_cases(args.work_dir, args.test_cases)
    grouped = _collect_case_slices(args.trus_roots, test_cases)
    missing = sorted(set(test_cases) - set(grouped))
    if missing:
        raise ValueError(f"Missing test case slices: {missing}")

    teacher_dataset = None
    teacher_sample_indices = {}
    effective_mri_window_radius = None
    if args.eval_branch == "teacher":
        if not args.mri_roots:
            raise ValueError("--mri_roots is required for teacher evaluation")
        if args.perturbation != "none":
            raise ValueError("Teacher evaluation currently supports only --perturbation none")
        from utils.paired_dataset import PairedNpyDataset

        pairing_mode = (
            "different_patient_random"
            if args.teacher_mri_mode == "mismatched"
            else "normal"
        )
        effective_mri_window_radius = (
            int(saved_args.get("mri_window_radius", 2))
            if args.mri_window_radius < 0
            else int(args.mri_window_radius)
        )
        if effective_mri_window_radius < 0 or effective_mri_window_radius > 3:
            raise ValueError("Teacher MRI window radius must be between 0 and 3")
        teacher_dataset = PairedNpyDataset(
            args.trus_roots,
            args.mri_roots,
            bbox_shift=0,
            data_aug=False,
            mri_window_radius=effective_mri_window_radius,
            slice_attention_mode=saved_args.get("slice_attention_mode", "ssca_entropy"),
            pairing_mode=pairing_mode,
            box_mode="gt",
        )
        teacher_sample_indices = {
            (pair["case_id"], int(pair["trus_slice_index"])): index
            for index, pair in enumerate(teacher_dataset.valid_pairs)
        }

    rows = []
    slice_rows = []
    depth_rows = []
    native_hw = (int(args.metric_native_height), int(args.metric_native_width))
    spacing_3d = (float(args.slice_spacing), float(args.pixel_spacing_y), float(args.pixel_spacing_x))
    spacing_2d = (float(args.pixel_spacing_y), float(args.pixel_spacing_x))
    os.makedirs(args.output_dir, exist_ok=True)
    prediction_dir = join(args.output_dir, "predictions")
    random.seed(int(args.teacher_mri_seed))
    if args.save_predictions:
        os.makedirs(prediction_dir, exist_ok=True)
    for case_id, entries in grouped.items():
        pred_slices = []
        gt_slices = []
        probability_slices = []
        model_grid_probability_slices = []
        slice_dices = []
        slice_ious = []
        slice_hd95s = []
        slice_asds = []
        n_slices = len(entries)
        for slice_order, (slice_idx, img_path, gt_path) in enumerate(entries):
            gt = np.load(gt_path, "r", allow_pickle=True)
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt_bin = (gt > 0).astype(np.uint8)
            prompt_key = f"{case_id}:{slice_idx}".encode("utf-8")
            prompt_seed = int(args.box_jitter_seed) + zlib.crc32(prompt_key)
            if args.skip_empty_gt and not gt_bin.any():
                pred = np.zeros_like(gt_bin, dtype=np.uint8)
                probability = np.zeros_like(gt_bin, dtype=np.float32)
            elif args.eval_branch == "teacher":
                sample_key = (case_id, int(slice_idx))
                if sample_key not in teacher_sample_indices:
                    raise ValueError(f"Missing paired teacher sample: {sample_key}")
                sample = teacher_dataset[teacher_sample_indices[sample_key]]
                pred, probability = _infer_teacher_sample(
                    model,
                    sample,
                    device,
                    mask_threshold=args.mask_threshold,
                    mri_mode=args.teacher_mri_mode,
                    bbox_shift=args.bbox_shift,
                    box_mode=args.box_mode,
                    box_jitter_fraction=args.box_jitter_fraction,
                    box_jitter_seed=prompt_seed,
                )
            else:
                context_indices, trus_valid_mask = _context_window_indices(
                    slice_order,
                    n_slices,
                    trus_window_radius,
                    stride=args.context_stride,
                )
                context_images = []
                for context_index in context_indices:
                    context_img = np.load(entries[context_index][1], "r", allow_pickle=True)
                    if context_img.ndim == 2:
                        context_img = np.repeat(context_img[:, :, None], 3, axis=-1)
                    context_slice_idx = entries[context_index][0]
                    perturb_key = f"{case_id}:{context_slice_idx}".encode("utf-8")
                    perturb_seed = int(args.perturb_seed) + zlib.crc32(perturb_key)
                    context_img = _apply_image_perturbation(
                        context_img,
                        mode=args.perturbation,
                        severity=args.perturbation_severity,
                        seed=perturb_seed,
                    )
                    context_images.append(context_img)
                img = context_images[0] if trus_window_radius == 0 else np.stack(context_images)
                pred, probability = _infer_slice(
                    model,
                    img,
                    gt_bin,
                    device,
                    bbox_shift=args.bbox_shift,
                    mask_threshold=args.mask_threshold,
                    box_mode=args.box_mode,
                    box_jitter_fraction=args.box_jitter_fraction,
                    box_jitter_seed=prompt_seed,
                    trus_valid_mask=trus_valid_mask,
                    return_probability=True,
                )
            pred_metric = _restore_native_metric_mask(pred, native_hw=native_hw)
            gt_metric = _restore_native_metric_mask(gt_bin, native_hw=native_hw)
            probability_metric = _restore_native_metric_probability(probability, native_hw=native_hw)
            dice, iou = _dice_iou(pred_metric, gt_metric)
            hd95_2d, asd_2d = _hd95_asd(pred_metric, gt_metric, spacing=spacing_2d)
            slice_dices.append(dice)
            slice_ious.append(iou)
            slice_hd95s.append(hd95_2d)
            slice_asds.append(asd_2d)
            pred_slices.append(pred_metric)
            gt_slices.append(gt_metric)
            probability_slices.append(probability_metric)
            model_grid_probability_slices.append(probability.astype(np.float32))
            relative_depth = 0.0 if n_slices <= 1 else slice_order / float(n_slices - 1)
            slice_rows.append(
                {
                    "case_id": case_id,
                    "slice_idx": int(slice_idx),
                    "slice_order": int(slice_order),
                    "num_slices": int(n_slices),
                    "relative_depth": float(relative_depth),
                    "depth_bin": _depth_bin(relative_depth),
                    "dice": dice,
                    "iou": iou,
                    "hd95": hd95_2d,
                    "asd": asd_2d,
                    "gt_area": int(gt_metric.sum()),
                    "pred_area": int(pred_metric.sum()),
                }
            )
        pred_3d = np.stack(pred_slices, axis=0)
        gt_3d = np.stack(gt_slices, axis=0)
        probability_3d = np.stack(probability_slices, axis=0)
        dice_3d, iou_3d = _dice_iou(pred_3d, gt_3d)
        hd95_3d, asd_3d = _hd95_asd(pred_3d, gt_3d, spacing=spacing_3d)
        nsd_3d = _normalized_surface_dice(
            pred_3d,
            gt_3d,
            spacing=spacing_3d,
            tolerance_mm=args.nsd_tolerance_mm,
        )
        gt_nonempty = volume_scope_masks(gt_3d)["gt_nonempty"]
        pred_3d_nonempty = pred_3d[gt_nonempty]
        gt_3d_nonempty = gt_3d[gt_nonempty]
        dice_3d_nonempty, iou_3d_nonempty = _dice_iou(
            pred_3d_nonempty, gt_3d_nonempty
        )
        hd95_3d_nonempty, asd_3d_nonempty = _hd95_asd(
            pred_3d_nonempty, gt_3d_nonempty, spacing=spacing_3d
        )
        nsd_3d_nonempty = _normalized_surface_dice(
            pred_3d_nonempty,
            gt_3d_nonempty,
            spacing=spacing_3d,
            tolerance_mm=args.nsd_tolerance_mm,
        )
        calibration = _binary_calibration(
            probability_3d,
            gt_3d,
            n_bins=args.calibration_bins,
        )
        hd95_2d_finite_mean, hd95_2d_valid_fraction = _finite_mean_and_fraction(slice_hd95s)
        asd_2d_finite_mean, asd_2d_valid_fraction = _finite_mean_and_fraction(slice_asds)
        rows.append(
            {
                "case_id": case_id,
                "num_slices": len(entries),
                "dice_2d_mean": float(np.mean(slice_dices)),
                "iou_2d_mean": float(np.mean(slice_ious)),
                "hd95_2d_mean": hd95_2d_finite_mean,
                "asd_2d_mean": asd_2d_finite_mean,
                "hd95_2d_valid_fraction": hd95_2d_valid_fraction,
                "asd_2d_valid_fraction": asd_2d_valid_fraction,
                "dice_3d": dice_3d,
                "iou_3d": iou_3d,
                "hd95_3d": hd95_3d,
                "asd_3d": asd_3d,
                "nsd_3d": nsd_3d,
                "num_gt_nonempty_slices": int(gt_nonempty.sum()),
                "dice_3d_gt_nonempty": dice_3d_nonempty,
                "iou_3d_gt_nonempty": iou_3d_nonempty,
                "hd95_3d_gt_nonempty": hd95_3d_nonempty,
                "asd_3d_gt_nonempty": asd_3d_nonempty,
                "nsd_3d_gt_nonempty": nsd_3d_nonempty,
                **calibration,
            }
        )

        for depth_bin, indices in _prostate_depth_thirds(gt_3d).items():
            if not indices:
                continue
            pred_region = pred_3d[indices]
            gt_region = gt_3d[indices]
            depth_dice, depth_iou = _dice_iou(pred_region, gt_region)
            depth_hd95, depth_asd = _hd95_asd(pred_region, gt_region, spacing=spacing_3d)
            depth_rows.append(
                {
                    "case_id": case_id,
                    "depth_bin": depth_bin,
                    "start_slice_order": min(indices),
                    "end_slice_order": max(indices),
                    "num_slices": len(indices),
                    "dice_3d": depth_dice,
                    "iou_3d": depth_iou,
                    "hd95_3d": depth_hd95,
                    "asd_3d": depth_asd,
                    "nsd_3d": _normalized_surface_dice(
                        pred_region,
                        gt_region,
                        spacing=spacing_3d,
                        tolerance_mm=args.nsd_tolerance_mm,
                    ),
                }
            )
        if args.save_predictions:
            np.savez_compressed(
                join(prediction_dir, f"{case_id}.npz"),
                probability=probability_3d.astype(np.float16),
                probability_model_grid=np.stack(model_grid_probability_slices).astype(np.float16),
                prediction=pred_3d.astype(np.uint8),
                ground_truth=gt_3d.astype(np.uint8),
                spacing_mm=np.asarray(spacing_3d, dtype=np.float32),
            )

    csv_path = join(args.output_dir, "cv_test_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    slice_csv_path = join(args.output_dir, "cv_test_slice_metrics.csv")
    with open(slice_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
        writer.writeheader()
        writer.writerows(slice_rows)
    depth_csv_path = join(args.output_dir, "cv_test_depth_metrics.csv")
    with open(depth_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(depth_rows)

    summary = {
        "eval_branch": args.eval_branch,
        "teacher_mri_mode": args.teacher_mri_mode if args.eval_branch == "teacher" else None,
        "mri_window_radius": effective_mri_window_radius,
        "mri_num_slices": (
            2 * effective_mri_window_radius + 1
            if effective_mri_window_radius is not None
            else None
        ),
        "box_mode": args.box_mode,
        "box_jitter_fraction": float(args.box_jitter_fraction),
        "box_jitter_seed": int(args.box_jitter_seed),
        "skip_empty_gt": bool(args.skip_empty_gt),
        "perturbation": args.perturbation,
        "perturbation_severity": float(args.perturbation_severity),
        "distance_unit": "mm",
        "nsd_tolerance_mm": float(args.nsd_tolerance_mm),
        "metric_native_hw": list(native_hw),
        "spacing_3d_mm": list(spacing_3d),
        "num_cases": len(rows),
        "trus_window_radius": trus_window_radius,
        "context_stride": int(args.context_stride),
        "dice_2d_mean": float(np.mean([r["dice_2d_mean"] for r in rows])),
        "iou_2d_mean": float(np.mean([r["iou_2d_mean"] for r in rows])),
        "hd95_2d_mean": float(np.mean([r["hd95_2d_mean"] for r in rows])),
        "asd_2d_mean": float(np.mean([r["asd_2d_mean"] for r in rows])),
        "dice_3d_mean": float(np.mean([r["dice_3d"] for r in rows])),
        "dice_3d_std": float(np.std([r["dice_3d"] for r in rows], ddof=0)),
        "iou_3d_mean": float(np.mean([r["iou_3d"] for r in rows])),
        "hd95_3d_mean": float(np.mean([r["hd95_3d"] for r in rows])),
        "hd95_3d_std": float(np.std([r["hd95_3d"] for r in rows], ddof=0)),
        "asd_3d_mean": float(np.mean([r["asd_3d"] for r in rows])),
        "asd_3d_std": float(np.std([r["asd_3d"] for r in rows], ddof=0)),
        "nsd_3d_mean": float(np.mean([r["nsd_3d"] for r in rows])),
        "nsd_3d_std": float(np.std([r["nsd_3d"] for r in rows], ddof=0)),
        "evaluation_scopes": ["all", "gt_nonempty"],
        "num_gt_nonempty_slices": int(sum(r["num_gt_nonempty_slices"] for r in rows)),
        "dice_3d_gt_nonempty_mean": float(np.mean([r["dice_3d_gt_nonempty"] for r in rows])),
        "dice_3d_gt_nonempty_std": float(np.std([r["dice_3d_gt_nonempty"] for r in rows], ddof=0)),
        "iou_3d_gt_nonempty_mean": float(np.mean([r["iou_3d_gt_nonempty"] for r in rows])),
        "hd95_3d_gt_nonempty_mean": float(np.mean([r["hd95_3d_gt_nonempty"] for r in rows])),
        "hd95_3d_gt_nonempty_std": float(np.std([r["hd95_3d_gt_nonempty"] for r in rows], ddof=0)),
        "asd_3d_gt_nonempty_mean": float(np.mean([r["asd_3d_gt_nonempty"] for r in rows])),
        "asd_3d_gt_nonempty_std": float(np.std([r["asd_3d_gt_nonempty"] for r in rows], ddof=0)),
        "nsd_3d_gt_nonempty_mean": float(np.mean([r["nsd_3d_gt_nonempty"] for r in rows])),
        "nsd_3d_gt_nonempty_std": float(np.std([r["nsd_3d_gt_nonempty"] for r in rows], ddof=0)),
        "ece_mean": float(np.mean([r["ece"] for r in rows])),
        "brier_mean": float(np.mean([r["brier"] for r in rows])),
        "balanced_brier_mean": float(np.mean([r["balanced_brier"] for r in rows])),
        "nll_mean": float(np.mean([r["nll"] for r in rows])),
        "balanced_nll_mean": float(np.mean([r["balanced_nll"] for r in rows])),
    }
    with open(join(args.output_dir, "cv_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {slice_csv_path}")
    print(f"[OK] wrote {depth_csv_path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a MIRAUS CV fold on held-out .npy cases.")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trus_roots", required=True, help="Semicolon-separated TRUS channel roots.")
    parser.add_argument("--mri_roots", default="", help="Semicolon-separated MRI roots for teacher evaluation.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_cases", default="", help="Optional explicit case IDs. Defaults to work_dir training_config.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_branch", choices=["student", "teacher"], default="student")
    parser.add_argument(
        "--teacher_mri_mode",
        choices=["matched", "mismatched", "zero"],
        default="matched",
    )
    parser.add_argument("--teacher_mri_seed", type=int, default=2026)
    parser.add_argument("--bbox_shift", type=int, default=5)
    parser.add_argument("--box_jitter_fraction", type=float, default=0.0)
    parser.add_argument("--box_jitter_seed", type=int, default=2026)
    parser.add_argument(
        "--skip_empty_gt",
        action="store_true",
        help="Do not issue a prompt on GT-empty slices; prediction is set to empty.",
    )
    parser.add_argument(
        "--box_mode",
        choices=["full_image", "center_90", "no_prompt", "gt"],
        default="gt",
    )
    parser.add_argument("--trus_window_radius", type=int, default=-1)
    parser.add_argument(
        "--mri_window_radius",
        type=int,
        default=-1,
        help="Teacher MRI radius override; -1 uses the training configuration.",
    )
    parser.add_argument("--context_stride", type=int, default=1)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--nsd_tolerance_mm", type=float, default=1.6)
    parser.add_argument("--calibration_bins", type=int, default=15)
    parser.add_argument(
        "--perturbation",
        choices=["none", "speckle", "gaussian_noise", "gaussian_blur", "contrast"],
        default="none",
    )
    parser.add_argument("--perturbation_severity", type=float, default=0.0)
    parser.add_argument("--perturb_seed", type=int, default=2026)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--pixel_spacing_y", type=float, default=0.8)
    parser.add_argument("--pixel_spacing_x", type=float, default=0.8)
    parser.add_argument("--slice_spacing", type=float, default=0.8)
    parser.add_argument("--metric_native_height", type=int, default=118)
    parser.add_argument("--metric_native_width", type=int, default=81)
    parser.add_argument("--no_fusion_ablation", action="store_true")
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
