"""Feature-transfer and SSCA-attention analysis for MIRAUS checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from prism_cv_npy_eval import build_model_from_checkpoint, _load_test_cases
from utils.paired_dataset import PairedNpyDataset


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    k = k - k.mean(axis=0, keepdims=True) - k.mean(axis=1, keepdims=True) + k.mean()
    l = l - l.mean(axis=0, keepdims=True) - l.mean(axis=1, keepdims=True) + l.mean()
    denom = np.linalg.norm(k, "fro") * np.linalg.norm(l, "fro")
    if denom <= 1e-12:
        return float("nan")
    return float((k * l).sum() / denom)


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pool_feature(x: torch.Tensor) -> np.ndarray:
    return x.detach().mean(dim=(2, 3)).cpu().numpy()


def _save_attention_plots(rows, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable: {exc}")
        return

    if not rows:
        return
    max_len = max(len(row["beta"]) for row in rows)
    beta_mat = np.full((len(rows), max_len), np.nan, dtype=np.float32)
    rel_depth = np.asarray([row["relative_depth"] for row in rows], dtype=np.float32)
    order = np.argsort(rel_depth)
    for dst, src in enumerate(order):
        beta = np.asarray(rows[src]["beta"], dtype=np.float32)
        beta_mat[dst, : len(beta)] = beta
    rel_depth_sorted = rel_depth[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    im = ax.imshow(beta_mat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("MRI window offset index")
    ax.set_ylabel("TRUS relative depth sorted slices")
    ax.set_title("SSCA MRI slice attention beta")
    tick_positions = np.linspace(0, len(rel_depth_sorted) - 1, 6).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{rel_depth_sorted[i]:.2f}" for i in tick_positions])
    if max_len % 2 == 1:
        offsets = np.arange(max_len) - max_len // 2
        ax.set_xticks(np.arange(max_len))
        ax.set_xticklabels([str(int(x)) for x in offsets])
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_dir / "ssca_beta_heatmap.png")
    plt.close(fig)

    centers = []
    entropies = []
    for row in rows:
        beta = np.asarray(row["beta"], dtype=np.float64)
        offsets = np.arange(len(beta), dtype=np.float64) - (len(beta) // 2)
        centers.append(float((beta * offsets).sum()))
        entropy = -(beta * np.log(np.clip(beta, 1e-8, 1.0))).sum()
        entropies.append(float(entropy / np.log(max(len(beta), 2))))
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax1.scatter(rel_depth, centers, s=14, alpha=0.65, color="#1f77b4", label="attention center")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Relative prostate depth (0=apex, 1=base)")
    ax1.set_ylabel("Expected MRI window offset")
    ax2 = ax1.twinx()
    ax2.scatter(rel_depth, entropies, s=14, alpha=0.45, color="#d62728", label="normalized entropy")
    ax2.set_ylabel("Normalized attention entropy")
    lines = ax1.collections + ax2.collections
    labels = ["attention center", "normalized entropy"]
    ax1.legend(lines, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "ssca_attention_depth_scatter.png")
    plt.close(fig)


def _model_kwargs_from_training_config(work_dir: str):
    config_path = Path(work_dir) / "training_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    args = config.get("args") or {}
    keys = [
        "use_adaptive_fusion",
        "slice_attention_mode",
        "ssca_descriptor_dim",
        "ssca_beta_temperature",
        "ssca_min_confidence",
        "ssca_position_prior_weight",
        "ssca_position_prior_sigma",
        "ssca_max_window_size",
        "slice_corr_loss_weight",
        "slice_corr_prior_sigma",
        "ssca_use_box_aware_pooling",
        "ssca_boundary_ring_width",
        "slice_utility_loss_weight",
        "slice_utility_temperature",
        "use_transition_aware_beta",
        "transition_loss_weight",
        "jump_logit_penalty",
        "use_reliability_gate",
        "reliability_use_candidate_agreement",
        "transition_cls_loss_weight",
        "transition_reg_loss_weight",
        "use_dynamic_bandwidth_beta",
        "sigma_min",
        "sigma_max",
        "dynamic_bandwidth_use_gt_transition_prob",
        "dynamic_bandwidth_warmup_epochs",
        "dynamic_beta_mode",
        "dynamic_prior_weight",
        "use_neighbor_residual_fusion",
        "use_gt_transition_for_beta",
        "use_gt_transition_for_neighbor_trust",
        "depth_prior_enabled",
        "depth_embed_dim",
        "depth_prior_hidden_dim",
        "lambda_depth_prior",
        "depth_gate_enabled",
        "depth_gate_alpha_min",
        "depth_gate_alpha_max",
        "use_depth_gated_mmd",
        "dg_mmd_project_dim",
        "dg_mmd_lambda_center",
        "dg_mmd_lambda_priv",
        "dg_mmd_min_priv_weight",
        "beta_modulation_enabled",
        "beta_modulation_mix_max",
        "use_privileged_distillation",
        "use_spatial_fusion_gate",
        "spatial_gate_conf_floor",
        "use_foreground_mmd",
        "foreground_mmd_min_tokens",
        "trus_context_mix",
    ]
    kwargs = {key: args[key] for key in keys if key in args}
    if "depth_prior_no_u_shape_regularization" in args:
        kwargs["depth_prior_use_u_shape_regularization"] = not bool(
            args["depth_prior_no_u_shape_regularization"]
        )
    return kwargs


def _box_mode_from_training_config(work_dir: str):
    config_path = Path(work_dir) / "training_config.json"
    if not config_path.exists():
        return "gt"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    box_mode = str((config.get("args") or {}).get("training_box_mode", "gt"))
    if box_mode not in ("gt", "full_image"):
        raise ValueError(f"Unknown training_box_mode in {config_path}: {box_mode}")
    return box_mode


def _no_fusion_from_training_config(work_dir: str):
    config_path = Path(work_dir) / "training_config.json"
    if not config_path.exists():
        return False
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return bool((config.get("args") or {}).get("no_fusion", False))


def _teacher_checkpoint_from_training_config(work_dir: str):
    config_path = Path(work_dir) / "training_config.json"
    if not config_path.exists():
        return ""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return str((config.get("args") or {}).get("teacher_checkpoint", ""))


def _trus_window_radius_from_training_config(work_dir: str):
    config_path = Path(work_dir) / "training_config.json"
    if not config_path.exists():
        return 0
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return int((config.get("args") or {}).get("trus_window_radius", 0))


@torch.no_grad()
def analyze(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_kwargs = _model_kwargs_from_training_config(args.work_dir)
    no_fusion = _no_fusion_from_training_config(args.work_dir)
    model = build_model_from_checkpoint(
        args.checkpoint,
        device,
        no_fusion_ablation=no_fusion,
        model_kwargs=model_kwargs,
    )
    model.eval()
    teacher_checkpoint = (
        args.teacher_checkpoint
        or _teacher_checkpoint_from_training_config(args.work_dir)
    )
    teacher_model = model
    if teacher_checkpoint:
        teacher_model = build_model_from_checkpoint(
            teacher_checkpoint,
            device,
            no_fusion_ablation=no_fusion,
            model_kwargs=model_kwargs,
        )
        teacher_model.eval()

    test_cases = _load_test_cases(args.work_dir, args.test_cases)
    dataset = PairedNpyDataset(
        args.trus_roots,
        args.mri_roots,
        image_size=256,
        bbox_shift=args.bbox_shift,
        data_aug=False,
        trus_window_radius=_trus_window_radius_from_training_config(args.work_dir),
        mri_window_radius=args.mri_window_radius,
        slice_attention_mode=args.slice_attention_mode,
        pairing_mode="normal",
        include_cases=test_cases,
        box_mode=_box_mode_from_training_config(args.work_dir),
    )
    indices = np.linspace(0, len(dataset) - 1, min(args.max_slices, len(dataset))).astype(int)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)

    trus_feats = []
    student_feats = []
    teacher_feats = []
    student_residual_feats = []
    teacher_delta_feats = []
    residual_norms = []
    attention_rows = []

    for batch in loader:
        trus = batch["trus_image"].to(device)
        mri = batch["mri_image"].to(device)
        boxes = batch["bboxes"].to(device)
        gt = batch["gt2D"].to(device)
        trus_valid_mask = batch.get("trus_valid_mask")
        if trus_valid_mask is not None:
            trus_valid_mask = trus_valid_mask.to(device)
        student_out = model.forward_student(
            trus, boxes=boxes, trus_valid_mask=trus_valid_mask
        )
        teacher_out = teacher_model.forward_teacher(
            trus,
            mri,
            boxes=boxes,
            mask_gt=gt,
            mri_valid_mask=batch.get("mri_valid_mask").to(device),
            relative_depth=batch.get("relative_depth").to(device),
            trus_valid_mask=trus_valid_mask,
        )
        out = {**teacher_out, **student_out}
        trus_feats.append(_pool_feature(out["feat_trus"]))
        student_feats.append(_pool_feature(out["feat_student"]))
        teacher_feats.append(_pool_feature(out["feat_teacher"]))
        student_residual_feats.append(_pool_feature(out["student_residual"]))
        teacher_delta_feats.append(_pool_feature(out["feat_teacher"] - out["feat_trus"]))
        residual_norms.extend(
            out["student_residual"].detach().pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().tolist()
        )

        info = getattr(teacher_model.cross_modal_extractor, "last_slice_attention", None) or {}
        beta = _to_numpy(info.get("beta"))
        confidence = _to_numpy(info.get("confidence"))
        entropy = _to_numpy(info.get("entropy"))
        fusion_gate = _to_numpy(info.get("fusion_gate"))
        case_ids = batch.get("case_id")
        image_names = batch.get("image_name")
        for i in range(trus.shape[0]):
            beta_i = beta[i].astype(float).tolist() if beta is not None else []
            confidence_i = float(np.ravel(confidence[i])[0]) if confidence is not None else float("nan")
            entropy_i = float(np.ravel(entropy[i])[0]) if entropy is not None else float("nan")
            fusion_i = float(np.ravel(fusion_gate[i])[0]) if fusion_gate is not None else float("nan")
            attention_rows.append(
                {
                    "case_id": case_ids[i] if isinstance(case_ids, (list, tuple)) else str(case_ids),
                    "image_name": image_names[i] if isinstance(image_names, (list, tuple)) else str(image_names),
                    "relative_depth": float(batch["relative_depth"][i].item()),
                    "mri_window_indices": " ".join(str(int(x)) for x in batch["mri_window_indices"][i].tolist()),
                    "mri_valid_mask": " ".join(str(int(x)) for x in batch["mri_valid_mask"][i].tolist()),
                    "confidence": confidence_i,
                    "entropy": entropy_i,
                    "fusion_gate": fusion_i,
                    "beta": beta_i,
                    **{f"beta_{j}": value for j, value in enumerate(beta_i)},
                }
            )

    trus_arr = np.concatenate(trus_feats, axis=0)
    student_arr = np.concatenate(student_feats, axis=0)
    teacher_arr = np.concatenate(teacher_feats, axis=0)
    student_residual_arr = np.concatenate(student_residual_feats, axis=0)
    teacher_delta_arr = np.concatenate(teacher_delta_feats, axis=0)
    residual_cosine = (
        student_residual_arr * teacher_delta_arr
    ).sum(axis=1) / (
        np.linalg.norm(student_residual_arr, axis=1) * np.linalg.norm(teacher_delta_arr, axis=1) + 1e-12
    )
    summary = {
        "num_slices": int(trus_arr.shape[0]),
        "cka_trus_teacher": _linear_cka(trus_arr, teacher_arr),
        "cka_student_teacher": _linear_cka(student_arr, teacher_arr),
        "cka_trus_student": _linear_cka(trus_arr, student_arr),
        "cka_student_residual_teacher_delta": _linear_cka(student_residual_arr, teacher_delta_arr),
        "residual_delta_cosine_mean": float(np.mean(residual_cosine)),
        "residual_delta_cosine_std": float(np.std(residual_cosine, ddof=0)),
        "student_residual_norm_mean": float(np.mean(residual_norms)),
        "student_residual_norm_std": float(np.std(residual_norms, ddof=0)),
    }
    with open(out_dir / "cka_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fieldnames = sorted({key for row in attention_rows for key in row.keys() if key != "beta"})
    with open(out_dir / "slice_attention.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in attention_rows:
            row = {k: v for k, v in row.items() if k != "beta"}
            writer.writerow(row)
    _save_attention_plots(attention_rows, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[OK] wrote {out_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze MIRAUS feature transfer and SSCA attention.")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--trus_roots", required=True)
    parser.add_argument("--mri_roots", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_cases", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox_shift", type=int, default=3)
    parser.add_argument("--mri_window_radius", type=int, default=2)
    parser.add_argument("--slice_attention_mode", default="ssca_entropy")
    parser.add_argument("--max_slices", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
