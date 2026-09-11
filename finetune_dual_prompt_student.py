"""Adapt a distilled MIRAUS student to full-image and coarse-box prompts."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME = ROOT
CASE_PATTERN = re.compile(r"case\d{6}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a distilled student's image representation and adapt only its "
            "prompt encoder/mask decoder using progressive mixed prompts."
        )
    )
    parser.add_argument("--source-work-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--trus-roots", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prompt-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gt-box-probability-start", type=float, default=0.0)
    parser.add_argument("--gt-box-probability-end", type=float, default=0.5)
    parser.add_argument("--gt-box-ramp-epochs", type=int, default=5)
    parser.add_argument("--train-bbox-shift", type=int, default=10)
    parser.add_argument("--validation-bbox-shift", type=int, default=5)
    parser.add_argument("--full-image-rehearsal-weight", type=float, default=0.5)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--save-latest",
        action="store_true",
        help="also keep a rolling latest checkpoint; disabled by default to save disk",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="exclude empty slices; default keeps them as full-image rehearsal samples",
    )
    return parser.parse_args(argv)


def progressive_probability(epoch, start, end, ramp_epochs):
    if ramp_epochs <= 0:
        return float(end)
    progress = min(max(float(epoch) / float(ramp_epochs), 0.0), 1.0)
    return float(start + (end - start) * progress)


def full_image_boxes(batch_size, height, width, device=None):
    box = torch.tensor(
        [0.0, 0.0, float(width - 1), float(height - 1)],
        dtype=torch.float32,
        device=device,
    )
    return box.view(1, 1, 4).repeat(int(batch_size), 1, 1)


def prompt_boxes_from_masks(
    masks,
    gt_box_probability,
    bbox_shift,
    generator=None,
    deterministic_gt=False,
):
    """Create per-sample mixed prompts; GT-empty samples always use full image."""
    masks = torch.as_tensor(masks).detach().cpu()
    if masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Expected masks shaped Bx1xHxW, got {tuple(masks.shape)}")
    batch_size, _, height, width = masks.shape
    boxes = full_image_boxes(batch_size, height, width)
    uses_gt_box = torch.zeros(batch_size, dtype=torch.bool)
    probability = float(gt_box_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"GT-box probability must be in [0, 1], got {probability}")
    if int(bbox_shift) < 0:
        raise ValueError("bbox_shift must be non-negative")

    for index in range(batch_size):
        coordinates = torch.nonzero(masks[index, 0] > 0, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        use_gt = deterministic_gt or bool(
            torch.rand((), generator=generator).item() < probability
        )
        if not use_gt:
            continue
        y_min, x_min = coordinates.min(dim=0).values.tolist()
        y_max, x_max = coordinates.max(dim=0).values.tolist()
        if deterministic_gt:
            shifts = [int(bbox_shift)] * 4
        else:
            shifts = torch.randint(
                0,
                int(bbox_shift) + 1,
                (4,),
                generator=generator,
            ).tolist()
        left, top, right, bottom = shifts
        boxes[index, 0] = torch.tensor(
            [
                max(0, int(x_min) - left),
                max(0, int(y_min) - top),
                min(width - 1, int(x_max) + right),
                min(height - 1, int(y_max) + bottom),
            ],
            dtype=torch.float32,
        )
        uses_gt_box[index] = True
    return boxes, uses_gt_box


class TrusSliceDataset(Dataset):
    def __init__(self, root_spec, case_ids, positive_only=False):
        self.case_ids = set(case_ids)
        self.records = []
        for root_text in str(root_spec).split(";"):
            root = Path(root_text.strip()).resolve()
            for image_path in sorted((root / "imgs").glob("*.npy")):
                match = CASE_PATTERN.search(image_path.name)
                if match is None or match.group(0) not in self.case_ids:
                    continue
                gt_path = root / "gts" / image_path.name
                if not gt_path.is_file():
                    raise FileNotFoundError(f"Missing GT for {image_path}")
                if positive_only and not np.any(np.load(gt_path, mmap_mode="r") > 0):
                    continue
                self.records.append((image_path, gt_path, match.group(0)))
        if not self.records:
            raise ValueError(f"No TRUS slices found for {sorted(self.case_ids)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, gt_path, case_id = self.records[index]
        image = np.asarray(np.load(image_path), dtype=np.float32)
        gt = (np.asarray(np.load(gt_path)) > 0).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape != (256, 256, 3) or gt.shape != (256, 256):
            raise ValueError(
                f"Expected 256x256 source arrays, got {image.shape}, {gt.shape}"
            )
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "gt": torch.from_numpy(gt[None]),
            "case_id": case_id,
            "name": image_path.name,
        }


def import_formal_runtime(runtime_dir):
    runtime_dir = Path(runtime_dir).resolve()
    sys.path.insert(0, str(runtime_dir))
    cv_eval = importlib.import_module("prism_cv_npy_eval")
    interpretability = importlib.import_module("prism_interpretability")
    if runtime_dir not in Path(cv_eval.__file__).resolve().parents:
        raise RuntimeError(f"Unexpected formal runtime module: {cv_eval.__file__}")
    return cv_eval, interpretability


def load_formal_student(source_work_dir, checkpoint, runtime_dir, device):
    cv_eval, interpretability = import_formal_runtime(runtime_dir)
    config = json.loads(
        (source_work_dir / "training_config.json").read_text(encoding="utf-8")
    )
    saved_args = config.get("args", {})
    model_kwargs = interpretability._model_kwargs_from_training_config(
        str(source_work_dir)
    )
    if "trus_context_mix" in saved_args:
        model_kwargs["trus_context_mix"] = float(saved_args["trus_context_mix"])
    model = cv_eval.build_model_from_checkpoint(
        str(checkpoint),
        device,
        no_fusion_ablation=bool(saved_args.get("no_fusion", False)),
        model_kwargs=model_kwargs,
    )
    return model, config


def soft_dice_loss(logits, target, epsilon=1e-6):
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def hard_iou_target(logits, target, epsilon=1e-6):
    prediction = torch.sigmoid(logits).detach() > 0.5
    target_bool = target > 0.5
    intersection = (prediction & target_bool).sum(dim=(1, 2, 3)).float()
    union = (prediction | target_bool).sum(dim=(1, 2, 3)).float()
    return torch.where(union > 0, intersection / (union + epsilon), torch.ones_like(union))


def supervised_loss(logits, iou_prediction, target, weights):
    dice = soft_dice_loss(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    iou_target = hard_iou_target(logits, target)
    iou_prediction = iou_prediction.reshape(iou_prediction.shape[0], -1)[:, 0]
    iou = F.mse_loss(iou_prediction, iou_target)
    total = weights["dice"] * dice + weights["bce"] * bce + weights["iou"] * iou
    return total, {"dice_loss": dice, "bce_loss": bce, "iou_loss": iou}


def student_feature(model, images):
    with torch.no_grad():
        trus_feature = model._encode_trus(images)
        return model._student_feature(trus_feature)


def configure_prompt_adaptation(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.prompt_encoder, model.mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.eval()
    model.prompt_encoder.train()
    model.mask_decoder.train()


def update_case_overlap(accumulator, case_ids, prediction, target):
    prediction = prediction.detach().cpu().bool()
    target = target.detach().cpu().bool()
    for index, case_id in enumerate(case_ids):
        stats = accumulator[str(case_id)]
        stats[0] += int((prediction[index] & target[index]).sum())
        stats[1] += int(prediction[index].sum())
        stats[2] += int(target[index].sum())


def summarize_case_dice(accumulator):
    values = []
    for intersection, predicted, target in accumulator.values():
        denominator = predicted + target
        values.append(1.0 if denominator == 0 else 2.0 * intersection / denominator)
    return float(np.mean(values)), values


@torch.no_grad()
def validate_dual_prompt(model, loader, device, bbox_shift):
    model.eval()
    full_stats = defaultdict(lambda: [0, 0, 0])
    box_stats = defaultdict(lambda: [0, 0, 0])
    for batch in loader:
        images = batch["image"].to(device)
        target = batch["gt"].to(device)
        features = student_feature(model, images)
        full_boxes = full_image_boxes(
            images.shape[0], images.shape[-2], images.shape[-1], device=device
        )
        gt_boxes, _ = prompt_boxes_from_masks(
            target,
            gt_box_probability=1.0,
            bbox_shift=bbox_shift,
            deterministic_gt=True,
        )
        full_logits, _ = model._decode(features, full_boxes)
        box_logits, _ = model._decode(features, gt_boxes.to(device))
        update_case_overlap(
            full_stats, batch["case_id"], torch.sigmoid(full_logits) > 0.5, target
        )
        update_case_overlap(
            box_stats, batch["case_id"], torch.sigmoid(box_logits) > 0.5, target
        )
    full_dice, full_values = summarize_case_dice(full_stats)
    box_dice, box_values = summarize_case_dice(box_stats)
    harmonic = (
        0.0
        if full_dice + box_dice == 0
        else 2.0 * full_dice * box_dice / (full_dice + box_dice)
    )
    return {
        "full_image_dice": full_dice,
        "gt_box_dice": box_dice,
        "harmonic_dice": harmonic,
        "worst_mode_dice": min(full_dice, box_dice),
        "full_image_case_sd": float(np.std(full_values, ddof=1)),
        "gt_box_case_sd": float(np.std(box_values, ddof=1)),
    }


def write_history(path, history):
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(path, model, optimizer, epoch, validation, adaptation_config):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "val_dice": float(validation["harmonic_dice"]),
            "val_dice_3d": float(validation["harmonic_dice"]),
            "dual_prompt_validation": validation,
            "dual_prompt_adaptation": adaptation_config,
        },
        path,
    )


def train(args):
    source_work_dir = args.source_work_dir.resolve()
    source_checkpoint = (
        args.source_checkpoint.resolve()
        if args.source_checkpoint is not None
        else source_work_dir / "dual_modal_best_3d.pth"
    )
    runtime_dir = args.runtime_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, source_config = load_formal_student(
        source_work_dir, source_checkpoint, runtime_dir, device
    )
    split = source_config.get("case_split") or {}
    train_cases = split.get("train_cases") or []
    val_cases = split.get("val_cases") or []
    test_cases = split.get("test_cases") or []
    if not train_cases or not val_cases or not test_cases:
        raise ValueError("Source training_config must contain train/val/test case splits")
    if set(train_cases) & set(val_cases) or set(train_cases) & set(test_cases) or set(val_cases) & set(test_cases):
        raise ValueError("Source case split is not disjoint")

    train_dataset = TrusSliceDataset(
        args.trus_roots, train_cases, positive_only=args.positive_only
    )
    val_dataset = TrusSliceDataset(args.trus_roots, val_cases, positive_only=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    configure_prompt_adaptation(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.prompt_encoder.parameters(),
                "lr": args.learning_rate * args.prompt_lr_scale,
            },
            {"params": model.mask_decoder.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    weights = {
        "dice": args.dice_loss_weight,
        "bce": args.bce_loss_weight,
        "iou": args.iou_loss_weight,
    }
    adaptation_config = {
        "source_work_dir": str(source_work_dir),
        "source_checkpoint": str(source_checkpoint),
        "runtime_dir": str(runtime_dir),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "held_out_test_cases": test_cases,
        "args": vars(args),
        "trainable_modules": ["prompt_encoder", "mask_decoder"],
        "frozen_representation": [
            "trus_encoder",
            "student_adapter",
            "student_refinement_adapter",
        ],
        "checkpoint_selection": "harmonic mean of full-image and deterministic GT-box validation Dice",
    }
    exported_config = deepcopy(source_config)
    exported_config["dual_prompt_adaptation"] = adaptation_config
    (output_dir / "training_config.json").write_text(
        json.dumps(exported_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"train slices={len(train_dataset)}, val slices={len(val_dataset)}, "
        f"held-out test cases={len(test_cases)}"
    )
    print(f"trainable parameters={trainable_count:,}/{total_count:,}")

    baseline = validate_dual_prompt(
        model, val_loader, device, args.validation_bbox_shift
    )
    print(f"validation baseline: {baseline}")
    best_score = baseline["harmonic_dice"]
    best_epoch = -1
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        configure_prompt_adaptation(model)
        gt_probability = progressive_probability(
            epoch,
            args.gt_box_probability_start,
            args.gt_box_probability_end,
            args.gt_box_ramp_epochs,
        )
        epoch_rng = np.random.default_rng(args.seed + epoch)
        sample_count = min(args.samples_per_epoch, len(train_dataset))
        indices = epoch_rng.choice(
            len(train_dataset), size=sample_count, replace=False
        ).tolist()
        loader_generator = torch.Generator().manual_seed(args.seed + 10000 + epoch)
        train_loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        box_generator = torch.Generator().manual_seed(args.seed + 20000 + epoch)
        running_loss = 0.0
        running_box_fraction = 0.0
        steps = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["gt"].to(device, non_blocking=True)
            mixed_boxes, uses_gt_box = prompt_boxes_from_masks(
                target,
                gt_box_probability=gt_probability,
                bbox_shift=args.train_bbox_shift,
                generator=box_generator,
            )
            full_boxes = full_image_boxes(
                images.shape[0], images.shape[-2], images.shape[-1], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.mixed_precision):
                features = student_feature(model, images)
                full_logits, full_iou = model._decode(features, full_boxes)
                full_loss, _ = supervised_loss(full_logits, full_iou, target, weights)
                if bool(uses_gt_box.any()):
                    mixed_logits, mixed_iou = model._decode(
                        features, mixed_boxes.to(device)
                    )
                    mixed_loss, _ = supervised_loss(
                        mixed_logits, mixed_iou, target, weights
                    )
                else:
                    mixed_loss = full_loss
                loss = mixed_loss + args.full_image_rehearsal_weight * full_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            running_box_fraction += float(uses_gt_box.float().mean())
            steps += 1

        scheduler.step()
        validation = validate_dual_prompt(
            model, val_loader, device, args.validation_bbox_shift
        )
        record = {
            "epoch": epoch,
            "gt_box_probability": gt_probability,
            "observed_gt_box_fraction": running_box_fraction / max(steps, 1),
            "train_loss": running_loss / max(steps, 1),
            "learning_rate": optimizer.param_groups[-1]["lr"],
            **validation,
        }
        history.append(record)
        write_history(output_dir / "training_history.csv", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.save_latest:
            save_checkpoint(
                output_dir / "dual_prompt_latest.pth",
                model,
                optimizer,
                epoch,
                validation,
                adaptation_config,
            )
        if validation["harmonic_dice"] > best_score + args.min_delta:
            best_score = validation["harmonic_dice"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output_dir / "dual_prompt_best.pth",
                model,
                optimizer,
                epoch,
                validation,
                adaptation_config,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stopping_patience:
            print(f"early stopping at epoch {epoch}")
            break

    summary = {
        "baseline_validation": baseline,
        "best_harmonic_dice": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "adaptation_config": adaptation_config,
    }
    (output_dir / "adaptation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if best_epoch < 0:
        print("No adapted checkpoint exceeded the source validation baseline.")
    return summary


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
