"""Losses for transferring MRI-guided privileged features to a TRUS-only student."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _strip_common_state_prefix(key):
    for _ in range(4):
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            break
    return key


def extract_model_state_dict(checkpoint_or_state_dict):
    """Return a normalized model state dict from common checkpoint formats."""
    state_dict = checkpoint_or_state_dict
    if isinstance(state_dict.get("model"), dict):
        state_dict = state_dict["model"]
    elif isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict.get("model_state_dict"), dict):
        state_dict = state_dict["model_state_dict"]
    return {
        _strip_common_state_prefix(key): value
        for key, value in state_dict.items()
    }


def infer_student_architecture_from_state_dict(state_dict):
    """Infer optional student modules before constructing a deployment model."""
    normalized_state = extract_model_state_dict(state_dict)
    adapter_weight = normalized_state.get("student_adapter.0.weight")
    if adapter_weight is None:
        adapter_weight = normalized_state.get("student_refinement_adapter.0.weight")
    hidden = int(adapter_weight.shape[0]) if adapter_weight is not None else 256
    architecture = {
        "student_adapter_hidden": hidden,
        "use_gated_student_adapter": any(
            key.startswith("student_gate.") for key in normalized_state
        ),
        "use_student_refinement_adapter": any(
            key.startswith("student_refinement_adapter.") for key in normalized_state
        ),
    }
    descriptor_weight = normalized_state.get(
        "cross_modal_extractor.ssca.trus_descriptor.0.weight"
    )
    if descriptor_weight is not None:
        descriptor_input_dim = int(descriptor_weight.shape[1])
        if descriptor_input_dim not in (256, 768):
            raise ValueError(
                "Unsupported SSCA descriptor input dimension in checkpoint: "
                f"{descriptor_input_dim}"
            )
        architecture.update(
            {
                "ssca_descriptor_dim": int(descriptor_weight.shape[0]),
                "ssca_use_box_aware_pooling": descriptor_input_dim == 768,
            }
        )
    return architecture


def set_student_only_trainable(model):
    """Freeze the privileged teacher and expose only student adaptation modules."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_modules = []
    refinement = getattr(model, "student_refinement_adapter", None)
    module_names = (
        ("student_refinement_adapter", "student_gate")
        if refinement is not None
        else ("student_adapter", "student_gate")
    )
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
        trainable_modules.append(name)
    if not trainable_modules:
        raise ValueError("Model has no student adapter or gate to fine-tune")
    return trainable_modules


def binary_iou(result, reference):
    """Compute binary IoU per sample, treating two empty masks as perfect."""
    intersection = torch.count_nonzero(
        torch.logical_and(result, reference),
        dim=[i for i in range(1, result.ndim)],
    )
    union = torch.count_nonzero(
        torch.logical_or(result, reference),
        dim=[i for i in range(1, result.ndim)],
    )
    intersection = intersection.float()
    union = union.float()
    iou = torch.where(
        union == 0,
        torch.ones_like(union),
        intersection / union.clamp_min(1.0),
    )
    return iou.unsqueeze(1)


def primary_logits_from_model_output(model_output):
    """Return student logits from either a legacy tuple or a MIRAUS output dictionary."""
    if isinstance(model_output, dict):
        return model_output["logits_student"]
    return model_output[0]


def compute_teacher_correctness_weight_map(
    logits_teacher,
    logits_student,
    gt,
    gamma_max=5.0,
    min_weight=0.0,
    gain_margin=0.0,
    foreground_boost=1.0,
    boundary_boost=1.0,
    boundary_radius=1,
    teacher_confidence_min=0.0,
    teacher_confidence_power=0.0,
    slice_confidence=None,
):
    """Weight distillation where the teacher is correct and better than student."""
    with torch.no_grad():
        gt_f = gt.to(device=logits_teacher.device, dtype=logits_teacher.dtype)
        teacher_pixel_loss = F.binary_cross_entropy_with_logits(
            logits_teacher.detach(), gt_f, reduction="none"
        )
        student_pixel_loss = F.binary_cross_entropy_with_logits(
            logits_student.detach(), gt_f, reduction="none"
        )
        teacher_correct = (
            (torch.sigmoid(logits_teacher.detach()) > 0.5) == (gt_f > 0.5)
        ).to(logits_teacher.dtype)
        teacher_prob = torch.sigmoid(logits_teacher.detach())
        teacher_confidence = (teacher_prob - 0.5).abs() * 2.0
        teacher_confidence_min = min(max(float(teacher_confidence_min), 0.0), 1.0)
        if teacher_confidence_min > 0.0:
            confident_teacher = (teacher_confidence >= teacher_confidence_min).to(
                logits_teacher.dtype
            )
            teacher_correct = teacher_correct * confident_teacher
        gain_margin = max(float(gain_margin), 0.0)
        teacher_gain = (student_pixel_loss - teacher_pixel_loss - gain_margin).clamp_min(0.0)
        teacher_confidence_power = max(float(teacher_confidence_power), 0.0)
        if teacher_confidence_power > 0.0:
            teacher_gain = teacher_gain * teacher_confidence.clamp_min(1e-6).pow(
                teacher_confidence_power
            )
        foreground_boost = max(float(foreground_boost), 1.0)
        if foreground_boost > 1.0:
            foreground_weight = torch.where(
                gt_f > 0.5,
                torch.full_like(gt_f, foreground_boost),
                torch.ones_like(gt_f),
            )
            teacher_gain = teacher_gain * foreground_weight
        boundary_boost = max(float(boundary_boost), 1.0)
        if boundary_boost > 1.0:
            fg = (gt_f > 0.5).to(dtype=gt_f.dtype)
            radius = max(int(boundary_radius), 0)
            kernel_size = 2 * radius + 1
            dilated = F.max_pool2d(fg, kernel_size=kernel_size, stride=1, padding=radius)
            eroded = -F.max_pool2d(-fg, kernel_size=kernel_size, stride=1, padding=radius)
            boundary = (dilated - eroded).clamp(0.0, 1.0)
            boundary_weight = torch.where(
                boundary > 0.0,
                torch.full_like(gt_f, boundary_boost),
                torch.ones_like(gt_f),
            )
            teacher_gain = teacher_gain * boundary_weight
        gamma = teacher_gain * teacher_correct
        gamma = gamma / gamma.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        gamma = gamma.clamp(0.0, max(float(gamma_max), 0.0))
        min_weight = min(max(float(min_weight), 0.0), 1.0)
        if min_weight > 0:
            improved_and_correct = (teacher_gain > 0).to(logits_teacher.dtype) * teacher_correct
            gamma = torch.where(improved_and_correct > 0, gamma.clamp_min(min_weight), gamma)
        if slice_confidence is not None:
            confidence = slice_confidence.to(device=gamma.device, dtype=gamma.dtype).view(-1, 1, 1, 1)
            gamma = gamma * confidence.clamp(0.0, 1.0)
        return gamma.detach()


def _box_mask(boxes, feature_hw, image_hw, device, dtype):
    batch = boxes.shape[0]
    feat_h, feat_w = feature_hw
    image_h, image_w = image_hw
    mask = torch.zeros((batch, 1, feat_h, feat_w), device=device, dtype=dtype)
    flat_boxes = boxes.reshape(batch, -1, 4)

    for batch_idx in range(batch):
        for box in flat_boxes[batch_idx]:
            x0, y0, x1, y1 = [float(value) for value in box]
            left = max(0, min(feat_w, math.floor(x0 * feat_w / image_w)))
            top = max(0, min(feat_h, math.floor(y0 * feat_h / image_h)))
            right = max(left + 1, min(feat_w, math.ceil((x1 + 1.0) * feat_w / image_w)))
            bottom = max(top + 1, min(feat_h, math.ceil((y1 + 1.0) * feat_h / image_h)))
            mask[batch_idx, :, top:bottom, left:right] = 1
    return mask


def _masked_mean(values, mask, preserve_weight_scale=False, support_mask=None):
    if mask is None:
        return values.mean()
    expanded = mask.expand_as(values)
    if preserve_weight_scale:
        if support_mask is None:
            return (values * expanded).sum() / values.numel()
        support = support_mask.to(device=values.device, dtype=values.dtype).expand_as(values)
        return (values * expanded).sum() / support.sum().clamp_min(1.0)
    return (values * expanded).sum() / expanded.sum().clamp_min(1e-6)


def reliability_weighted_mean(values, weight, preserve_weight_scale=False):
    """Average a loss map with reliability weights.

    The default is a normalized weighted average. When preserve_weight_scale is
    enabled, lower mean reliability also lowers the returned loss magnitude.
    """
    if weight is None:
        return values.mean()
    expanded = weight.to(device=values.device, dtype=values.dtype).expand_as(values)
    if preserve_weight_scale:
        return (values * expanded).sum() / values.numel()
    return (values * expanded).sum() / expanded.sum().clamp_min(1e-6)


def _binary_boundary_band(gt, radius):
    radius = max(int(radius), 0)
    binary = (gt > 0.5).to(dtype=gt.dtype)
    if radius == 0:
        return binary
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(binary, kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel_size, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def compute_boundary_weight_map(gt, boundary_radius=3):
    with torch.no_grad():
        return _binary_boundary_band(gt, boundary_radius).detach()


def compute_slice_gain_boundary_weight_map(
    logits_teacher,
    logits_student,
    gt,
    boundary_radius=3,
    gain_margin=0.0,
):
    with torch.no_grad():
        gt_f = gt.to(device=logits_teacher.device, dtype=logits_teacher.dtype)
        boundary = _binary_boundary_band(gt_f, boundary_radius)
        support = boundary.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        teacher_loss = F.binary_cross_entropy_with_logits(
            logits_teacher.detach(), gt_f, reduction="none"
        )
        student_loss = F.binary_cross_entropy_with_logits(
            logits_student.detach(), gt_f, reduction="none"
        )
        teacher_boundary_loss = (teacher_loss * boundary).sum(
            dim=(1, 2, 3), keepdim=True
        ) / support
        student_boundary_loss = (student_loss * boundary).sum(
            dim=(1, 2, 3), keepdim=True
        ) / support
        gain = student_boundary_loss - teacher_boundary_loss
        slice_gate = (gain > max(float(gain_margin), 0.0)).to(boundary.dtype)
        return (boundary * slice_gate).detach()


def compute_slice_gain_full_weight_map(
    logits_teacher,
    logits_student,
    gt,
    gain_margin=0.0,
):
    with torch.no_grad():
        gt_f = gt.to(device=logits_teacher.device, dtype=logits_teacher.dtype)
        teacher_loss = F.binary_cross_entropy_with_logits(
            logits_teacher.detach(), gt_f, reduction="none"
        ).mean(dim=(1, 2, 3), keepdim=True)
        student_loss = F.binary_cross_entropy_with_logits(
            logits_student.detach(), gt_f, reduction="none"
        ).mean(dim=(1, 2, 3), keepdim=True)
        slice_gate = (
            student_loss - teacher_loss > max(float(gain_margin), 0.0)
        ).to(gt_f.dtype)
        return torch.ones_like(gt_f) * slice_gate


def compute_privileged_feature_loss(
    student_feat,
    teacher_feat,
    trus_feat,
    predicted_residual,
    boxes=None,
    image_hw=(256, 256),
    use_roi=True,
    spatial_weight=None,
    cosine_weight=0.25,
    residual_weight=1.0,
    preserve_weight_scale=False,
):
    """Return ROI-aware feature and residual hallucination losses.

    The teacher is always detached. The residual target is the MRI-guided change
    relative to the TRUS encoder feature, rather than the MRI feature itself.
    """
    teacher_target = teacher_feat.detach()
    residual_target = teacher_target - trus_feat.detach()
    mask = None
    support_mask = None
    if use_roi and boxes is not None:
        mask = _box_mask(
            boxes.detach(),
            student_feat.shape[-2:],
            image_hw,
            student_feat.device,
            student_feat.dtype,
        )
        support_mask = mask
    if spatial_weight is not None:
        weight = spatial_weight.detach().to(device=student_feat.device, dtype=student_feat.dtype)
        if weight.shape[-2:] != student_feat.shape[-2:]:
            weight = F.interpolate(
                weight,
                size=student_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if mask is None:
            mask = weight
        else:
            mask = mask * weight

    feature_smooth_l1 = _masked_mean(
        F.smooth_l1_loss(student_feat, teacher_target, reduction="none"),
        mask,
        preserve_weight_scale=preserve_weight_scale,
        support_mask=support_mask,
    )
    residual_smooth_l1 = _masked_mean(
        F.smooth_l1_loss(predicted_residual, residual_target, reduction="none"),
        mask,
        preserve_weight_scale=preserve_weight_scale,
        support_mask=support_mask,
    )
    cosine_map = 1.0 - F.cosine_similarity(student_feat, teacher_target, dim=1, eps=1e-6)
    active_direction = (
        student_feat.pow(2).sum(dim=1) + teacher_target.pow(2).sum(dim=1)
    ) > 1e-12
    cosine_map = torch.where(active_direction, cosine_map, torch.zeros_like(cosine_map))
    cosine = _masked_mean(
        cosine_map[:, None],
        mask,
        preserve_weight_scale=preserve_weight_scale,
        support_mask=support_mask,
    )
    total = feature_smooth_l1 + float(cosine_weight) * cosine
    total = total + float(residual_weight) * residual_smooth_l1
    return {
        "total": total,
        "feature_smooth_l1": feature_smooth_l1,
        "residual_smooth_l1": residual_smooth_l1,
        "cosine": cosine,
        "weight_mean": mask.mean().detach() if mask is not None else student_feat.new_tensor(1.0),
    }
