"""Utilities for strict two-stage privileged distillation."""

from __future__ import annotations

from copy import deepcopy

import torch

from prism_checkpoint_utils import normalize_prism_checkpoint_state_dict


STAGES = ("joint", "teacher_pretrain", "student_distill")


class CaseDiceAccumulator:
    def __init__(self):
        self._stats = {}

    def update(self, prediction, target, case_ids):
        prediction = prediction.detach().bool()
        target = target.detach().bool()
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        if len(case_ids) != prediction.shape[0]:
            raise ValueError("case_ids length must match batch size")
        for index, case_id in enumerate(case_ids):
            pred = prediction[index]
            gt = target[index]
            intersection = float(torch.logical_and(pred, gt).sum().item())
            pred_sum = float(pred.sum().item())
            gt_sum = float(gt.sum().item())
            current = self._stats.setdefault(str(case_id), [0.0, 0.0, 0.0])
            current[0] += intersection
            current[1] += pred_sum
            current[2] += gt_sum

    @staticmethod
    def _dice(stats):
        intersection, pred_sum, gt_sum = stats
        denominator = pred_sum + gt_sum
        return 1.0 if denominator == 0 else 2.0 * intersection / denominator

    def per_case(self):
        return {case_id: self._dice(stats) for case_id, stats in self._stats.items()}

    def mean(self):
        values = list(self.per_case().values())
        return float(sum(values) / len(values)) if values else 0.0

    def pooled(self):
        if not self._stats:
            return 0.0
        pooled = [sum(stats[index] for stats in self._stats.values()) for index in range(3)]
        return float(self._dice(pooled))


def _set_module_trainable(module, enabled):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def configure_stage_trainability(model, stage, freeze_encoders=True):
    if stage not in STAGES:
        raise ValueError(f"Unknown distillation stage: {stage}")
    if stage == "joint":
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage == "teacher_pretrain":
        if not freeze_encoders:
            _set_module_trainable(model.trus_encoder, True)
            _set_module_trainable(model.mri_encoder, True)
        _set_module_trainable(model.cross_modal_extractor, True)
    else:
        if not freeze_encoders:
            _set_module_trainable(model.trus_encoder, True)
        _set_module_trainable(model.student_adapter, True)
        _set_module_trainable(model.student_refinement_adapter, True)
        _set_module_trainable(model.student_gate, True)

    _set_module_trainable(model.mask_decoder, True)
    _set_module_trainable(model.prompt_encoder, True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def make_frozen_teacher(student_model, checkpoint, device=None):
    teacher = deepcopy(student_model)
    state_dict = normalize_prism_checkpoint_state_dict(checkpoint)
    teacher.load_state_dict(state_dict, strict=True)
    if device is not None:
        teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def require_effective_mmd(stage, freeze_encoders, mmd_loss_weight):
    if (
        stage in ("teacher_pretrain", "student_distill")
        and freeze_encoders
        and float(mmd_loss_weight) > 0
    ):
        raise ValueError(
            "MMD has no trainable target when both encoders are frozen; set "
            "mmd_loss_weight=0 or unfreeze the encoders."
        )


def offline_student_teacher_output(student, teacher, batch):
    student_output = student.forward_student(batch["trus_image"], batch.get("bboxes"))
    with torch.no_grad():
        teacher_output = teacher.forward_teacher(
            batch["trus_image"],
            batch["mri_image"],
            boxes=batch.get("bboxes"),
            mask_gt=batch.get("gt2D"),
            mri_valid_mask=batch.get("mri_valid_mask"),
            transition_target=batch.get("transition_target"),
            transition_label=batch.get("transition_label"),
            transition_valid=batch.get("transition_valid"),
            relative_depth=batch.get("relative_depth"),
        )
    return {**teacher_output, **student_output}
