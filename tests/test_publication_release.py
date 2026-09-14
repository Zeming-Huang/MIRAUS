import json
from pathlib import Path

import torch
import torch.nn as nn


class _IdentityModule(nn.Module):
    def forward(self, *args, **kwargs):
        if args:
            return args[0]
        return None


def test_student_uses_zero_initialized_refinement_adapter():
    from enhanced_dual_modal import EnhancedDualModalMedSAM_Lite

    model = EnhancedDualModalMedSAM_Lite(
        image_encoder=nn.Conv2d(3, 256, kernel_size=1),
        mask_decoder=_IdentityModule(),
        prompt_encoder=_IdentityModule(),
        use_cross_modal=False,
        use_student_refinement_adapter=True,
    )

    assert model.student_refinement_adapter is not None
    final_layer = model.student_refinement_adapter[-1]
    assert torch.count_nonzero(final_layer.weight) == 0
    assert torch.count_nonzero(final_layer.bias) == 0

    trus_feature = torch.randn(1, 256, 8, 8)
    student_feature, _, _, predicted_residual = model._student_feature(
        trus_feature,
        return_details=True,
    )
    base_residual = model.student_adapter(trus_feature)
    assert torch.allclose(predicted_residual, base_residual)
    assert torch.allclose(student_feature, trus_feature + base_residual)


def test_privileged_loss_matches_teacher_defined_increment():
    from privileged_distillation import compute_privileged_feature_loss

    torch.manual_seed(7)
    trus_feature = torch.randn(2, 8, 4, 4)
    predicted_residual = torch.randn_like(trus_feature)
    target_residual = torch.randn_like(trus_feature)
    student_feature = trus_feature + predicted_residual
    teacher_feature = trus_feature + target_residual

    losses = compute_privileged_feature_loss(
        student_feat=student_feature,
        teacher_feat=teacher_feature,
        trus_feat=trus_feature,
        predicted_residual=predicted_residual,
        use_roi=False,
        cosine_weight=0.0,
        residual_weight=0.0,
    )

    assert torch.allclose(
        losses["feature_smooth_l1"],
        losses["residual_smooth_l1"],
        atol=1e-7,
    )
    assert not teacher_feature.requires_grad


def test_student_distill_stage_freezes_privileged_teacher_modules():
    from prism_offline_distillation import configure_stage_trainability

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.trus_encoder = nn.Linear(2, 2)
            self.mri_encoder = nn.Linear(2, 2)
            self.cross_modal_extractor = nn.Linear(2, 2)
            self.student_adapter = nn.Linear(2, 2)
            self.student_refinement_adapter = nn.Linear(2, 2)
            self.student_gate = None
            self.mask_decoder = nn.Linear(2, 2)
            self.prompt_encoder = nn.Linear(2, 2)

    model = DummyModel()
    configure_stage_trainability(model, "student_distill", freeze_encoders=True)

    assert not any(p.requires_grad for p in model.trus_encoder.parameters())
    assert not any(p.requires_grad for p in model.mri_encoder.parameters())
    assert not any(p.requires_grad for p in model.cross_modal_extractor.parameters())
    assert all(p.requires_grad for p in model.student_adapter.parameters())
    assert all(p.requires_grad for p in model.student_refinement_adapter.parameters())
    assert all(p.requires_grad for p in model.mask_decoder.parameters())
    assert all(p.requires_grad for p in model.prompt_encoder.parameters())


def test_candidate_utility_head_receives_direct_segmentation_gradient():
    from tiny_vit_sam import CrossModalFeatureExtractor

    torch.manual_seed(11)
    extractor = CrossModalFeatureExtractor(
        in_channels=16,
        num_heads=4,
        use_fusion=False,
        ssca_max_window_size=5,
        slice_utility_loss_weight=1.0,
        candidate_seg_loss_weight=1.0,
    )
    trus_feature = torch.randn(2, 16, 4, 4)
    mri_features = torch.randn(2, 5, 16, 4, 4)
    masks = (torch.rand(2, 1, 4, 4) > 0.5).float()
    boxes = torch.tensor(
        [[[0.0, 0.0, 255.0, 255.0]], [[0.0, 0.0, 255.0, 255.0]]]
    )

    _, _, auxiliary_loss = extractor(
        trus_feature,
        mri_features,
        return_loss=True,
        slice_attention_mode="ssca_entropy",
        boxes=boxes,
        image_hw=(256, 256),
        mask_gt=masks,
        mri_valid_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    auxiliary_loss.backward()

    gradients = [
        parameter.grad for parameter in extractor.ssca.candidate_head.parameters()
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_public_fivefold_split_is_disjoint_and_complete():
    split_path = Path(__file__).parents[1] / "configs" / "muregpro_5fold_splits.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    folds = payload["folds"]

    assert len(folds) == 5
    held_out = [patient for fold in folds for patient in fold["test"]]
    assert len(held_out) == 73
    assert len(set(held_out)) == 73
    assert set(held_out) == set(payload["patients"])
    for fold in folds:
        assert set(fold["validation"]).isdisjoint(fold["test"])
        assert len(set(payload["patients"]) - set(fold["validation"]) - set(fold["test"])) in (52, 53)


def test_prompt_parser_supports_full_image_and_explicit_box():
    from inference_single_frame import resolve_prompt_box

    assert resolve_prompt_box(None, image_width=640, image_height=480) == (
        0.0,
        0.0,
        639.0,
        479.0,
    )
    assert resolve_prompt_box([10, 20, 310, 260], 640, 480) == (
        10.0,
        20.0,
        310.0,
        260.0,
    )


def test_prompt_parser_rejects_invalid_box():
    from inference_single_frame import resolve_prompt_box

    try:
        resolve_prompt_box([50, 20, 10, 100], 640, 480)
    except ValueError as error:
        assert "x1" in str(error)
        return
    raise AssertionError("Expected invalid box coordinates to raise ValueError")


def test_scaled_prompt_has_sam_box_shape():
    from inference_single_frame import scale_box_to_model

    scaled = scale_box_to_model((0.0, 0.0, 639.0, 479.0), 640, 480, "cpu")
    assert scaled.shape == (1, 1, 4)
    assert torch.allclose(scaled[0, 0], torch.tensor([0.0, 0.0, 255.0, 255.0]))
