import numpy as np
import torch

from argparse import Namespace

from tools.run_ssca_experiments import build_command, normalize_experiment, selected_experiments
from tiny_vit_sam import CrossModalFeatureExtractor, DepthGatedAnatomicalMMD, DepthPriorMLP
from utils.paired_dataset import PairedNpyDataset


def test_ablation_modes_emit_valid_attention_diagnostics():
    torch.manual_seed(0)
    extractor = CrossModalFeatureExtractor(in_channels=16, num_heads=4)
    trus = torch.randn(2, 16, 4, 4)
    mri = torch.randn(2, 3, 16, 4, 4)

    modes = [
        "original_index",
        "random_neighbor_sampling",
        "average_neighbor_fusion",
        "ssca_no_entropy",
        "ssca_entropy",
        "center_only_with_same_params",
    ]
    for mode in modes:
        fused, mmd = extractor(trus, mri, slice_attention_mode=mode)
        assert fused.shape == trus.shape
        assert torch.isfinite(mmd)
        info = extractor.last_slice_attention
        assert info is not None
        beta = info["beta"]
        assert torch.allclose(beta.sum(dim=1), torch.ones(beta.size(0)), atol=1e-5)
        if "entropy" in info:
            assert torch.all((info["entropy"] >= 0) & (info["entropy"] <= 1))
        if "confidence" in info:
            assert torch.all((info["confidence"] >= 0) & (info["confidence"] <= 1))


def test_ssca_slice_correspondence_prior_provides_beta_learning_signal():
    torch.manual_seed(3)
    extractor = CrossModalFeatureExtractor(
        in_channels=16,
        num_heads=4,
        slice_corr_loss_weight=0.5,
        slice_corr_prior_sigma=0.75,
    )
    trus = torch.randn(2, 16, 4, 4)
    mri = torch.randn(2, 3, 16, 4, 4)

    fused, loss = extractor(trus, mri, slice_attention_mode="ssca_entropy")
    info = extractor.last_slice_attention

    assert fused.shape == trus.shape
    assert "slice_corr_loss" in info
    assert torch.isfinite(info["slice_corr_loss"])
    assert info["slice_corr_loss"] > 0
    assert torch.isfinite(loss)


def test_dynamic_bandwidth_cross_modal_forward_returns_finite_loss():
    torch.manual_seed(4)
    extractor = CrossModalFeatureExtractor(
        in_channels=16,
        num_heads=4,
        use_dynamic_bandwidth_beta=True,
        dynamic_beta_mode="prior_only",
        use_neighbor_residual_fusion=True,
        use_gt_transition_for_beta=True,
        transition_loss_weight=1.0,
    )
    trus = torch.randn(2, 16, 4, 4)
    mri = torch.randn(2, 5, 16, 4, 4)
    transition_target = torch.tensor([0.0, 1.0])
    transition_label = torch.tensor([0.0, 1.0])

    fused, loss = extractor(
        trus,
        mri,
        slice_attention_mode="ssca_entropy",
        transition_target=transition_target,
        transition_label=transition_label,
    )

    info = extractor.last_slice_attention
    assert fused.shape == trus.shape
    assert torch.isfinite(loss)
    assert torch.allclose(info["beta"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert info["sigma"][1] < info["sigma"][0]


def _write_case(root, modality_prefix, case_num, num_slices):
    img_dir = root / "imgs"
    gt_dir = root / "gts"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for slice_idx in range(num_slices):
        name = f"{modality_prefix}_Prostate_case{case_num:06d}-{slice_idx:03d}.npy"
        image = np.zeros((8, 8, 3), dtype=np.float32)
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[2:6, 2:6] = 1
        np.save(img_dir / name, image)
        np.save(gt_dir / name, gt)


def _make_pairing_dataset(tmp_path, pairing_mode, radius=1):
    trus_root = tmp_path / "trus" / "channel_0"
    mri_root = tmp_path / "mri" / "channel_0"
    for case_num in (0, 1):
        _write_case(trus_root, "TRUS", case_num, 5)
        _write_case(mri_root, "MRI", case_num, 5)
    return PairedNpyDataset(
        str(trus_root),
        str(mri_root),
        mri_window_radius=radius,
        slice_attention_mode="ssca_entropy",
        pairing_mode=pairing_mode,
    )


def test_pairing_modes_shift_and_pad_without_out_of_bounds(tmp_path):
    ds = _make_pairing_dataset(tmp_path, "shift_plus_3", radius=1)
    sample = ds[0]
    indices = sample["mri_window_indices"].tolist()
    assert len(indices) == 3
    assert all(0 <= idx <= 4 for idx in indices)
    assert indices[-1] == 4
    assert float(sample["relative_depth"]) == 0.0
    assert int(sample["slice_index"]) == 0
    assert int(sample["num_slices"]) == 5


def test_different_patient_random_never_uses_same_patient(tmp_path):
    ds = _make_pairing_dataset(tmp_path, "different_patient_random", radius=1)
    for idx in range(len(ds)):
        sample = ds[idx]
        assert sample["mri_pairing_case_id"] != sample["case_id"]
        assert len(sample["mri_window_indices"]) == 3


def test_depth_prior_mlp_outputs_valid_rho_and_weak_prior_loss():
    module = DepthPriorMLP(embed_dim=8, hidden_dim=16)
    depth = torch.tensor([0.0, 0.5, 1.0])

    depth_embed, rho, prior_loss, u_depth = module(depth)

    assert depth_embed.shape == (3, 8)
    assert rho.shape == (3,)
    assert torch.all((rho >= 0) & (rho <= 1))
    assert torch.allclose(u_depth, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.isfinite(prior_loss)


def test_depth_gated_anatomical_mmd_returns_finite_components():
    torch.manual_seed(5)
    module = DepthGatedAnatomicalMMD(in_channels=16, project_dim=8)
    trus = torch.randn(3, 16, 4, 4)
    center = torch.randn(3, 16, 4, 4)
    privileged = torch.randn(3, 16, 4, 4)
    rho = torch.tensor([0.0, 0.5, 1.0])
    alpha = torch.tensor([0.5, 0.4, 0.3])

    loss, info = module(trus, center, privileged, rho=rho, alpha=alpha)

    assert torch.isfinite(loss)
    assert torch.isfinite(info["mmd_center"])
    assert torch.isfinite(info["mmd_priv"])
    assert torch.all((info["mmd_priv_weight"] >= 0.2) & (info["mmd_priv_weight"] <= 1.0))


def test_depth_gate_and_beta_modulation_keep_fixed_prior_as_backbone():
    torch.manual_seed(6)
    extractor = CrossModalFeatureExtractor(
        in_channels=16,
        num_heads=4,
        ssca_position_prior_weight=0.3,
        ssca_position_prior_sigma=0.75,
        depth_prior_enabled=True,
        depth_gate_enabled=True,
        depth_gate_alpha_min=0.05,
        depth_gate_alpha_max=0.60,
        beta_modulation_enabled=True,
        beta_modulation_mix_max=0.2,
    )
    trus = torch.randn(2, 16, 4, 4)
    mri = torch.randn(2, 5, 16, 4, 4)
    relative_depth = torch.tensor([0.5, 1.0])

    fused, loss = extractor(
        trus,
        mri,
        slice_attention_mode="ssca_entropy",
        relative_depth=relative_depth,
    )
    info = extractor.last_slice_attention

    assert fused.shape == trus.shape
    assert torch.isfinite(loss)
    assert torch.all((info["rho"] >= 0) & (info["rho"] <= 1))
    assert torch.all((info["fusion_gate"].flatten() >= 0.05) & (info["fusion_gate"].flatten() <= 0.60))
    assert torch.allclose(info["beta"].sum(dim=1), torch.ones(2), atol=1e-6)
    uniform_beta = torch.full((2, 5), 0.2)
    modulated = extractor.ssca._depth_modulate_beta(uniform_beta, torch.tensor([0.0, 1.0]))
    assert modulated[1, 2] > modulated[0, 2]


def _runner_args(tmp_path):
    return Namespace(
        python="python",
        results_root=str(tmp_path / "results"),
        feature_cache_root=str(tmp_path / "cache"),
        trus_data_root="trus",
        mri_data_root="mri",
        val_trus_data_root="val_trus",
        val_mri_data_root="val_mri",
        ssca_descriptor_dim=128,
        ssca_beta_temperature=0.1,
        ssca_min_confidence=0.2,
        ssca_position_prior_weight=0.0,
        ssca_position_prior_sigma=0.75,
        ssca_max_window_size=7,
        slice_corr_loss_weight=0.0,
        slice_corr_prior_sigma=0.75,
        transition_cls_loss_weight=0.05,
        transition_reg_loss_weight=0.01,
        sigma_min=0.30,
        sigma_max=1.25,
        dynamic_bandwidth_use_gt_transition_prob=0.0,
        dynamic_bandwidth_warmup_epochs=0,
        diagnostic_checkpoint_epochs="1,3,6,8,12",
        batch_size=2,
        samples_per_epoch=512,
        num_epochs=100,
        lr=0.00054900441949227454,
        weight_decay=8.088464292618784e-05,
        iou_loss_weight=1.038183443069801,
        seg_loss_weight=0.6536756603510201,
        ce_loss_weight=1.2138877941863124,
        mmd_loss_weight=0.050427937624551035,
        depth_embed_dim=32,
        depth_prior_hidden_dim=64,
        lambda_depth_prior=0.01,
        depth_prior_no_u_shape_regularization=False,
        depth_gate_alpha_min=0.05,
        depth_gate_alpha_max=0.60,
        dg_mmd_project_dim=128,
        dg_mmd_lambda_center=0.005,
        dg_mmd_lambda_priv=0.005,
        dg_mmd_min_priv_weight=0.2,
        beta_modulation_mix_max=0.0,
        bbox_shift=3,
        early_stopping_patience=10,
        min_delta=0.0061295169501196555,
        lr_scheduler="cosine",
        trus_pretrained_checkpoint="trus_best.pth",
        mri_pretrained_checkpoint="mri_best.pth",
        slice_attention_log_interval=20,
        final_results_checkpoint="best_2d",
        mixed_precision=True,
    )


def test_main_group_selects_fixed_prior_r2_only():
    experiments = selected_experiments("main")

    assert len(experiments) == 1
    name, ablation_mode, radius, pairing_mode, extra = normalize_experiment(experiments[0])
    assert name == "E1_fixed_prior_r2_main"
    assert ablation_mode == "ssca_entropy"
    assert radius == 2
    assert pairing_mode == "normal"
    assert "-use_dynamic_bandwidth_beta" not in extra
    assert "-use_neighbor_residual_fusion" not in extra
    assert "-use_gt_transition_for_beta" not in extra
    assert "-use_gt_transition_for_neighbor_trust" not in extra
    assert extra[extra.index("-ssca_position_prior_weight") + 1] == "0.3"
    assert extra[extra.index("-ssca_position_prior_sigma") + 1] == "0.75"
    assert extra[extra.index("-transition_loss_weight") + 1] == "0.0"


def test_main_group_command_keeps_dynamic_and_oracle_modes_disabled(tmp_path):
    exp = normalize_experiment(selected_experiments("main")[0])
    cmd = build_command(_runner_args(tmp_path), *exp)

    assert "-mri_window_radius" in cmd
    assert cmd[cmd.index("-mri_window_radius") + 1] == "2"
    assert cmd[cmd.index("-ssca_position_prior_weight") + 1] == "0.0"
    assert cmd[cmd.index("-ssca_position_prior_weight", cmd.index("-ssca_position_prior_weight") + 1) + 1] == "0.3"
    assert cmd[cmd.index("-ssca_position_prior_sigma", cmd.index("-ssca_position_prior_sigma") + 1) + 1] == "0.75"
    assert cmd[cmd.index("-transition_loss_weight") + 1] == "0.0"
    assert "-use_dynamic_bandwidth_beta" not in cmd
    assert "-use_neighbor_residual_fusion" not in cmd
    assert "-use_gt_transition_for_beta" not in cmd
    assert "-use_gt_transition_for_neighbor_trust" not in cmd
    assert "--cache_encoder_features" in cmd
    assert "--generate_final_results" in cmd


def test_dgmmd_group_defines_controlled_d0_to_d4_matrix(tmp_path):
    experiments = [normalize_experiment(exp) for exp in selected_experiments("dgmmd")]
    names = [exp[0] for exp in experiments]

    assert names == [
        "D0_fixed_prior_r2",
        "D1_fixed_prior_r2_depth_gate",
        "D2_fixed_prior_r2_dg_mmd",
        "D3_fixed_prior_r2_depth_gate_dg_mmd",
        "D4_fixed_prior_r2_depth_gate_dg_mmd_beta_mod",
    ]
    forbidden = {
        "-use_dynamic_bandwidth_beta",
        "-use_gt_transition_for_beta",
        "-use_gt_transition_for_neighbor_trust",
        "-use_neighbor_residual_fusion",
        "-use_transition_aware_beta",
    }
    for name, ablation_mode, radius, pairing_mode, extra in experiments:
        assert ablation_mode == "ssca_entropy"
        assert radius == 2
        assert pairing_mode == "normal"
        assert forbidden.isdisjoint(extra)
        assert extra[extra.index("-ssca_position_prior_weight") + 1] == "0.3"
        assert extra[extra.index("-transition_loss_weight") + 1] == "0.0"

    d4_cmd = build_command(_runner_args(tmp_path), *experiments[-1])
    assert "-depth_gate_enabled" in d4_cmd
    assert "-use_depth_gated_mmd" in d4_cmd
    assert "-beta_modulation_enabled" in d4_cmd
    assert d4_cmd[d4_cmd.index("-mri_window_radius") + 1] == "2"
    assert d4_cmd[d4_cmd.index("-beta_modulation_mix_max", d4_cmd.index("-beta_modulation_mix_max") + 1) + 1] == "0.2"
