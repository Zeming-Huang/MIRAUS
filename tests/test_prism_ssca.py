import numpy as np
import torch


def _write_slice(root, subdir, name, value):
    path = root / subdir
    path.mkdir(parents=True, exist_ok=True)
    if subdir == "imgs":
        arr = np.full((4, 4, 3), value, dtype=np.float32)
    else:
        arr = np.ones((4, 4), dtype=np.uint8)
    np.save(path / name, arr)


def test_paired_dataset_returns_normalized_mri_window_with_replicate_padding(tmp_path):
    from utils.paired_dataset import PairedNpyDataset

    trus_root = tmp_path / "trus" / "channel_0"
    mri_root = tmp_path / "mri" / "channel_0"

    for idx in range(5):
        name = f"TRUS_Prostate_case000001-{idx:03d}.npy"
        _write_slice(trus_root, "imgs", name, 0.1 + idx * 0.1)
        _write_slice(trus_root, "gts", name, 1)

    for idx in range(3):
        name = f"MRI_Prostate_case000001-{10 + idx:03d}.npy"
        _write_slice(mri_root, "imgs", name, 0.2 + idx * 0.1)
        _write_slice(mri_root, "gts", name, 1)

    dataset = PairedNpyDataset(
        trus_root,
        mri_root,
        mri_window_radius=1,
        slice_attention_mode="ssca_entropy",
        data_aug=False,
    )

    first = dataset[0]
    last = dataset[4]

    assert first["mri_image"].shape == (3, 3, 4, 4)
    assert first["case_id"] == "case000001"
    assert first["trus_slice_index"].item() == 0
    assert first["mri_center_index"].item() == 10
    assert first["mri_window_indices"].tolist() == [10, 10, 11]

    assert last["trus_slice_index"].item() == 4
    assert last["mri_center_index"].item() == 12
    assert last["mri_window_indices"].tolist() == [11, 12, 12]


def test_soft_slice_correspondence_attention_outputs_diagnostics():
    from tiny_vit_sam import SoftSliceCorrespondenceAttention

    torch.manual_seed(7)
    module = SoftSliceCorrespondenceAttention(in_channels=4, num_heads=2)
    module.eval()

    trus_feat = torch.randn(2, 4, 3, 3)
    mri_set = torch.randn(2, 3, 4, 3, 3)

    with torch.no_grad():
        fused, privileged, info = module(trus_feat, mri_set)
    beta = info["beta"]
    entropy = info["entropy"]
    confidence = info["confidence"]
    gate = info["fusion_gate"]

    assert fused.shape == trus_feat.shape
    assert privileged.shape == trus_feat.shape
    assert beta.shape == (2, 3)
    assert entropy.shape == (2,)
    assert confidence.shape == (2,)
    assert gate.shape == (2, 1, 1, 1)
    assert torch.allclose(beta.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all((entropy >= 0) & (entropy <= 1))
    assert torch.all(confidence >= module.min_confidence)
    assert torch.all(confidence <= 1.0)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_ssca_temperature_projection_and_min_confidence_keep_mri_branch_trainable():
    from tiny_vit_sam import SoftSliceCorrespondenceAttention

    module = SoftSliceCorrespondenceAttention(
        in_channels=4,
        num_heads=2,
        descriptor_dim=6,
        beta_temperature=0.25,
        min_confidence=0.2,
    )
    module.eval()

    trus_feat = torch.ones(1, 4, 3, 3)
    mri_set = torch.ones(1, 3, 4, 3, 3)

    with torch.no_grad():
        _, _, info = module(trus_feat, mri_set)
    beta = info["beta"]
    entropy = info["entropy"]
    confidence = info["confidence"]
    gate = info["fusion_gate"]

    assert hasattr(module, "trus_descriptor")
    assert hasattr(module, "mri_descriptor")
    assert torch.allclose(beta, torch.full_like(beta, 1 / 3), atol=1e-6)
    assert torch.allclose(entropy, torch.ones_like(entropy), atol=1e-6)
    assert torch.allclose(confidence, torch.full_like(confidence, 0.2), atol=1e-6)
    assert torch.all(gate > 0)


def test_ssca_position_prior_breaks_uniform_beta_for_ambiguous_neighbors():
    from tiny_vit_sam import SoftSliceCorrespondenceAttention

    module = SoftSliceCorrespondenceAttention(
        in_channels=4,
        num_heads=2,
        beta_temperature=1.0,
        min_confidence=0.0,
        position_prior_weight=1.0,
        position_prior_sigma=0.75,
    )
    module.eval()

    trus_feat = torch.ones(1, 4, 3, 3)
    mri_set = torch.ones(1, 3, 4, 3, 3)

    with torch.no_grad():
        _, _, info = module(trus_feat, mri_set)
    beta = info["beta"]
    entropy = info["entropy"]
    confidence = info["confidence"]

    assert beta[0, 1] > beta[0, 0]
    assert beta[0, 1] > beta[0, 2]
    assert entropy.item() < 1.0
    assert confidence.item() > 0.0


def test_dynamic_bandwidth_beta_narrows_on_transition_target():
    from tiny_vit_sam import SoftSliceCorrespondenceAttention

    torch.manual_seed(11)
    module = SoftSliceCorrespondenceAttention(
        in_channels=4,
        num_heads=2,
        beta_temperature=1.0,
        min_confidence=0.2,
        position_prior_weight=0.0,
        use_dynamic_bandwidth_beta=True,
        dynamic_beta_mode="prior_only",
        use_gt_transition_for_beta=True,
        use_gt_transition_for_neighbor_trust=True,
        sigma_min=0.30,
        sigma_max=1.25,
        use_neighbor_residual_fusion=True,
    )
    module.eval()

    trus_feat = torch.ones(2, 4, 3, 3)
    mri_set = torch.ones(2, 5, 4, 3, 3)
    transition_target = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        _, _, info = module(
            trus_feat,
            mri_set,
            transition_target=transition_target,
        )

    beta = info["beta"]
    assert beta.shape == (2, 5)
    assert torch.allclose(beta.sum(dim=1), torch.ones(2), atol=1e-6)
    assert beta[1, 2] > beta[0, 2]
    assert beta[1, 0] < beta[0, 0]
    assert beta[1, 4] < beta[0, 4]
    assert info["sigma"][1] < info["sigma"][0]
    assert torch.allclose(info["neighbor_trust"], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_dynamic_bandwidth_can_use_gt_transition_for_neighbor_trust_only():
    from tiny_vit_sam import SoftSliceCorrespondenceAttention

    torch.manual_seed(12)
    module = SoftSliceCorrespondenceAttention(
        in_channels=4,
        num_heads=2,
        beta_temperature=1.0,
        min_confidence=0.0,
        position_prior_weight=0.0,
        use_dynamic_bandwidth_beta=True,
        dynamic_beta_mode="prior_only",
        use_gt_transition_for_beta=False,
        use_gt_transition_for_neighbor_trust=True,
        sigma_min=0.30,
        sigma_max=1.25,
        use_neighbor_residual_fusion=False,
    )
    module.eval()

    trus_feat = torch.ones(2, 4, 3, 3)
    mri_set = torch.ones(2, 5, 4, 3, 3)
    transition_target = torch.tensor([0.0, 1.0])

    with torch.no_grad():
        _, _, info = module(
            trus_feat,
            mri_set,
            transition_target=transition_target,
        )

    assert torch.allclose(info["neighbor_trust"], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert info["sigma"][1] > 0.30
