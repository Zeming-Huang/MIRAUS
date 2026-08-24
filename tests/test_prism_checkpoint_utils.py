from prism_checkpoint_utils import (
    normalize_prism_checkpoint_state_dict,
    remap_legacy_prism_state_dict_keys,
)


def test_remap_legacy_adaptive_fusion_keys():
    state = {
        "cross_modal_extractor.adaptive_fusion.attention.0.weight": 1,
        "cross_modal_extractor.adaptive_fusion.fusion_conv.0.weight": 2,
        "cross_modal_extractor.cross_modal_attn.q_linear.weight": 3,
        "mask_decoder.some_key": 4,
    }

    remapped = remap_legacy_prism_state_dict_keys(state)

    assert (
        "cross_modal_extractor.fusion.attention.0.weight" in remapped
    )
    assert (
        "cross_modal_extractor.fusion.fusion_conv.0.weight" in remapped
    )
    assert (
        "cross_modal_extractor.adaptive_fusion.attention.0.weight"
        not in remapped
    )
    assert remapped["cross_modal_extractor.cross_modal_attn.q_linear.weight"] == 3
    assert remapped["mask_decoder.some_key"] == 4


def test_normalize_checkpoint_extracts_model_and_removes_data_parallel_prefix():
    checkpoint = {
        "model": {
            "module.cross_modal_extractor.adaptive_fusion.weight": 1,
            "module.mask_decoder.weight": 2,
        },
        "epoch": 3,
    }

    normalized = normalize_prism_checkpoint_state_dict(checkpoint)

    assert normalized == {
        "cross_modal_extractor.fusion.weight": 1,
        "mask_decoder.weight": 2,
    }
