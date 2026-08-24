def select_prism_inference_embedding(use_dual_modal, model, trus_feat):
    if use_dual_modal:
        return trus_feat
    return model.image_encoder


def test_prism_dual_modal_inference_uses_trus_feat_directly():
    trus_feat = object()

    class Dummy:
        image_encoder = object()

    selected = select_prism_inference_embedding(True, Dummy(), trus_feat)
    assert selected is trus_feat


def test_single_modal_path_keeps_image_encoder_behavior_marker():
    trus_feat = object()

    class Dummy:
        image_encoder = object()

    selected = select_prism_inference_embedding(False, Dummy(), trus_feat)
    assert selected is Dummy.image_encoder
