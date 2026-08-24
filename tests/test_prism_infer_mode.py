def choose_prism_embedding(mode, trus_feat, cross_modal_extractor=None):
    if mode == "train_consistent":
        return trus_feat
    if mode == "self_attn":
        if cross_modal_extractor is None:
            raise ValueError("cross_modal_extractor is required for self_attn mode")
        enhanced, _ = cross_modal_extractor(trus_feat, trus_feat)
        return enhanced
    if mode == "zero_teacher_attn":
        raise ValueError(
            "zero_teacher_attn requires explicit teacher feature generation in inference code"
        )
    raise ValueError(f"Unknown prism inference mode: {mode}")


def test_choose_prism_embedding_train_consistent_returns_trus_feat():
    trus_feat = object()
    assert choose_prism_embedding("train_consistent", trus_feat) is trus_feat


def test_choose_prism_embedding_self_attn_uses_extractor():
    trus_feat = object()

    class DummyExtractor:
        def __call__(self, q, kv):
            assert q is trus_feat
            assert kv is trus_feat
            return "enhanced_feat", None

    assert (
        choose_prism_embedding("self_attn", trus_feat, DummyExtractor())
        == "enhanced_feat"
    )


def test_choose_prism_embedding_zero_teacher_attn_requires_special_handling():
    trus_feat = object()
    try:
        choose_prism_embedding("zero_teacher_attn", trus_feat)
    except ValueError as e:
        assert "zero_teacher_attn" in str(e)
        return
    raise AssertionError("Expected zero_teacher_attn to require special handling")
