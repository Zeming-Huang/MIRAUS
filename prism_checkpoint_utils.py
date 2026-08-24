def remap_legacy_prism_state_dict_keys(state_dict):
    """
    Normalize older PRISM checkpoint keys to the current inference model names.

    Older checkpoints stored adaptive fusion weights under:
      cross_modal_extractor.adaptive_fusion.*

    Current code expects:
      cross_modal_extractor.fusion.*
    """
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("cross_modal_extractor.adaptive_fusion."):
            new_key = key.replace(
                "cross_modal_extractor.adaptive_fusion.",
                "cross_modal_extractor.fusion.",
                1,
            )
        remapped[new_key] = value
    return remapped


def normalize_prism_checkpoint_state_dict(checkpoint):
    """Extract and normalize a PRISM state dict from common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError("PRISM checkpoint must be a state dict or checkpoint dictionary")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("PRISM checkpoint 'model' entry must be a state dict")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    return remap_legacy_prism_state_dict_keys(state_dict)
