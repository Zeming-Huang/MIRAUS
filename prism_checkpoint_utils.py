from privileged_distillation import extract_model_state_dict


def remap_legacy_prism_state_dict_keys(state_dict):
    """
    Normalize older PRISM/MIRAUS checkpoint keys to current model names.

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
    """Extract and normalize a MIRAUS state dict from common checkpoint layouts."""
    state_dict = extract_model_state_dict(checkpoint)
    return remap_legacy_prism_state_dict_keys(state_dict)


def normalize_prism_training_checkpoint(checkpoint_or_state_dict):
    """Normalize model weights while preserving training checkpoint metadata."""
    model_state = normalize_prism_checkpoint_state_dict(checkpoint_or_state_dict)
    if isinstance(checkpoint_or_state_dict, dict) and any(
        isinstance(checkpoint_or_state_dict.get(key), dict)
        for key in ("model", "state_dict", "model_state_dict")
    ):
        normalized = dict(checkpoint_or_state_dict)
    else:
        normalized = {}
    normalized["model"] = model_state
    return normalized


def get_resume_loss_from_checkpoint(checkpoint, default=1e10):
    """Return a compatible resume loss from legacy or current checkpoint fields."""
    for key in ("loss", "train_loss", "val_loss"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return default


def filter_checkpoint_state_dict_by_prefix(
    checkpoint_or_state_dict,
    source_prefixes,
    target_prefix=None,
):
    """Select normalized checkpoint keys by prefix, optionally rewriting one prefix."""
    state_dict = extract_model_state_dict(checkpoint_or_state_dict)
    if isinstance(source_prefixes, str):
        source_prefixes = (source_prefixes,)
    filtered = {}
    for key, value in state_dict.items():
        for source_prefix in source_prefixes:
            if not key.startswith(source_prefix):
                continue
            new_key = key
            if target_prefix is not None:
                new_key = target_prefix + key[len(source_prefix):]
            filtered[new_key] = value
            break
    return filtered
