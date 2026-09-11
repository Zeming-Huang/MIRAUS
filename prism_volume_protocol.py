"""Shared full-volume inference and evaluation protocol helpers."""

from __future__ import annotations

import numpy as np


def inference_label_ids(gt_slice, box_mode):
    """Return labels to infer without using GT presence in full-image mode."""
    if box_mode == "full_image":
        return [1]
    return [int(label) for label in np.unique(gt_slice) if int(label) != 0]


def context_window_indices(center, length, radius, stride=1):
    """Build a fixed-size context window with edge replication and validity."""
    if int(length) <= 0:
        raise ValueError("length must be positive")
    stride = max(1, int(stride))
    indices = []
    valid = []
    for offset in range(-int(radius), int(radius) + 1):
        raw = int(center) + offset * stride
        valid.append(0 <= raw < int(length))
        indices.append(min(max(raw, 0), int(length) - 1))
    return indices, valid


def volume_scope_masks(gt_volume):
    """Return slice selectors for strict full-volume and legacy comparison scopes."""
    gt_volume = np.asarray(gt_volume)
    if gt_volume.ndim < 3:
        raise ValueError("gt_volume must have shape (slices, ...)")
    nonempty = np.any(gt_volume > 0, axis=tuple(range(1, gt_volume.ndim)))
    return {
        "all": np.ones(gt_volume.shape[0], dtype=bool),
        "gt_nonempty": nonempty.astype(bool),
    }
