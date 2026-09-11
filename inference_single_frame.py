"""Run MIRAUS on one TRUS frame with a full-image or clinician box prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from prism_checkpoint_utils import normalize_prism_checkpoint_state_dict
from prism_cv_npy_eval import build_model_from_checkpoint
from privileged_distillation import (
    infer_student_architecture_from_state_dict,
    primary_logits_from_model_output,
)


def resolve_prompt_box(box, image_width: int, image_height: int) -> tuple[float, ...]:
    """Return a validated xyxy prompt; None selects the complete image."""
    if image_width < 2 or image_height < 2:
        raise ValueError("The input image must be at least 2 x 2 pixels")
    if box is None:
        return (0.0, 0.0, float(image_width - 1), float(image_height - 1))
    if len(box) != 4:
        raise ValueError("A box must contain four values: x0 y0 x1 y1")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= image_width - 1):
        raise ValueError("Box coordinates must satisfy 0 <= x0 < x1 <= image_width - 1")
    if not (0.0 <= y0 < y1 <= image_height - 1):
        raise ValueError("Box coordinates must satisfy 0 <= y0 < y1 <= image_height - 1")
    return (x0, y0, x1, y1)


def load_trus_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read TRUS image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a grayscale or three-channel image, got {image.shape}")
    return image


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    if resized.max() > 1.0:
        scale = 255.0 if resized.max() <= 255.0 else float(resized.max())
        resized = resized / max(scale, 1.0)
    return (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )


def scale_box_to_model(box, image_width: int, image_height: int, device) -> torch.Tensor:
    x0, y0, x1, y1 = box
    scaled = [
        x0 * 255.0 / float(image_width - 1),
        y0 * 255.0 / float(image_height - 1),
        x1 * 255.0 / float(image_width - 1),
        y1 * 255.0 / float(image_height - 1),
    ]
    return torch.tensor([[scaled]], dtype=torch.float32, device=device)


def load_deployment_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalize_prism_checkpoint_state_dict(checkpoint)
    architecture = infer_student_architecture_from_state_dict(state_dict)
    no_fusion = not any(
        key.startswith("cross_modal_extractor.fusion.") for key in state_dict
    )
    return build_model_from_checkpoint(
        str(checkpoint_path),
        str(device),
        no_fusion_ablation=no_fusion,
        model_kwargs=architecture,
    )


@torch.no_grad()
def predict(model, image: np.ndarray, box, device: torch.device, threshold: float):
    height, width = image.shape[:2]
    image_tensor = preprocess_image(image, device)
    box_tensor = scale_box_to_model(box, width, height, device)
    output = model(image_tensor, None, box_tensor, training=False)
    logits = primary_logits_from_model_output(output)
    probability = torch.sigmoid(
        model.postprocess_masks(logits, (256, 256), (height, width))
    )[0, 0]
    probability_np = probability.cpu().numpy().astype(np.float32)
    return (probability_np >= float(threshold)).astype(np.uint8), probability_np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Single TRUS image")
    parser.add_argument("--output", type=Path, required=True, help="Output binary PNG")
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional clinician box in original-image xyxy coordinates",
    )
    parser.add_argument("--probability-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    device = torch.device(args.device)
    image = load_trus_image(args.input)
    box = resolve_prompt_box(args.box, image.shape[1], image.shape[0])
    model = load_deployment_model(args.checkpoint, device)
    mask, probability = predict(model, image, box, device, args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask * 255):
        raise RuntimeError(f"Unable to write mask: {args.output}")
    if args.probability_output is not None:
        args.probability_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.probability_output, probability)


if __name__ == "__main__":
    main()
