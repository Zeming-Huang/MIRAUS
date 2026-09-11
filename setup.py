# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import find_packages, setup

setup(
    name="miraus",
    version="0.1.0",
    description="MRI-privileged learning for single-frame TRUS prostate segmentation",
    python_requires=">=3.9",
    install_requires=[
        "matplotlib",
        "monai",
        "nibabel",
        "numpy",
        "opencv-python",
        "pandas",
        "scikit-image",
        "scipy",
        "SimpleITK>=2.2.1",
        "timm",
        "torch",
        "tqdm",
    ],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["pytest", "flake8", "isort", "black", "mypy"],
    },
)
