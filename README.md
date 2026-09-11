# MIRAUS

**MRI-privileged learning for single-frame TRUS prostate segmentation**

[Interactive Results](https://zeming-huang.github.io/MIRAUS/) | [μRegPro Dataset](https://doi.org/10.5281/zenodo.8004388) | [MedSAM](https://github.com/bowang-lab/MedSAM)

MIRAUS uses paired MRI as privileged information during training to improve a lightweight single-frame TRUS segmentation model. MRI and the privileged teacher are removed after training; deployment requires only one TRUS frame and supports automatic full-image or box-assisted inference.

## Framework

<p align="center">
  <a href="assets/miraus-framework.png">
    <img src="assets/miraus-framework.png" alt="MIRAUS framework: MRI-privileged teacher pretraining, residual privileged transfer, and MRI-free single-frame TRUS deployment" width="100%">
  </a>
</p>

The privileged teacher uses a local multi-slice MRI neighborhood during training and forms an MRI-informed feature through utility-aware aggregation. A residual adapter transfers the teacher-defined feature increment to the TRUS student. At deployment, the MRI branch and teacher are removed, leaving the same lightweight student for automatic full-image or clinician box-assisted segmentation.

## Publication Implementation

This release contains the complete implementation of the proposed method:

- modality-specific TRUS and MRI encoders;
- local multi-slice MRI context construction;
- TRUS-conditioned cross-attention and utility-supervised soft aggregation;
- frozen-teacher privileged residual transfer;
- the zero-initialized student refinement adapter;
- mixed full-image and coarse-box prompt adaptation;
- GT-free single-frame TRUS inference with an optional clinician box;
- the patient-disjoint five-fold μRegPro split used in the publication.

The publication checkpoints are not included in the repository and will be released separately. Internal ablation launchers and paper-specific sweep manifests are not part of this method-verification release.

## Interactive Results

The [result explorer](https://zeming-huang.github.io/MIRAUS/) contains 15 anonymized μRegPro examples. Each example includes apex, mid-gland, and base frames with the manual reference, a TRUS-only baseline, MIRAUS, and the signed probability difference. The gallery contains favorable, representative, and difficult cases.

## Code Layout

- `train_dual_modal.py`: privileged teacher pretraining and frozen-teacher student distillation.
- `privileged_distillation.py`: residual transfer losses and checkpoint architecture inference.
- `prism_offline_distillation.py`: strict teacher/student stage control.
- `finetune_dual_prompt_student.py`: mixed full-image and coarse-box prompt adaptation.
- `inference_single_frame.py`: GT-free automatic or clinician-box single-frame inference.
- `enhanced_dual_modal.py`: cross-modal aggregation, mask decoder, and residual student adapter.
- `tiny_vit_sam.py`: lightweight image encoder and feature modules.
- `configs/muregpro_5fold_splits.json`: anonymized publication fold assignments.
- `tools/validate_splits.py`: split integrity check.
- `inference_3D.py`: retrospective evaluation of independently processed frame stacks.
- `evaluate_comprehensive.py`: overlap and surface-distance evaluation.
- `utils/paired_dataset.py`: paired MRI/TRUS training data loading.
- `tools/`: evaluation, split validation, and result-export utilities.
- `docs/`: static result explorer published through GitHub Pages.

## Installation

```bash
git clone https://github.com/Zeming-Huang/MIRAUS.git
cd MIRAUS
pip install -r requirements.txt
pip install -e .
```

The implementation was developed with Python 3.10 and PyTorch 2.x. CUDA is recommended for training and inference. Development tools, including pytest, can be installed with `pip install -e .[dev]`.

## Method Training

Teacher pretraining and student distillation use the same entry point. Dataset locations, pretrained modality encoders, optimization settings, and output directories are supplied from the command line.

```bash
# Stage 1: MRI-privileged teacher
python train_dual_modal.py \
  -distillation_stage teacher_pretrain \
  -trus_data_root <TRUS_TRAIN_ROOT> \
  -mri_data_root <MRI_TRAIN_ROOT> \
  -val_trus_data_root <TRUS_VALIDATION_ROOT> \
  -val_mri_data_root <MRI_VALIDATION_ROOT> \
  -trus_pretrained_checkpoint <TRUS_ENCODER_CHECKPOINT> \
  -mri_pretrained_checkpoint <MRI_ENCODER_CHECKPOINT> \
  -mri_window_radius 2 \
  -slice_attention_mode ssca_entropy \
  -ssca_use_box_aware_pooling \
  -slice_utility_loss_weight <UTILITY_WEIGHT> \
  -mmd_loss_weight 0 \
  -freeze_encoders \
  --no_fusion \
  -work_dir <TEACHER_OUTPUT_DIR>

# Stage 2: TRUS-only student with a frozen teacher
python train_dual_modal.py \
  -distillation_stage student_distill \
  -teacher_checkpoint <TEACHER_CHECKPOINT> \
  -trus_data_root <TRUS_TRAIN_ROOT> \
  -mri_data_root <MRI_TRAIN_ROOT> \
  -val_trus_data_root <TRUS_VALIDATION_ROOT> \
  -val_mri_data_root <MRI_VALIDATION_ROOT> \
  -trus_pretrained_checkpoint <TRUS_ENCODER_CHECKPOINT> \
  -mri_pretrained_checkpoint <MRI_ENCODER_CHECKPOINT> \
  -use_student_refinement_adapter \
  -distill_feature_mode roi_residual_cosine \
  -distill_feature_scope box \
  -distill_feat_weight <PRIVILEGED_LOSS_WEIGHT> \
  -distill_cosine_weight 0 \
  -distill_residual_weight 0 \
  -distill_kd_weight 0 \
  -mmd_loss_weight 0 \
  -freeze_encoders \
  --no_fusion \
  -work_dir <STUDENT_OUTPUT_DIR>
```

The residual and full-feature Smooth-L1 forms are algebraically equivalent because the teacher and student share the same TRUS base feature. The publication model uses this single teacher-defined increment rather than output-logit distillation.

## Mixed-Prompt Adaptation

After student distillation, mixed-prompt adaptation freezes the learned image representation and optimizes the prompt encoder and mask decoder with full-image rehearsal and coarse training boxes.

```bash
python finetune_dual_prompt_student.py \
  --source-work-dir <STUDENT_OUTPUT_DIR> \
  --source-checkpoint <STUDENT_CHECKPOINT> \
  --trus-roots <TRUS_DATA_ROOTS> \
  --output-dir <MIXED_PROMPT_OUTPUT_DIR>
```

## Single-Frame Inference

The deployment entry point does not load a reference mask or MRI. Omitting `--box` selects the automatic full-image prompt.

```bash
# Automatic full-image prompt
python inference_single_frame.py \
  --checkpoint <MIRAUS_CHECKPOINT> \
  --input <TRUS_FRAME> \
  --output <PREDICTED_MASK>

# Clinician box in original-image xyxy coordinates
python inference_single_frame.py \
  --checkpoint <MIRAUS_CHECKPOINT> \
  --input <TRUS_FRAME> \
  --box X0 Y0 X1 Y1 \
  --output <PREDICTED_MASK>
```

## Five-Fold Split

The validation and held-out patient assignments are provided in `configs/muregpro_5fold_splits.json`; the training set is the complement of those lists. Each anonymized patient appears in exactly one test fold. Validate the manifest with:

```bash
python tools/validate_splits.py
```

## Deployment Contract

The deployed student accepts a single TRUS frame. MRI, the teacher network, and adjacent TRUS slices are training-time resources only and are not required for inference.

## Data and Checkpoints

Experiments use the public μRegPro MRI--TRUS cohort. Dataset files and model checkpoints are not committed to Git. Checkpoint download instructions will be added when the publication weights are released.

## Reproducing the Result Explorer

The public gallery is static and contains no patient identifiers. Its anonymized assets can be regenerated from local out-of-fold predictions with:

```bash
python tools/build_results_browser_assets.py \
  --source-root /path/to/train/us_images \
  --source-root /path/to/val/us_images
```

## License

Code is distributed under the repository license and remains subject to the licenses of the upstream MedSAM, LiteMedSAM, TinyViT, and MobileSAM components. Web examples are provided under the μRegPro CC BY-NC-SA 4.0 terms.

## Acknowledgements

We thank the organizers and contributors of the [μRegPro challenge](https://doi.org/10.5281/zenodo.8004388) for making the paired prostate MRI--TRUS dataset publicly available. We also thank the [MedSAM team](https://github.com/bowang-lab/MedSAM) for releasing the open-source implementation and model resources that supported this work.

MIRAUS also builds on LiteMedSAM, TinyViT, MobileSAM, and Segment Anything. We thank their authors for releasing the corresponding research code.
