# MIRAUS

**MRI-privileged learning for single-frame TRUS prostate segmentation**

[Interactive Results](https://anonymous.4open.science/w/MIRAUS/) | [μRegPro Dataset](https://doi.org/10.5281/zenodo.8004388) | [MedSAM](https://github.com/bowang-lab/MedSAM)

MIRAUS uses paired MRI as privileged information during training to improve a lightweight single-frame TRUS segmentation model. MRI and the privileged teacher are removed after training; deployment requires only one TRUS frame and supports automatic full-image or box-assisted inference.

## Framework

<p align="center">
  <a href="https://anonymous.4open.science/api/repo/MIRAUS/file/assets/miraus-framework.png">
    <img src="https://anonymous.4open.science/api/repo/MIRAUS/file/assets/miraus-framework.png" alt="MIRAUS framework: MRI-privileged teacher pretraining, residual privileged transfer, and MRI-free single-frame TRUS deployment" width="100%">
  </a>
</p>

The privileged teacher uses a local multi-slice MRI neighborhood during training and forms an MRI-informed feature through utility-aware aggregation. A residual adapter transfers the teacher-defined feature increment to the TRUS student. At deployment, the MRI branch and teacher are removed, leaving the same lightweight student for automatic full-image or clinician box-assisted segmentation.

## Publication Implementation

This release contains the complete implementation of the proposed method:

- modality-specific TRUS and MRI encoders;
- local multi-slice MRI context construction;
- TRUS-conditioned cross-attention and utility-supervised soft aggregation;
- direct candidate-head segmentation supervision with a detached utility target;
- frozen-teacher privileged residual transfer;
- the zero-initialized student refinement adapter;
- mixed full-image and coarse-box prompt adaptation;
- GT-free single-frame TRUS inference with an optional clinician box;
- the patient-disjoint five-fold μRegPro split used in the publication.

The publication checkpoints are not included in the repository and will be released separately. Internal ablation launchers and paper-specific sweep manifests are not part of this method-verification release.

## Interactive Results

The [result explorer](https://anonymous.4open.science/w/MIRAUS/) contains 15 anonymized μRegPro examples. Each example includes apex, mid-gland, and base frames with the manual reference, a TRUS-only baseline, MIRAUS, and the signed probability difference. The gallery contains favorable, representative, and difficult cases.

### View the results locally

If the hosted anonymous viewer cannot load because of browser sandbox restrictions, the same examples can be viewed locally after downloading the repository:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/` in a browser. This viewer uses the included results and does not require model checkpoints or a GPU.

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

Download and extract the repository ZIP, then open a terminal in the extracted repository directory. Use the **Download** button on the anonymous repository page or **Code > Download ZIP** on GitHub.

```bash
pip install -r requirements.txt
pip install -e .
```

The implementation was developed with Python 3.10 and PyTorch 2.x. CUDA is recommended for training and inference. Development tools, including pytest, can be installed with `pip install -e .[dev]`.

## License

Code is distributed under the repository license and remains subject to the licenses of the upstream MedSAM, LiteMedSAM, TinyViT, and MobileSAM components. Web examples are provided under the μRegPro CC BY-NC-SA 4.0 terms.

## Acknowledgements

We thank the organizers and contributors of the [μRegPro challenge](https://doi.org/10.5281/zenodo.8004388) for making the paired prostate MRI--TRUS dataset publicly available. We also thank the [MedSAM team](https://github.com/bowang-lab/MedSAM) for releasing the open-source implementation and model resources that supported this work.

MIRAUS also builds on LiteMedSAM, TinyViT, MobileSAM, and Segment Anything. We thank their authors for releasing the corresponding research code.
