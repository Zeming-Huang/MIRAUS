# %%
import os
import ast
import csv
import json
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT, CrossModalFeatureExtractor
from enhanced_dual_modal import EnhancedMaskDecoder, EnhancedDualModalMedSAM_Lite
from utils.paired_dataset import PairedNpyDataset
from utils.efficient_paired_dataset import EfficientPairedNpyDataset
import cv2
import torch.nn.functional as F

from matplotlib import pyplot as plt
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-trus_data_root", type=str, default="./data/npy/TRUS_MRI_paired_train_lesion/trus",
    help="Path to the TRUS data root."
)
parser.add_argument(
    "-mri_data_root", type=str, default="./data/npy/TRUS_MRI_paired_train_lesion/mri",
    help="Path to the MRI data root."
)
parser.add_argument(
    "-val_trus_data_root", type=str, default="./data/npy/TRUS_MRI_paired_val_lesion/trus",
    help="Path to the validation TRUS data root."
)
parser.add_argument(
    "-val_mri_data_root", type=str, default="./data/npy/TRUS_MRI_paired_val_lesion/mri",
    help="Path to the validation MRI data root."
)
parser.add_argument(
    "-val_interval", type=int, default=1,
    help="Evaluate on validation set every N epochs."
)
parser.add_argument(
    "-pretrained_checkpoint", type=str, default="lite_medsam.pth",
    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
    "-trus_pretrained_checkpoint", type=str, 
    default="./checkpoints/trus_encoder.pth",
    help="Path to the TRUS pretrained checkpoint."
)
parser.add_argument(
    "-mri_pretrained_checkpoint", type=str,
    default="./checkpoints/mri_encoder.pth",
    help="Path to the MRI pretrained checkpoint."
)
parser.add_argument(
    "-resume", type=str, default='workdir/dual_modal_latest.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir/dual_modal_lesion",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=20,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-seed", type=int, default=2026,
    help="Random seed used by Python, NumPy, and PyTorch."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-training_box_mode", choices=["gt", "full_image"], default="gt",
    help="Prompt protocol used for both training and validation."
)
parser.add_argument(
    "-lr", type=float, default=0.0001,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "-mmd_loss_weight", type=float, default=0.1,
    help="Weight of MMD loss for cross-modal alignment."
)
parser.add_argument(
    "--use_adaptive_fusion", action="store_true", default=True,
    help="Use adaptive fusion (default). Set to False to use linear fusion for ablation."
)
parser.add_argument(
    "--no_adaptive_fusion", action="store_false", dest="use_adaptive_fusion",
    help="Use linear fusion instead of adaptive fusion (for ablation study)."
)
parser.add_argument(
    "--no_fusion", action="store_true", default=False,
    help="Ablation: disable fusion, directly use enhanced features (for ablation study)."
)
parser.add_argument(
    "--ablation_no_mmd", action="store_true", default=False,
    help="Ablation: disable MMD loss (set weight to 0)."
)
parser.add_argument(
    "-slice_attention_mode", type=str, default="index_pairing",
    choices=[
        "index_pairing",
        "random_neighbor",
        "average_neighbor",
        "original_index",
        "random_neighbor_sampling",
        "average_neighbor_fusion",
        "ssca_no_entropy",
        "ssca_entropy",
        "center_only_with_same_params",
    ],
    help="MRI-TRUS slice correspondence mode for training."
)
parser.add_argument(
    "-ablation_mode", type=str, default=None,
    choices=[
        "original_index",
        "random_neighbor_sampling",
        "average_neighbor_fusion",
        "ssca_no_entropy",
        "ssca_entropy",
        "center_only_with_same_params",
    ],
    help="Ablation mode alias for slice attention mode. Overrides -slice_attention_mode when set."
)
parser.add_argument(
    "-pairing_mode", type=str, default="normal",
    choices=[
        "normal",
        "shift_plus_1",
        "shift_minus_1",
        "shift_plus_3",
        "shift_minus_3",
        "same_patient_random",
        "different_patient_random",
    ],
    help="Dataset-level MRI-TRUS pairing robustness mode."
)
parser.add_argument(
    "-mri_window_radius", type=int, default=0,
    help="MRI local window radius. S = 2 * radius + 1."
)
parser.add_argument(
    "-ssca_descriptor_dim", type=int, default=128,
    help="Descriptor projection dimension for SSCA beta scoring."
)
parser.add_argument(
    "-ssca_beta_temperature", type=float, default=0.1,
    help="Softmax temperature for SSCA beta scoring."
)
parser.add_argument(
    "-ssca_min_confidence", type=float, default=0.2,
    help="Minimum correspondence confidence used by the SSCA entropy gate."
)
parser.add_argument(
    "-ssca_position_prior_weight", type=float, default=0.0,
    help="Weight of the soft relative-position prior added to SSCA beta logits."
)
parser.add_argument(
    "-ssca_position_prior_sigma", type=float, default=0.75,
    help="Gaussian sigma, in slice offsets, for the SSCA relative-position prior."
)
parser.add_argument(
    "-ssca_max_window_size", type=int, default=7,
    help="Maximum MRI window size supported by the learnable SSCA relative-position bias."
)
parser.add_argument(
    "-slice_corr_loss_weight", type=float, default=0.0,
    help="Weight of the soft center-prior correspondence loss inside the cross-modal auxiliary loss."
)
parser.add_argument(
    "-slice_corr_prior_sigma", type=float, default=0.75,
    help="Gaussian sigma, in slice offsets, for the soft center-prior correspondence target."
)
parser.add_argument(
    "-ssca_use_box_aware_pooling", action="store_true",
    help="Enable SSCA-v3 box-aware descriptor pooling."
)
parser.add_argument(
    "-ssca_boundary_ring_width", type=int, default=3,
    help="Boundary ring width, in feature pixels, for box-aware descriptor pooling."
)
parser.add_argument(
    "-slice_utility_loss_weight", type=float, default=0.0,
    help="Weight of the SSCA-v3 candidate-utility supervision loss."
)
parser.add_argument(
    "-slice_utility_temperature", type=float, default=0.5,
    help="Softmax temperature used to convert candidate losses into a utility target."
)
parser.add_argument(
    "-use_transition_aware_beta", action="store_true",
    help="Enable transition-aware neighbor suppression inside SSCA-v3 beta logits."
)
parser.add_argument(
    "-transition_loss_weight", type=float, default=0.0,
    help="Weight of the SSCA-v3 transition prediction loss."
)
parser.add_argument(
    "-transition_cls_loss_weight", type=float, default=0.05,
    help="Internal BCE classification weight for SSCA-v4 transition loss."
)
parser.add_argument(
    "-transition_reg_loss_weight", type=float, default=0.01,
    help="Internal SmoothL1 regression weight for SSCA-v4 transition loss."
)
parser.add_argument(
    "-jump_logit_penalty", type=float, default=1.0,
    help="Penalty multiplier applied to far MRI offsets when transition score is high."
)
parser.add_argument(
    "-use_reliability_gate", action="store_true",
    help="Enable SSCA-v3 reliability gate instead of entropy-only gating."
)
parser.add_argument(
    "-reliability_use_candidate_agreement", action="store_true",
    help="Use candidate agreement in the SSCA-v3 reliability score."
)
parser.add_argument(
    "-use_dynamic_bandwidth_beta", action="store_true",
    help="Enable SSCA-v4 transition-aware dynamic bandwidth beta."
)
parser.add_argument(
    "-sigma_min", type=float, default=0.30,
    help="Minimum Gaussian bandwidth for transition slices in SSCA-v4."
)
parser.add_argument(
    "-sigma_max", type=float, default=1.25,
    help="Maximum Gaussian bandwidth for stable slices in SSCA-v4."
)
parser.add_argument(
    "-dynamic_bandwidth_use_gt_transition_prob", type=float, default=0.0,
    help="Probability of using GT transition target for dynamic bandwidth during training."
)
parser.add_argument(
    "-dynamic_bandwidth_warmup_epochs", type=int, default=0,
    help="Epoch warmup before probabilistic GT transition usage is allowed."
)
parser.add_argument(
    "-dynamic_beta_mode", type=str, default="content_plus_prior",
    choices=["prior_only", "content_plus_prior"],
    help="SSCA-v4 beta mode."
)
parser.add_argument(
    "-dynamic_prior_weight", type=float, default=1.0,
    help="Weight of dynamic bandwidth prior in content_plus_prior mode."
)
parser.add_argument(
    "-use_neighbor_residual_fusion", action="store_true",
    help="Use center-anchored neighbor residual fusion for SSCA-v4."
)
parser.add_argument(
    "-use_gt_transition_for_beta", action="store_true",
    help="Diagnostic oracle mode: use GT transition target for dynamic beta bandwidth."
)
parser.add_argument(
    "-use_gt_transition_for_neighbor_trust", action="store_true",
    help="Diagnostic oracle mode: use GT transition target for neighbor trust and confidence gating."
)
parser.add_argument(
    "-depth_prior_enabled", action="store_true",
    help="Enable relative-depth prior embedding and rho regularization."
)
parser.add_argument(
    "-depth_embed_dim", type=int, default=32,
    help="Embedding dimension for the relative-depth prior."
)
parser.add_argument(
    "-depth_prior_hidden_dim", type=int, default=64,
    help="Hidden dimension for the relative-depth prior MLP."
)
parser.add_argument(
    "-lambda_depth_prior", type=float, default=0.01,
    help="Weight of the weak U-shaped rho depth-prior regularization inside auxiliary loss."
)
parser.add_argument(
    "-depth_prior_no_u_shape_regularization", action="store_true",
    help="Disable weak U-shaped rho regularization while keeping depth embedding available."
)
parser.add_argument(
    "-depth_gate_enabled", action="store_true",
    help="Enable relative-depth-conditioned fusion gate."
)
parser.add_argument(
    "-depth_gate_alpha_min", type=float, default=0.05,
    help="Minimum alpha for relative-depth-conditioned fusion gate."
)
parser.add_argument(
    "-depth_gate_alpha_max", type=float, default=0.60,
    help="Maximum alpha for relative-depth-conditioned fusion gate."
)
parser.add_argument(
    "-use_depth_gated_mmd", action="store_true",
    help="Use depth-gated anatomical descriptor MMD instead of the plain feature MMD."
)
parser.add_argument(
    "-dg_mmd_project_dim", type=int, default=128,
    help="Projection dimension for depth-gated anatomical MMD descriptors."
)
parser.add_argument(
    "-dg_mmd_lambda_center", type=float, default=0.005,
    help="Center MRI descriptor MMD coefficient."
)
parser.add_argument(
    "-dg_mmd_lambda_priv", type=float, default=0.005,
    help="Privileged fused MRI descriptor MMD coefficient."
)
parser.add_argument(
    "-dg_mmd_min_priv_weight", type=float, default=0.2,
    help="Minimum privileged MMD weight at unreliable apex/base depths."
)
parser.add_argument(
    "-beta_modulation_enabled", action="store_true",
    help="Enable small rho-conditioned beta modulation toward the center-biased template."
)
parser.add_argument(
    "-beta_modulation_mix_max", type=float, default=0.0,
    help="Maximum rho-conditioned beta modulation mix."
)
# ===== 特权蒸馏 (LUPI) 相关参数 =====
parser.add_argument(
    "-use_privileged_distillation", action="store_true", default=True,
    help="启用特权蒸馏:Teacher(TRUS+MRI/SSCA)指导Student(纯TRUS/adapter),推理只跑Student。"
)
parser.add_argument(
    "-no_privileged_distillation", dest="use_privileged_distillation", action="store_false",
    help="关闭特权蒸馏(回退到旧的单路径行为,不推荐)。"
)
parser.add_argument(
    "-student_adapter_hidden", type=int, default=256,
    help="Student adapter 的隐藏通道数。"
)
parser.add_argument(
    "-distill_feat_weight", type=float, default=1.0,
    help="特征级蒸馏损失权重(Student decoder输入特征逼近Teacher,目标detach)。"
)
parser.add_argument(
    "-distill_kd_weight", type=float, default=1.0,
    help="logits级KD蒸馏损失权重(KL(Student||Teacher),目标detach)。"
)
parser.add_argument(
    "-distill_kd_temp", type=float, default=2.0,
    help="logits KD蒸馏温度T。"
)
parser.add_argument(
    "-teacher_loss_weight", type=float, default=1.0,
    help="Teacher路径分割损失权重gamma,保证蒸馏目标质量。"
)
parser.add_argument(
    "-slice_attention_log_interval", type=int, default=50,
    help="Log SSCA diagnostics every N training steps. Set <=0 to disable."
)
parser.add_argument(
    "--cache_encoder_features", action="store_true",
    help="Precompute frozen TRUS/MRI encoder features and train from cached features."
)
parser.add_argument(
    "-feature_cache_dir", type=str, default=None,
    help="Directory for cached encoder features. Defaults to <work_dir>/feature_cache."
)
parser.add_argument(
    "--regenerate_feature_cache", action="store_true",
    help="Regenerate cached encoder features even if cache files already exist."
)
parser.add_argument(
    "--no_best_result_generation", action="store_true",
    help="Do not generate full 3D validation outputs whenever a new best Dice is found."
)
parser.add_argument(
    "--generate_final_results", action="store_true",
    help="Generate full 3D validation outputs once after training finishes."
)
parser.add_argument(
    "-final_results_checkpoint", type=str, default="best_2d",
    choices=["best_2d", "best_3d", "latest"],
    help="Checkpoint used for final full-volume export."
)
parser.add_argument(
    "-dropout", type=float, default=0.0,
    help="Dropout rate for regularization."
)
parser.add_argument(
    "-early_stopping_patience", type=int, default=20,
    help="Early stopping patience."
)
parser.add_argument(
    "-min_delta", type=float, default=0.0001,
    help="Minimum delta for early stopping."
)
parser.add_argument(
    "-freeze_encoders", action="store_true",
    help="Freeze encoder parameters during training."
)
parser.add_argument(
    "-lr_scheduler", type=str, default="plateau", 
    choices=["plateau", "cosine", "step"],
    help="Learning rate scheduler type."
)
parser.add_argument(
    "-gradient_checkpointing", action="store_true",
    help="Use gradient checkpointing to save memory."
)
parser.add_argument(
    "-mixed_precision", action="store_true",
    help="Use mixed precision training to save memory."
)
parser.add_argument(
    "-samples_per_epoch", type=int, default=64,
    help="Number of samples per epoch (for efficient training)."
)
parser.add_argument(
    "-diagnostic_checkpoint_epochs", type=str, default="",
    help="Comma-separated epoch numbers to save extra diagnostic checkpoints, e.g. 1,3,6,8,12."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)

args = parser.parse_args()
if args.ablation_mode is not None:
    args.slice_attention_mode = args.ablation_mode

# %%
work_dir = args.work_dir
trus_data_root = args.trus_data_root
mri_data_root = args.mri_data_root
val_trus_data_root = args.val_trus_data_root
val_mri_data_root = args.val_mri_data_root
medsam_lite_checkpoint = args.pretrained_checkpoint
trus_pretrained_checkpoint = args.trus_pretrained_checkpoint
mri_pretrained_checkpoint = args.mri_pretrained_checkpoint
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
mmd_loss_weight = args.mmd_loss_weight
do_sancheck = args.sanity_check
checkpoint = args.resume
val_interval = args.val_interval
diagnostic_checkpoint_epochs = {
    int(x.strip()) for x in str(args.diagnostic_checkpoint_epochs).split(",")
    if x.strip()
}

makedirs(work_dir, exist_ok=True)
with open(join(work_dir, "training_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(
        {"created_at": datetime.now().isoformat(), "args": vars(args)},
        config_file,
        indent=2,
        sort_keys=True,
    )

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def cal_iou(result, reference):
    reduce_dims = tuple(range(1, result.ndim))
    intersection = torch.count_nonzero(
        torch.logical_and(result, reference), dim=reduce_dims
    ).float()
    union = torch.count_nonzero(
        torch.logical_or(result, reference), dim=reduce_dims
    ).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0), 1.0)
    return iou.unsqueeze(1)

@torch.no_grad()
def compute_dice_score(pred_logits: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute mean Dice score for a batch."""
    pred = torch.sigmoid(pred_logits)
    pred = (pred > 0.5).float()
    gt = gt.float()
    intersection = torch.sum(pred * gt, dim=[1, 2, 3])
    union = torch.sum(pred, dim=[1, 2, 3]) + torch.sum(gt, dim=[1, 2, 3])
    # 处理union为0的情况
    dice = torch.where(union == 0, 
                      torch.where(intersection == 0, torch.ones_like(intersection), torch.zeros_like(intersection)),
                      2.0 * intersection / union)
    return dice.mean().item()


def _feature_cache_path(cache_dir: str, image_path: str) -> str:
    stem = os.path.splitext(basename(image_path))[0]
    return join(cache_dir, stem + ".pt")


class NpyImageFeatureDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.load(image_path, 'r', allow_pickle=True)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        return image_tensor, image_path


@torch.no_grad()
def cache_encoder_features(encoder, image_paths, cache_dir, device, batch_size=16, regenerate=False, mixed_precision=False):
    makedirs(cache_dir, exist_ok=True)
    pending_paths = [
        path for path in image_paths
        if regenerate or not isfile(_feature_cache_path(cache_dir, path))
    ]
    print(f"[CACHE] {cache_dir}: {len(image_paths)} total, {len(pending_paths)} pending")
    if len(pending_paths) == 0:
        return

    was_training = encoder.training
    encoder.eval()
    loader = DataLoader(
        NpyImageFeatureDataset(pending_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    for images, paths in tqdm(loader, desc=f"Caching features -> {basename(cache_dir)}"):
        images = images.to(device)
        if mixed_precision:
            with torch.amp.autocast('cuda'):
                features = encoder(images)
        else:
            features = encoder(images)
        features = features.detach().cpu().half()
        for feature, image_path in zip(features, paths):
            torch.save(feature.clone(), _feature_cache_path(cache_dir, image_path))
    encoder.train(was_training)

@torch.no_grad()
def evaluate_on_val(model: nn.Module, trus_data_root: str, mri_data_root: str, batch_size_eval: int, num_workers_eval: int, device_eval: str) -> float:
    """Run validation on paired dataset and return mean Dice."""
    if trus_data_root is None or mri_data_root is None or not isdir(trus_data_root) or not isdir(mri_data_root):
        return float("nan")

    model.eval()
    val_dataset = PairedNpyDataset(trus_data_root, mri_data_root, data_aug=False)
    if len(val_dataset) == 0:
        return float("nan")
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=num_workers_eval, pin_memory=True)

    dices: list[float] = []
    for batch in val_loader:
        trus_image = batch["trus_image"].to(device_eval)
        mri_image = batch["mri_image"].to(device_eval)
        gt2D = batch["gt2D"].to(device_eval)
        boxes = batch["bboxes"].to(device_eval)
        
        # 推理时只使用TRUS
        logits_pred, _ = model(trus_image, mri_image, boxes, training=False)
        dice = compute_dice_score(logits_pred, gt2D)
        dices.append(dice)

    model.train()
    if len(dices) == 0:
        return float("nan")
    return float(np.mean(dices))

# %%
class DualModalMedSAM_Lite(nn.Module):
    """
    双模态MedSAM模型
    训练时使用MRI和TRUS双模态,推理时仅需TRUS
    """
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                use_cross_modal=True,
                use_adaptive_fusion=True,
                use_fusion=True,
                slice_attention_mode="index_pairing"
                ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_fusion = use_fusion
        self.slice_attention_mode = slice_attention_mode
        
        # 双编码器
        self.trus_encoder = image_encoder
        self.mri_encoder = deepcopy(image_encoder) if use_cross_modal else None
        
        # 跨模态特征提取器
        if use_cross_modal:
            self.cross_modal_extractor = CrossModalFeatureExtractor(
                in_channels=256, 
                num_heads=8, 
                mmd_weight=mmd_loss_weight,
                use_adaptive_fusion=use_adaptive_fusion,
                use_fusion=getattr(self, 'use_fusion', True)
            )
        
        # SAM组件
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, trus_image, mri_image, boxes, training=True):
        """
        Args:
            trus_image: TRUS图像 (B, 3, H, W)
            mri_image: MRI图像 (B, 3, H, W) - 训练时使用,推理时可为None
            boxes: 边界框 (B, 1, 4)
            training: 是否为训练模式
        """
        # TRUS特征编码
        trus_feat = self.trus_encoder(trus_image)  # (B, 256, 64, 64)
        
        if training and self.use_cross_modal and mri_image is not None:
            # 训练时: MRI引导TRUS特征学习
            if mri_image.dim() == 5:
                B, S, C, H, W = mri_image.shape
                mri_feat = self.mri_encoder(mri_image.reshape(B * S, C, H, W))
                _, feat_C, feat_H, feat_W = mri_feat.shape
                mri_feat = mri_feat.reshape(B, S, feat_C, feat_H, feat_W)
            else:
                mri_feat = self.mri_encoder(mri_image)  # (B, 256, 64, 64)
            enhanced_trus_feat, mmd_loss = self.cross_modal_extractor(
                trus_feat,
                mri_feat,
                slice_attention_mode=self.slice_attention_mode,
            )
            image_embedding = enhanced_trus_feat
        else:
            # 推理时: 仅使用TRUS特征
            image_embedding = trus_feat
            mmd_loss = None
        
        # SAM解码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        if training and mmd_loss is not None:
            return low_res_masks, iou_predictions, mmd_loss
        else:
            return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """Do cropping and resizing"""
        masks = masks[:, :, :new_size[0], :new_size[1]]
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks

# %%
# 构建模型组件
medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[64, 128, 160, 320],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = EnhancedMaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    ),
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    use_src_enhancement=True,
)

# 消融实验配置
use_adaptive_fusion = args.use_adaptive_fusion
use_fusion = not args.no_fusion  # 如果--no_fusion，则use_fusion=False
if args.ablation_no_mmd:
    mmd_loss_weight = 0.0
    print(f"[ABLATION] MMD损失已禁用 (权重设为0)")
if args.no_fusion:
    print(f"[ABLATION] 融合已禁用，直接使用增强后的特征")

# 创建双模态模型
medsam_lite_model = EnhancedDualModalMedSAM_Lite(
    image_encoder=medsam_lite_image_encoder,
    mask_decoder=medsam_lite_mask_decoder,
    prompt_encoder=medsam_lite_prompt_encoder,
    use_cross_modal=True,
    use_src_enhancement=True,
    use_adaptive_fusion=use_adaptive_fusion,
    use_fusion=use_fusion,
    slice_attention_mode=args.slice_attention_mode,
    ssca_descriptor_dim=args.ssca_descriptor_dim,
    ssca_beta_temperature=args.ssca_beta_temperature,
    ssca_min_confidence=args.ssca_min_confidence,
    ssca_position_prior_weight=args.ssca_position_prior_weight,
    ssca_position_prior_sigma=args.ssca_position_prior_sigma,
    ssca_max_window_size=args.ssca_max_window_size,
    slice_corr_loss_weight=args.slice_corr_loss_weight,
    slice_corr_prior_sigma=args.slice_corr_prior_sigma,
    ssca_use_box_aware_pooling=args.ssca_use_box_aware_pooling,
    ssca_boundary_ring_width=args.ssca_boundary_ring_width,
    slice_utility_loss_weight=args.slice_utility_loss_weight,
    slice_utility_temperature=args.slice_utility_temperature,
    use_transition_aware_beta=args.use_transition_aware_beta,
    transition_loss_weight=args.transition_loss_weight,
    jump_logit_penalty=args.jump_logit_penalty,
    use_reliability_gate=args.use_reliability_gate,
    reliability_use_candidate_agreement=args.reliability_use_candidate_agreement,
    transition_cls_loss_weight=args.transition_cls_loss_weight,
    transition_reg_loss_weight=args.transition_reg_loss_weight,
    use_dynamic_bandwidth_beta=args.use_dynamic_bandwidth_beta,
    sigma_min=args.sigma_min,
    sigma_max=args.sigma_max,
    dynamic_bandwidth_use_gt_transition_prob=args.dynamic_bandwidth_use_gt_transition_prob,
    dynamic_bandwidth_warmup_epochs=args.dynamic_bandwidth_warmup_epochs,
    dynamic_beta_mode=args.dynamic_beta_mode,
    dynamic_prior_weight=args.dynamic_prior_weight,
    use_neighbor_residual_fusion=args.use_neighbor_residual_fusion,
    use_gt_transition_for_beta=args.use_gt_transition_for_beta,
    use_gt_transition_for_neighbor_trust=args.use_gt_transition_for_neighbor_trust,
    depth_prior_enabled=args.depth_prior_enabled,
    depth_embed_dim=args.depth_embed_dim,
    depth_prior_hidden_dim=args.depth_prior_hidden_dim,
    lambda_depth_prior=args.lambda_depth_prior,
    depth_prior_use_u_shape_regularization=not args.depth_prior_no_u_shape_regularization,
    depth_gate_enabled=args.depth_gate_enabled,
    depth_gate_alpha_min=args.depth_gate_alpha_min,
    depth_gate_alpha_max=args.depth_gate_alpha_max,
    use_depth_gated_mmd=args.use_depth_gated_mmd,
    dg_mmd_project_dim=args.dg_mmd_project_dim,
    dg_mmd_lambda_center=args.dg_mmd_lambda_center,
    dg_mmd_lambda_priv=args.dg_mmd_lambda_priv,
    dg_mmd_min_priv_weight=args.dg_mmd_min_priv_weight,
    beta_modulation_enabled=args.beta_modulation_enabled,
    beta_modulation_mix_max=args.beta_modulation_mix_max,
    use_privileged_distillation=args.use_privileged_distillation,
    student_adapter_hidden=args.student_adapter_hidden,
)

# 加载预训练权重
def load_pretrained_weights(model, checkpoint_path, component_name):
    """加载预训练权重到指定组件"""
    if isfile(checkpoint_path):
        print(f"Loading {component_name} pretrained weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # 处理检查点格式
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        # 加载权重
        if component_name == "TRUS":
            # 加载TRUS编码器权重
            trus_state_dict = {k.replace('image_encoder.', 'trus_encoder.'): v 
                              for k, v in state_dict.items() 
                              if k.startswith('image_encoder.')}
            model.load_state_dict(trus_state_dict, strict=False)
            print("Loaded TRUS encoder weights")
        elif component_name == "MRI":
            # 加载MRI编码器权重
            mri_state_dict = {k.replace('image_encoder.', 'mri_encoder.'): v 
                             for k, v in state_dict.items() 
                             if k.startswith('image_encoder.')}
            model.load_state_dict(mri_state_dict, strict=False)
            print("Loaded MRI encoder weights")
        elif component_name == "SAM":
            # 加载SAM组件权重
            sam_state_dict = {k: v for k, v in state_dict.items() 
                             if k.startswith(('mask_decoder.', 'prompt_encoder.'))}
            model.load_state_dict(sam_state_dict, strict=False)
            print("Loaded SAM components weights")
    else:
        print(f"Pretrained weights {checkpoint_path} not found for {component_name}")

# 加载预训练权重
print("=== 开始加载预训练权重 ===")
load_pretrained_weights(medsam_lite_model, trus_pretrained_checkpoint, "TRUS")
load_pretrained_weights(medsam_lite_model, mri_pretrained_checkpoint, "MRI")
load_pretrained_weights(medsam_lite_model, trus_pretrained_checkpoint, "SAM")
print("=== 预训练权重加载完成 ===")

def main():
    global best_loss
    medsam_lite_model_local = medsam_lite_model.to(device)
    medsam_lite_model_local.train()

    print(f"Dual-Modal MedSAM size: {sum(p.numel() for p in medsam_lite_model_local.parameters())}")
    model_state_keys = list(medsam_lite_model_local.state_dict().keys())
    has_depth_or_dgmmd_keys = any(
        "depth_prior" in key or "depth_gated_mmd" in key
        for key in model_state_keys
    )
    print("=== Experiment config summary ===")
    print(f"experiment_name: {basename(work_dir)}")
    print(f"SSCA mode: {args.slice_attention_mode}")
    print(f"MRI window radius: {args.mri_window_radius}")
    print(f"fixed position prior enabled: {args.ssca_position_prior_weight > 0}")
    print(f"position prior weight: {args.ssca_position_prior_weight}")
    print(f"position prior sigma: {args.ssca_position_prior_sigma}")
    print(f"slice_corr_loss_weight: {args.slice_corr_loss_weight}")
    print(f"slice_locality_loss_weight(alias): {args.slice_corr_loss_weight}")
    print("slice_locality_loss interpretation: weak local-window positional regularizer, not anatomical correspondence supervision")
    print(f"DG-MMD enabled: {args.use_depth_gated_mmd}")
    print(f"depth gate enabled: {args.depth_gate_enabled}")
    print(f"beta modulation enabled: {args.beta_modulation_enabled}")
    print(f"dynamic bandwidth enabled: {args.use_dynamic_bandwidth_beta}")
    print(f"candidate utility loss weight: {args.slice_utility_loss_weight}")
    print(f"transition loss weight: {args.transition_loss_weight}")
    print(f"transition cls/reg weights: {args.transition_cls_loss_weight}/{args.transition_reg_loss_weight}")
    print(f"mmd_loss_weight: {mmd_loss_weight}")
    print(f"DG-MMD lambda center/priv/min_priv: {args.dg_mmd_lambda_center}/{args.dg_mmd_lambda_priv}/{args.dg_mmd_min_priv_weight}")
    print(f"checkpoint selection rule: save best 2D Dice as dual_modal_best.pth and best online 3D Dice as dual_modal_best_3d.pth; final export uses {args.final_results_checkpoint}")
    print(f"model contains depth/DG-MMD parameter keys: {has_depth_or_dgmmd_keys} (presence does not mean branch is enabled)")
    print("=================================")
    
    # 确保预训练权重已加载
    print("检查预训练权重加载状态:")
    print(f"TRUS编码器参数数量: {sum(p.numel() for p in medsam_lite_model_local.trus_encoder.parameters())}")
    if medsam_lite_model_local.mri_encoder is not None:
        print(f"MRI编码器参数数量: {sum(p.numel() for p in medsam_lite_model_local.mri_encoder.parameters())}")
    else:
        print("MRI编码器: 未使用")
    print(f"跨模态模块参数数量: {sum(p.numel() for p in medsam_lite_model_local.cross_modal_extractor.parameters())}")
    
    # 优化器设置 - 根据是否冻结编码器选择不同策略
    if args.freeze_encoders:
        # 只训练跨模态模块和SAM组件
        trainable_params = []
        trainable_params.extend(list(medsam_lite_model_local.cross_modal_extractor.parameters()))
        trainable_params.extend(list(medsam_lite_model_local.mask_decoder.parameters()))
        trainable_params.extend(list(medsam_lite_model_local.prompt_encoder.parameters()))
        trainable_params.extend(list(medsam_lite_model_local.student_adapter.parameters()))
        
        # 冻结编码器 - 在预训练权重加载后冻结
        for param in medsam_lite_model_local.trus_encoder.parameters():
            param.requires_grad = False
        for param in medsam_lite_model_local.mri_encoder.parameters():
            param.requires_grad = False
            
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        print("[OK] Freeze encoders; training cross-modal modules and SAM components only")
        print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    else:
        # 训练所有参数，但使用不同的学习率
        cross_modal_params = list(medsam_lite_model_local.cross_modal_extractor.parameters())
        
        # 检查MRI编码器是否与TRUS编码器共享参数
        if medsam_lite_model_local.mri_encoder is not None and medsam_lite_model_local.mri_encoder is not medsam_lite_model_local.trus_encoder:
            mri_params = list(medsam_lite_model_local.mri_encoder.parameters())
        else:
            mri_params = []
        
        optimizer = optim.AdamW([
            {'params': cross_modal_params, 'lr': lr, 'name': 'cross_modal'},
            {'params': medsam_lite_model_local.student_adapter.parameters(), 'lr': lr, 'name': 'student_adapter'},
            {'params': medsam_lite_model_local.trus_encoder.parameters(), 'lr': lr * 0.1, 'name': 'trus_encoder'},
            {'params': mri_params, 'lr': lr * 0.1, 'name': 'mri_encoder'} if mri_params else {'params': [], 'lr': lr * 0.1, 'name': 'mri_encoder'},
            {'params': medsam_lite_model_local.mask_decoder.parameters(), 'lr': lr * 0.1, 'name': 'mask_decoder'},
            {'params': medsam_lite_model_local.prompt_encoder.parameters(), 'lr': lr * 0.1, 'name': 'prompt_encoder'},
        ], weight_decay=weight_decay)
        print("训练所有参数，使用分层学习率")
    
    # 学习率调度器
    if args.lr_scheduler == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, cooldown=2, min_lr=1e-7
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-7
        )
    elif args.lr_scheduler == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    
    # 损失函数
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    iou_loss = nn.MSELoss(reduction='mean')
    
    # 混合精度训练
    if args.mixed_precision:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler()
        print("启用混合精度训练")
    else:
        scaler = None

    trus_feature_cache_dir = None
    mri_feature_cache_dir = None
    if args.cache_encoder_features:
        if not args.freeze_encoders:
            raise ValueError("--cache_encoder_features requires -freeze_encoders.")
        if checkpoint and isfile(checkpoint):
            raise ValueError("--cache_encoder_features should not be used with resume checkpoints unless the cache was built from the same encoder weights.")
        feature_cache_root = args.feature_cache_dir or join(work_dir, "feature_cache")
        trus_feature_cache_dir = join(feature_cache_root, "trus")
        mri_feature_cache_dir = join(feature_cache_root, "mri")
        trus_image_paths = sorted(glob(join(trus_data_root, "imgs", "*.npy")))
        mri_image_paths = sorted(glob(join(mri_data_root, "imgs", "*.npy")))
        print(f"[CACHE] Precomputing frozen encoder features: TRUS={len(trus_image_paths)}, MRI={len(mri_image_paths)}")
        cache_encoder_features(
            medsam_lite_model_local.trus_encoder,
            trus_image_paths,
            trus_feature_cache_dir,
            device,
            batch_size=max(1, batch_size * 8),
            regenerate=args.regenerate_feature_cache,
            mixed_precision=args.mixed_precision,
        )
        cache_encoder_features(
            medsam_lite_model_local.mri_encoder,
            mri_image_paths,
            mri_feature_cache_dir,
            device,
            batch_size=max(1, batch_size * 8),
            regenerate=args.regenerate_feature_cache,
            mixed_precision=args.mixed_precision,
        )
    
    # 数据加载器 - 使用高效数据集
    train_dataset = EfficientPairedNpyDataset(
        trus_data_root, mri_data_root, 
        data_aug=False, 
        samples_per_epoch=args.samples_per_epoch,
        mri_window_radius=args.mri_window_radius,
        slice_attention_mode=args.slice_attention_mode,
        pairing_mode=args.pairing_mode,
        trus_feature_cache_dir=trus_feature_cache_dir,
        mri_feature_cache_dir=mri_feature_cache_dir,
        box_mode=args.training_box_mode,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # 避免多进程问题

    # 恢复训练
    if checkpoint and isfile(checkpoint):
        print(f"Resuming from checkpoint {checkpoint}")
        ckpt = torch.load(checkpoint)
        medsam_lite_model_local.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = 1e10

    # 创建TRUS单模态验证数据集（用于推理验证）
    print("创建TRUS单模态验证数据集...")
    paired_npy_val = (
        val_trus_data_root
        and val_mri_data_root
        and isdir(join(val_trus_data_root, "imgs"))
        and isdir(join(val_trus_data_root, "gts"))
        and isdir(join(val_mri_data_root, "imgs"))
        and isdir(join(val_mri_data_root, "gts"))
    )
    if paired_npy_val:
        print("Using deterministic paired NPY validation set.")
        val_dataset = PairedNpyDataset(
            val_trus_data_root,
            val_mri_data_root,
            data_aug=False,
            mri_window_radius=args.mri_window_radius,
            slice_attention_mode=args.slice_attention_mode,
            pairing_mode=args.pairing_mode,
            box_mode=args.training_box_mode,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print(f"Validation slices: {len(val_dataset)}")
    elif val_trus_data_root and isdir(val_trus_data_root):
        # 使用独立验证集
        print("使用独立TRUS验证集...")
        
        # 创建单模态TRUS验证数据集
        class SingleModalTRUSDataset(Dataset):
            def __init__(self, trus_root, image_size=256, bbox_shift=5, box_mode="gt"):
                self.trus_root = trus_root
                self.image_size = image_size
                self.bbox_shift = bbox_shift
                self.box_mode = box_mode
                
                # 查找TRUS数据 - 验证集使用.npz文件
                self.trus_files = sorted(glob(join(trus_root, "*.npz")))
                
                if len(self.trus_files) == 0:
                    raise ValueError(f"TRUS验证集路径不存在或为空: {trus_root}")
                
                print(f"找到 {len(self.trus_files)} 个独立TRUS验证文件")
                
            def __len__(self):
                return len(self.trus_files)
            
            def __getitem__(self, idx):
                # 加载TRUS数据 - 从.npz文件加载
                npz_data = np.load(self.trus_files[idx], 'r', allow_pickle=True)
                img_3D = npz_data['imgs']  # (Num, H, W, 3)
                gt_3D = npz_data['gts']    # (Num, H, W)
                
                # 随机选择一个切片进行验证（而不是固定中间切片）
                num_slices = img_3D.shape[0]
                slice_idx = np.random.randint(0, num_slices)
                trus_img = img_3D[slice_idx]  # (H, W, 3)
                trus_gt = gt_3D[slice_idx]   # (H, W)
                
                # 确保数据格式正确 - 与训练集保持一致
                if len(trus_img.shape) == 2:
                    trus_img = np.repeat(trus_img[:, :, np.newaxis], 3, axis=2)
                
                # 调整图像大小到256x256（与训练集一致）
                import cv2
                trus_img = cv2.resize(trus_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                trus_gt = cv2.resize(trus_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # 归一化
                trus_img = trus_img.astype(np.float32) / 255.0
                
                # 处理标签
                if len(trus_gt.shape) == 3:
                    trus_gt = trus_gt[:, :, 0]
                trus_gt = (trus_gt > 0.5).astype(np.uint8)
                
                # 生成边界框
                y_indices, x_indices = np.where(trus_gt > 0)
                if self.box_mode == "full_image":
                    x_min, x_max = 0, trus_img.shape[1] - 1
                    y_min, y_max = 0, trus_img.shape[0] - 1
                elif len(x_indices) == 0:
                    # 如果没有前景，使用整个图像
                    x_min, x_max = 0, trus_img.shape[1]
                    y_min, y_max = 0, trus_img.shape[0]
                else:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # 添加扰动
                H, W = trus_img.shape[:2]
                if self.box_mode != "full_image":
                    x_min = max(0, x_min - self.bbox_shift)
                    y_min = max(0, y_min - self.bbox_shift)
                    x_max = min(W, x_max + self.bbox_shift)
                    y_max = min(H, y_max + self.bbox_shift)
                
                bbox = np.array([x_min, y_min, x_max, y_max])
                
                # 转换为tensor
                trus_img_tensor = torch.from_numpy(trus_img).permute(2, 0, 1).float()
                trus_gt_tensor = torch.from_numpy(trus_gt).unsqueeze(0).float()
                bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).float()
                
                return {
                    "trus_image": trus_img_tensor,
                    "gt2D": trus_gt_tensor,
                    "bboxes": bbox_tensor,
                    "image_name": basename(self.trus_files[idx])
                }
        
        # 增加验证样本数量 - 每个3D体积采样多个切片
        val_dataset = SingleModalTRUSDataset(
            val_trus_data_root,
            bbox_shift=bbox_shift,
            box_mode=args.training_box_mode,
        )
        # 通过重复采样增加验证样本数量
        val_dataset_multiplied = torch.utils.data.ConcatDataset([val_dataset] * 5)  # 每个体积采样5次
        val_loader = DataLoader(val_dataset_multiplied, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)  # 避免多进程问题
        print(f"使用独立TRUS验证集，原始大小: {len(val_dataset)}, 增强后大小: {len(val_dataset_multiplied)}")
    else:
        # 如果没有独立验证集，使用训练集的一部分
        print("未找到独立TRUS验证集，使用训练集分割...")
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # 重新创建训练数据加载器
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # 避免多进程问题
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)  # 避免多进程问题
        print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
    
    # 生成完整验证结果（nii、CSV、overlay）
    def generate_validation_results(model, val_trus_data_root, device, work_dir, epoch, metric_value, bbox_shift):
        """
        生成验证集中所有原始图像文件的完整分割结果、overlay图像和nii文件
        
        Args:
            model: 模型
            val_trus_data_root: 验证集TRUS数据根目录
            device: 设备
            work_dir: 工作目录
            epoch: 当前epoch
            metric_value: Dice值
            bbox_shift: 边界框扰动
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import pandas as pd
        import SimpleITK as sitk
        from scipy.spatial.distance import directed_hausdorff
        matplotlib.use('Agg')  # 使用非交互式后端
        
        model.eval()
        # 使用固定目录名，每次直接覆盖（因为测试集是固定的）
        results_dir = join(work_dir, "best_dice_results")
        pred_dir = join(results_dir, "predictions")
        overlay_dir = join(results_dir, "overlays")
        nii_dir = join(results_dir, "nii")
        # 如果目录已存在，先清空（确保是干净的状态）
        if exists(results_dir):
            import shutil
            shutil.rmtree(results_dir)
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(nii_dir, exist_ok=True)
        
        # 辅助函数：计算HD-95（优化版本，避免内存问题）
        def compute_hausdorff_distance(pred, gt, percentile=95):
            """计算Hausdorff距离或HD-95（优化版本）"""
            pred_points = np.argwhere(pred > 0)
            gt_points = np.argwhere(gt > 0)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                return float('inf')
            
            # 如果点数太多，进行采样以加快计算
            max_points = 1000
            if len(pred_points) > max_points:
                indices = np.random.choice(len(pred_points), max_points, replace=False)
                pred_points = pred_points[indices]
            if len(gt_points) > max_points:
                indices = np.random.choice(len(gt_points), max_points, replace=False)
                gt_points = gt_points[indices]
            
            # 使用向量化计算提高效率
            distances_pred_to_gt = []
            for p in pred_points:
                dists = np.linalg.norm(gt_points - p, axis=-1)
                distances_pred_to_gt.append(np.min(dists))
            
            distances_gt_to_pred = []
            for g in gt_points:
                dists = np.linalg.norm(pred_points - g, axis=-1)
                distances_gt_to_pred.append(np.min(dists))
            
            all_distances = distances_pred_to_gt + distances_gt_to_pred
            
            if len(all_distances) == 0:
                return float('inf')
            
            if percentile == 100:
                return np.max(all_distances)
            else:
                return np.percentile(all_distances, percentile)
        
        # 辅助函数：显示轮廓
        def show_contour(mask, ax, edgecolor='red', linewidth=2):
            """显示掩码轮廓"""
            if mask.sum() == 0:
                return
            if mask.dtype != np.float32 and mask.dtype != np.float64:
                mask = (mask > 0).astype(np.float32)
            try:
                ax.contour(mask, levels=[0.5], colors=[edgecolor], linewidths=linewidth)
            except:
                import cv2
                contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.squeeze()
                    if len(contour.shape) == 2 and contour.shape[1] == 2:
                        ax.plot(contour[:, 0], contour[:, 1], color=edgecolor, linewidth=linewidth)
        
        # 记录每个样本的指标
        all_results = []
        
        # 从验证数据集的路径获取所有.npz文件
        npz_files = sorted(glob(join(val_trus_data_root, "*.npz")))
        print(f"找到 {len(npz_files)} 个验证图像文件，开始生成完整结果...")
        
        with torch.no_grad():
            for npz_file in tqdm(npz_files, desc="生成验证结果"):
                try:
                    image_name = basename(npz_file)
                    print(f"  处理文件: {image_name}")
                    
                    # 加载3D数据
                    npz_data = np.load(npz_file, 'r', allow_pickle=True)
                    img_3D = npz_data['imgs']  # (Num, H, W, 3)
                    gt_3D = npz_data['gts']    # (Num, H, W)
                    
                    num_slices = img_3D.shape[0]
                    print(f"    切片数量: {num_slices}")
                    pred_3D = np.zeros_like(gt_3D, dtype=np.uint8)
                    slice_results = []
                    
                    # 对每个切片进行推理
                    for slice_idx in range(num_slices):
                        trus_img = img_3D[slice_idx]  # (H, W, 3)
                        trus_gt = gt_3D[slice_idx]   # (H, W)
                        
                        # 确保数据格式正确
                        if len(trus_img.shape) == 2:
                            trus_img = np.repeat(trus_img[:, :, np.newaxis], 3, axis=2)
                        
                        # 保存原始尺寸
                        original_h, original_w = trus_img.shape[:2]
                        
                        # 调整图像大小到256x256
                        trus_img_resized = cv2.resize(trus_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                        trus_gt_resized = cv2.resize(trus_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
                        
                        # 归一化
                        trus_img_resized = trus_img_resized.astype(np.float32) / 255.0
                        
                        # 处理标签
                        if len(trus_gt_resized.shape) == 3:
                            trus_gt_resized = trus_gt_resized[:, :, 0]
                        trus_gt_resized = (trus_gt_resized > 0.5).astype(np.uint8)
                        
                        # 生成边界框
                        y_indices, x_indices = np.where(trus_gt_resized > 0)
                        if len(x_indices) == 0:
                            x_min, x_max = 0, trus_img_resized.shape[1]
                            y_min, y_max = 0, trus_img_resized.shape[0]
                        else:
                            x_min, x_max = np.min(x_indices), np.max(x_indices)
                            y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # 添加扰动
                        H, W = trus_img_resized.shape[:2]
                        x_min = max(0, x_min - bbox_shift)
                        y_min = max(0, y_min - bbox_shift)
                        x_max = min(W, x_max + bbox_shift)
                        y_max = min(H, y_max + bbox_shift)
                        
                        bbox = np.array([x_min, y_min, x_max, y_max])
                        
                        # 转换为tensor
                        trus_img_tensor = torch.from_numpy(trus_img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
                        bbox_tensor = torch.from_numpy(bbox).unsqueeze(0).float().to(device)
                        gt_tensor = torch.from_numpy(trus_gt_resized).unsqueeze(0).unsqueeze(0).float().to(device)
                        
                        # 推理
                        logits_pred, _ = model(trus_img_tensor, None, bbox_tensor, training=False)
                        pred_mask = (torch.sigmoid(logits_pred) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
                        
                        # 将预测结果调整回原始大小
                        pred_original_size = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        pred_3D[slice_idx] = pred_original_size
                        
                        # 计算2D指标（在256x256上计算）
                        pred_binary = pred_mask.astype(np.float32)
                        gt_binary = trus_gt_resized.astype(np.float32)
                        
                        # Dice
                        intersection = np.sum(pred_binary * gt_binary)
                        union = np.sum(pred_binary) + np.sum(gt_binary)
                        dice = 2.0 * intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
                        
                        # IoU
                        intersection_iou = np.sum(pred_binary * gt_binary)
                        union_iou = np.sum(pred_binary) + np.sum(gt_binary) - intersection_iou
                        iou = intersection_iou / union_iou if union_iou > 0 else (1.0 if intersection_iou == 0 else 0.0)
                        
                        # HD-95（如果计算太慢，可以跳过或使用简化版本）
                        try:
                            hd_95 = compute_hausdorff_distance(pred_binary, gt_binary, percentile=95)
                            hd_95 = hd_95 if hd_95 != float('inf') else np.nan
                        except Exception as e:
                            print(f"      警告: 切片 {slice_idx} 的HD-95计算失败: {e}")
                            hd_95 = np.nan
                        
                        slice_results.append({
                            'slice_idx': slice_idx,
                            'dice': dice,
                            'iou': iou,
                            'hd_95': hd_95
                        })
                    
                    # 获取spacing信息
                    if 'spacing' in npz_data:
                        spacing = npz_data['spacing']
                    else:
                        spacing = np.array([1.0, 1.0, 1.0])
                    
                    if isinstance(spacing, np.ndarray):
                        spacing_array = spacing.flatten()
                    else:
                        spacing_array = np.array([spacing] if np.isscalar(spacing) else spacing)
                    
                    if len(spacing_array) == 2:
                        spacing_array = np.append(spacing_array, 1.0)
                    elif len(spacing_array) == 1:
                        spacing_array = np.array([spacing_array[0], spacing_array[0], 1.0])
                    
                    if len(spacing_array) >= 3:
                        spacing_3d = tuple(spacing_array[:3])
                    else:
                        spacing_3d = (1.0, 1.0, 1.0)
                    
                    # 保存完整的3D预测结果
                    np.savez_compressed(
                        join(pred_dir, image_name),
                        segs=pred_3D,
                        gts=gt_3D,
                        spacing=spacing_3d
                    )
                    
                    # 保存NIfTI格式
                    seg_3d_sitk = sitk.GetImageFromArray(pred_3D.astype(np.uint8))
                    seg_3d_sitk.SetSpacing(spacing_3d)
                    nii_pred_name = image_name.replace('.npz', '_pred.nii.gz')
                    nii_pred_path = join(nii_dir, nii_pred_name)
                    sitk.WriteImage(seg_3d_sitk, nii_pred_path)
                    
                    # 同时保存真实标签
                    gt_3d_sitk = sitk.GetImageFromArray(gt_3D.astype(np.uint8))
                    gt_3d_sitk.SetSpacing(spacing_3d)
                    nii_gt_name = image_name.replace('.npz', '_gt.nii.gz')
                    nii_gt_path = join(nii_dir, nii_gt_name)
                    sitk.WriteImage(gt_3d_sitk, nii_gt_path)
                    
                    # 计算3D Dice
                    pred_3d_binary = (pred_3D > 0).astype(np.float32)
                    gt_3d_binary = (gt_3D > 0).astype(np.float32)
                    intersection_3d = np.sum(pred_3d_binary * gt_3d_binary)
                    union_3d = np.sum(pred_3d_binary) + np.sum(gt_3d_binary)
                    dice_3d = 2.0 * intersection_3d / union_3d if union_3d > 0 else (1.0 if intersection_3d == 0 else 0.0)
                    
                    # 计算平均2D指标
                    valid_slices = [s for s in slice_results if not np.isnan(s['hd_95'])]
                    avg_dice = np.mean([s['dice'] for s in slice_results])
                    avg_iou = np.mean([s['iou'] for s in slice_results])
                    avg_hd_95 = np.mean([s['hd_95'] for s in valid_slices]) if valid_slices else np.nan
                    
                    # 保存到结果列表
                    all_results.append({
                        'case': image_name,
                        'num_slices': num_slices,
                        'dice_2d_mean': avg_dice,
                        'iou_2d_mean': avg_iou,
                        'hd_95_mean': avg_hd_95,
                        'dice_3d': dice_3d
                    })
                    
                    # 生成overlay图像（最佳、中间、最差切片）
                    try:
                        if len(slice_results) > 0:
                            best_slice_idx = max(slice_results, key=lambda x: x['dice'])['slice_idx']
                            worst_slice_idx = min(slice_results, key=lambda x: x['dice'])['slice_idx']
                            mid_idx = num_slices // 2
                            
                            slice_indices = [best_slice_idx, mid_idx, worst_slice_idx]
                            slice_names = ['Best', 'Middle', 'Worst']
                            
                            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                            
                            for col, (slice_idx, slice_name) in enumerate(zip(slice_indices, slice_names)):
                                trus_img_slice = img_3D[slice_idx]
                                pred_slice = pred_3D[slice_idx]
                                gt_slice = gt_3D[slice_idx]
                                
                                if len(trus_img_slice.shape) == 2:
                                    trus_img_slice = np.repeat(trus_img_slice[:, :, np.newaxis], 3, axis=2)
                                
                                img_display = (trus_img_slice - trus_img_slice.min()) / (trus_img_slice.max() - trus_img_slice.min() + 1e-8)
                                
                                axes[col].imshow(img_display)
                                
                                gt_binary = (gt_slice > 0).astype(np.float32)
                                pred_binary = (pred_slice > 0).astype(np.float32)
                                
                                if np.sum(gt_binary) > 0:
                                    show_contour(gt_binary, axes[col], edgecolor='red', linewidth=2)
                                
                                if np.sum(pred_binary) > 0:
                                    show_contour(pred_binary, axes[col], edgecolor='green', linewidth=2)
                                
                                slice_dice = next((s['dice'] for s in slice_results if s['slice_idx'] == slice_idx), 0.0)
                                axes[col].set_title(f'{slice_name} Slice {slice_idx}/{num_slices}\nDice: {slice_dice:.4f}\nRed: GT, Green: Pred', 
                                                    fontsize=12)
                                axes[col].axis('off')
                            
                            fig.suptitle(f'{image_name}\n2D Dice: {avg_dice:.4f}, 3D Dice: {dice_3d:.4f}', 
                                        fontsize=14, y=0.98)
                            
                            plt.tight_layout(rect=[0, 0, 1, 0.96])
                            overlay_path = join(overlay_dir, f'{image_name.replace(".npz", "")}_overlay.png')
                            plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            print(f"    已生成overlay: {basename(overlay_path)}")
                    except Exception as e:
                        print(f"      警告: 生成overlay图像失败: {e}")
                        plt.close('all')  # 确保关闭所有图形
                    
                    # 保存每个切片的详细结果到CSV
                    try:
                        slice_df = pd.DataFrame(slice_results)
                        slice_csv_path = join(results_dir, f'{image_name.replace(".npz", "")}_slices.csv')
                        slice_df.to_csv(slice_csv_path, index=False)
                    except Exception as e:
                        print(f"      警告: 保存切片CSV失败: {e}")
                    
                    print(f"    完成: {image_name}")
                    
                except Exception as e:
                    print(f"  错误: 处理文件 {npz_file} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue  # 跳过这个文件，继续处理下一个
        
        # 保存所有病例的汇总结果（包含epoch和dice信息在文件名中，便于追踪）
        df = pd.DataFrame(all_results)
        csv_path = join(results_dir, f'validation_results_epoch_{epoch}_dice_{metric_value:.4f}.csv')
        df.to_csv(csv_path, index=False)
        
        # 同时保存一个固定名称的CSV（最新的结果）
        csv_path_latest = join(results_dir, 'validation_results_latest.csv')
        df.to_csv(csv_path_latest, index=False)
        
        print(f"[OK] 验证集结果已保存到: {results_dir}")
        print(f"   - 预测结果: {pred_dir}")
        print(f"   - Overlay图像: {overlay_dir}")
        print(f"   - NIfTI文件: {nii_dir}")
        print(f"   - 结果统计: {csv_path}")
        print(f"   - 最新结果: {csv_path_latest}")
        print(f"   - 平均2D Dice: {df['dice_2d_mean'].mean():.4f} ± {df['dice_2d_mean'].std():.4f}")
        print(f"   - 平均3D Dice: {df['dice_3d'].mean():.4f} ± {df['dice_3d'].std():.4f}")
        
        model.train()
    
    # 验证函数 - 单模态TRUS推理验证
    def validate_model(model, val_loader, device, loss_weights):
        """验证模型并返回验证loss、Dice、IoU和3D Dice（单模态TRUS推理）"""
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        # 用于计算3D Dice：收集所有预测和GT
        all_pred_3d = []
        all_gt_3d = []
        
        with torch.no_grad():
            for batch in val_loader:
                trus_image = batch["trus_image"].to(device)
                gt2D = batch["gt2D"].to(device)
                boxes = batch["bboxes"].to(device)
                
                # 单模态TRUS推理（不传入MRI）
                logits_pred, iou_pred = model(trus_image, None, boxes, training=False)
                
                # 计算损失（不包含MMD loss，因为推理时没有MRI）
                l_seg = seg_loss(logits_pred, gt2D)
                l_ce = ce_loss(logits_pred, gt2D.float())
                mask_loss = loss_weights['seg'] * l_seg + loss_weights['ce'] * l_ce
                
                iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
                l_iou = iou_loss(iou_pred, iou_gt)
                
                # 总损失（推理时不包含MMD loss）
                total_loss = mask_loss + loss_weights['iou'] * l_iou
                val_loss += total_loss.item()
                
                # 计算2D Dice
                dice = compute_dice_score(logits_pred, gt2D)
                val_dice += dice
                
                # 计算2D IoU
                pred_binary = (torch.sigmoid(logits_pred) > 0.5).float()
                gt_binary = gt2D.float()
                if gt_binary.dim() == 3:
                    gt_binary = gt_binary.unsqueeze(1)
                
                # IoU计算（整体计算）
                pred_flat = pred_binary.flatten()
                gt_flat = gt_binary.flatten()
                intersection = torch.sum(pred_flat * gt_flat)
                union = torch.sum(pred_flat) + torch.sum(gt_flat) - intersection
                if union == 0:
                    iou = 1.0 if intersection == 0 else 0.0
                else:
                    iou = (intersection / union).item()
                val_iou += iou
                
                # 收集3D数据（用于计算3D Dice）
                all_pred_3d.append(pred_binary.cpu())
                all_gt_3d.append(gt_binary.cpu())
        
        # 计算3D Dice（所有batch合并计算）
        if len(all_pred_3d) > 0:
            pred_3d_all = torch.cat(all_pred_3d, dim=0)
            gt_3d_all = torch.cat(all_gt_3d, dim=0)
            pred_3d_flat = pred_3d_all.flatten()
            gt_3d_flat = gt_3d_all.flatten()
            intersection_3d = torch.sum(pred_3d_flat * gt_3d_flat)
            union_3d = torch.sum(pred_3d_flat) + torch.sum(gt_3d_flat)
            if union_3d == 0:
                dice_3d = 1.0 if intersection_3d == 0 else 0.0
            else:
                dice_3d = (2.0 * intersection_3d / union_3d).item()
        else:
            dice_3d = 0.0
        
        model.train()
        return val_loss / len(val_loader), val_dice / len(val_loader), val_iou / len(val_loader), dice_3d

    def write_validation_attention_log(model, output_path):
        if not (val_trus_data_root and val_mri_data_root and isdir(val_trus_data_root) and isdir(val_mri_data_root)):
            print("[INFO] Skipping validation attention log: paired validation MRI root is unavailable.")
            return

        val_pair_dataset = PairedNpyDataset(
            val_trus_data_root,
            val_mri_data_root,
            data_aug=False,
            mri_window_radius=args.mri_window_radius,
            slice_attention_mode=args.slice_attention_mode,
            pairing_mode=args.pairing_mode,
            box_mode=args.training_box_mode,
        )
        val_pair_loader = DataLoader(val_pair_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        radius = args.mri_window_radius
        beta_columns = [f"beta_{offset:+d}".replace("+0", "center").replace("+", "plus_").replace("-", "minus_") for offset in range(-radius, radius + 1)]
        beta_columns = [col.replace("beta_center", "beta_center") for col in beta_columns]
        beta_columns = [
            "beta_center" if offset == 0 else f"beta_minus_{abs(offset)}" if offset < 0 else f"beta_plus_{offset}"
            for offset in range(-radius, radius + 1)
        ]
        fieldnames = [
            "case_id",
            "trus_slice_idx",
            "mri_center_idx",
            "mri_indices",
            "ablation_mode",
            "pairing_mode",
            "mri_window_radius",
            *beta_columns,
            "final_beta_used_for_fusion",
            "raw_content_beta",
            "position_prior_beta",
            "entropy",
            "confidence",
            "relative_depth",
            "rho",
            "u_depth",
            "mmd_center",
            "mmd_priv",
            "mmd_priv_weight",
            "weighted_mmd_total",
            "depth_prior_loss",
            "depth_prior_loss_weighted",
            "dice_2d",
            "iou_2d",
            "hd95_2d",
        ]

        model.eval()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            with torch.no_grad():
                for batch in tqdm(val_pair_loader, desc="Validation attention log"):
                    trus_image = batch["trus_image"].to(device)
                    mri_image = batch["mri_image"].to(device)
                    gt2D = batch["gt2D"].to(device)
                    boxes = batch["bboxes"].to(device)
                    model_out = model(
                        trus_image,
                        mri_image,
                        boxes,
                        training=True,
                        mask_gt=gt2D,
                        mri_valid_mask=batch.get("mri_valid_mask"),
                        transition_target=batch.get("transition_target"),
                        transition_label=batch.get("transition_label"),
                        transition_valid=batch.get("transition_valid"),
                        relative_depth=batch.get("relative_depth"),
                    )
                    logits_pred = (
                        model_out["logits_teacher"]
                        if isinstance(model_out, dict)
                        else model_out[0]
                    )
                    pred_binary = (torch.sigmoid(logits_pred) > 0.5).float()
                    gt_binary = gt2D.float()
                    if gt_binary.dim() == 3:
                        gt_binary = gt_binary.unsqueeze(1)
                    intersection = torch.sum(pred_binary * gt_binary, dim=[1, 2, 3])
                    pred_sum = torch.sum(pred_binary, dim=[1, 2, 3])
                    gt_sum = torch.sum(gt_binary, dim=[1, 2, 3])
                    union_dice = pred_sum + gt_sum
                    dice = torch.where(union_dice == 0, torch.ones_like(union_dice), 2.0 * intersection / union_dice)
                    union_iou = pred_sum + gt_sum - intersection
                    iou = torch.where(union_iou == 0, torch.ones_like(union_iou), intersection / union_iou)

                    info = getattr(model.cross_modal_extractor, "last_slice_attention", {}) or {}
                    beta = info.get("beta")
                    final_beta_used_for_fusion = info.get("final_beta_used_for_fusion", beta)
                    raw_content_beta = info.get("raw_content_beta")
                    position_prior_beta = info.get("position_prior_beta")
                    entropy = info.get("entropy")
                    confidence = info.get("confidence")
                    rho = info.get("rho")
                    u_depth = info.get("u_depth")
                    relative_depth_info = info.get("relative_depth")
                    mmd_center = info.get("mmd_center")
                    mmd_priv = info.get("mmd_priv")
                    mmd_priv_weight = info.get("mmd_priv_weight")
                    weighted_mmd_total = info.get("weighted_mmd_total")
                    depth_prior_loss = info.get("depth_prior_loss")
                    depth_prior_loss_weighted = info.get("depth_prior_loss_weighted")
                    if beta is None:
                        beta = torch.ones(trus_image.size(0), 1, device=device)
                    def _val_item(value, idx):
                        if value is None:
                            return ""
                        if torch.is_tensor(value):
                            value = value.detach().cpu()
                            if value.dim() == 0:
                                return float(value)
                            if value.size(0) == trus_image.size(0):
                                item = value[idx]
                                return float(item.mean()) if item.numel() > 1 else float(item)
                            return value.tolist()
                        if isinstance(value, (list, tuple)):
                            return value[idx] if idx < len(value) else value
                        return value
                    for i in range(trus_image.size(0)):
                        beta_values = beta[i].detach().cpu().tolist()
                        row = {
                            "case_id": batch["case_id"][i] if isinstance(batch["case_id"], (list, tuple)) else batch["case_id"],
                            "trus_slice_idx": int(batch["trus_slice_index"][i]),
                            "mri_center_idx": int(batch["mri_center_index"][i]),
                            "mri_indices": batch["mri_window_indices"][i].detach().cpu().tolist(),
                            "ablation_mode": args.slice_attention_mode,
                            "pairing_mode": args.pairing_mode,
                            "mri_window_radius": radius,
                            "final_beta_used_for_fusion": _val_item(final_beta_used_for_fusion, i),
                            "raw_content_beta": _val_item(raw_content_beta, i),
                            "position_prior_beta": _val_item(position_prior_beta, i),
                            "entropy": float(entropy[i]) if entropy is not None else "",
                            "confidence": float(confidence[i]) if confidence is not None else "",
                            "relative_depth": _val_item(relative_depth_info, i) if relative_depth_info is not None else _val_item(batch.get("relative_depth"), i),
                            "rho": _val_item(rho, i),
                            "u_depth": _val_item(u_depth, i),
                            "mmd_center": _val_item(mmd_center, i),
                            "mmd_priv": _val_item(mmd_priv, i),
                            "mmd_priv_weight": _val_item(mmd_priv_weight, i),
                            "weighted_mmd_total": _val_item(weighted_mmd_total, i),
                            "depth_prior_loss": _val_item(depth_prior_loss, i),
                            "depth_prior_loss_weighted": _val_item(depth_prior_loss_weighted, i),
                            "dice_2d": float(dice[i]),
                            "iou_2d": float(iou[i]),
                            "hd95_2d": "",
                        }
                        for col_idx, col in enumerate(beta_columns):
                            row[col] = beta_values[col_idx] if col_idx < len(beta_values) else ""
                        writer.writerow(row)
        model.train()
    
    # 训练循环
    train_losses = []
    val_losses = []
    val_dices = []
    val_ious = []
    val_dice_3ds = []
    best_val_loss = float('inf')
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_val_dice_3d = -1.0
    early_stopping_counter = 0
    best_loss = float('inf')
    
    # 损失权重字典
    loss_weights = {
        'seg': seg_loss_weight,
        'ce': ce_loss_weight,
        'iou': iou_loss_weight,
        'mmd': mmd_loss_weight
    }

    slice_attention_log_path = join(work_dir, "slice_attention_log.csv")
    slice_attention_epoch_summary_path = join(work_dir, "slice_attention_epoch_summary.csv")
    epoch_slice_attention_records = []

    def _to_list(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    def log_slice_attention(batch, model, epoch, step):
        if args.slice_attention_log_interval <= 0:
            return
        if step % args.slice_attention_log_interval != 0:
            return

        extractor = getattr(model, "cross_modal_extractor", None)
        info = getattr(extractor, "last_slice_attention", None)
        if not info:
            return

        beta = _to_list(info.get("final_beta_used_for_fusion", info.get("beta")))
        raw_content_beta = _to_list(info.get("raw_content_beta"))
        position_prior_beta = _to_list(info.get("position_prior_beta"))
        final_beta_used_for_fusion = _to_list(info.get("final_beta_used_for_fusion", info.get("beta")))
        entropy = _to_list(info.get("entropy"))
        confidence = _to_list(info.get("confidence"))
        fusion_gate = _to_list(info.get("fusion_gate"))
        transition_score = _to_list(info.get("transition_score"))
        neighbor_trust = _to_list(info.get("neighbor_trust"))
        sigma = _to_list(info.get("sigma"))
        candidate_losses = _to_list(info.get("candidate_losses"))
        utility_loss = info.get("slice_utility_loss")
        transition_loss = info.get("transition_loss")
        transition_cls_loss = info.get("transition_cls_loss")
        transition_reg_loss = info.get("transition_reg_loss")
        oracle_best_candidate_dice = _to_list(info.get("oracle_best_candidate_dice"))
        slice_corr_loss = info.get("slice_corr_loss")
        if torch.is_tensor(slice_corr_loss):
            slice_corr_loss = float(slice_corr_loss.detach().cpu())
        if torch.is_tensor(utility_loss):
            utility_loss = float(utility_loss.detach().cpu())
        if torch.is_tensor(transition_loss):
            transition_loss = float(transition_loss.detach().cpu())
        if torch.is_tensor(transition_cls_loss):
            transition_cls_loss = float(transition_cls_loss.detach().cpu())
        if torch.is_tensor(transition_reg_loss):
            transition_reg_loss = float(transition_reg_loss.detach().cpu())
        case_ids = batch.get("case_id", [""] * len(beta))
        trus_slice_indices = _to_list(batch.get("trus_slice_index"))
        mri_center_indices = _to_list(batch.get("mri_center_index"))
        mri_window_indices = _to_list(batch.get("mri_window_indices"))
        mri_valid_masks = _to_list(batch.get("mri_valid_mask"))
        transition_targets = _to_list(batch.get("transition_target"))
        transition_labels = _to_list(batch.get("transition_label"))
        transition_valids = _to_list(batch.get("transition_valid"))
        mask_areas = _to_list(batch.get("mask_area"))
        relative_depths = _to_list(info.get("relative_depth"))
        if relative_depths is None:
            relative_depths = _to_list(batch.get("relative_depth"))
        rho = _to_list(info.get("rho"))
        u_depth = _to_list(info.get("u_depth"))
        mmd_center = _to_list(info.get("mmd_center"))
        mmd_priv = _to_list(info.get("mmd_priv"))
        mmd_priv_weight = _to_list(info.get("mmd_priv_weight"))
        weighted_mmd_total = _to_list(info.get("weighted_mmd_total"))
        depth_prior_loss = _to_list(info.get("depth_prior_loss"))
        depth_prior_loss_weighted = _to_list(info.get("depth_prior_loss_weighted"))

        def _row_value(value, idx):
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return ""
                if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                    return value[0]
                return value[idx] if idx < len(value) else value
            return value

        write_header = not exists(slice_attention_log_path)
        with open(slice_attention_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "step",
                    "mode",
                    "ablation_mode",
                    "pairing_mode",
                    "mri_window_radius",
                    "case_id",
                    "trus_slice_index",
                    "mri_center_index",
                    "mri_window_indices",
                    "mri_valid_mask",
                    "beta",
                    "final_beta_used_for_fusion",
                    "raw_content_beta",
                    "position_prior_beta",
                    "entropy",
                    "confidence",
                    "fusion_gate",
                    "transition_score",
                    "transition_target",
                    "transition_label",
                    "transition_valid",
                    "relative_depth",
                    "rho",
                    "u_depth",
                    "neighbor_trust",
                    "sigma",
                    "mask_area",
                    "candidate_losses",
                    "slice_utility_loss",
                    "slice_locality_loss",
                    "transition_loss",
                    "transition_cls_loss",
                    "transition_reg_loss",
                    "oracle_best_candidate_dice",
                    "slice_corr_loss",
                    "mmd_center",
                    "mmd_priv",
                    "mmd_priv_weight",
                    "weighted_mmd_total",
                    "depth_prior_loss",
                    "depth_prior_loss_weighted",
                ],
            )
            if write_header:
                writer.writeheader()
            for i in range(len(beta)):
                gate_value = fusion_gate[i] if fusion_gate is not None else None
                row = {
                    "epoch": epoch,
                    "step": step,
                    "mode": args.slice_attention_mode,
                    "ablation_mode": args.slice_attention_mode,
                    "pairing_mode": args.pairing_mode,
                    "mri_window_radius": args.mri_window_radius,
                    "case_id": case_ids[i] if isinstance(case_ids, (list, tuple)) else case_ids,
                    "trus_slice_index": trus_slice_indices[i] if trus_slice_indices is not None else "",
                    "mri_center_index": mri_center_indices[i] if mri_center_indices is not None else "",
                    "mri_window_indices": mri_window_indices[i] if mri_window_indices is not None else "",
                    "mri_valid_mask": mri_valid_masks[i] if mri_valid_masks is not None else "",
                    "beta": beta[i],
                    "final_beta_used_for_fusion": final_beta_used_for_fusion[i] if final_beta_used_for_fusion is not None else beta[i],
                    "raw_content_beta": raw_content_beta[i] if raw_content_beta is not None else "",
                    "position_prior_beta": position_prior_beta[i] if position_prior_beta is not None else "",
                    "entropy": entropy[i] if entropy is not None else "",
                    "confidence": confidence[i] if confidence is not None else "",
                    "fusion_gate": gate_value,
                    "transition_score": transition_score[i] if transition_score is not None else "",
                    "transition_target": transition_targets[i] if transition_targets is not None else "",
                    "transition_label": transition_labels[i] if transition_labels is not None else "",
                    "transition_valid": transition_valids[i] if transition_valids is not None else "",
                    "relative_depth": _row_value(relative_depths, i),
                    "rho": _row_value(rho, i),
                    "u_depth": _row_value(u_depth, i),
                    "neighbor_trust": neighbor_trust[i] if neighbor_trust is not None else "",
                    "sigma": sigma[i] if sigma is not None else "",
                    "mask_area": mask_areas[i] if mask_areas is not None else "",
                    "candidate_losses": candidate_losses[i] if candidate_losses is not None else "",
                    "slice_utility_loss": utility_loss if utility_loss is not None else "",
                    "slice_locality_loss": slice_corr_loss if slice_corr_loss is not None else "",
                    "transition_loss": transition_loss if transition_loss is not None else "",
                    "transition_cls_loss": transition_cls_loss if transition_cls_loss is not None else "",
                    "transition_reg_loss": transition_reg_loss if transition_reg_loss is not None else "",
                    "oracle_best_candidate_dice": oracle_best_candidate_dice[i] if oracle_best_candidate_dice is not None else "",
                    "slice_corr_loss": slice_corr_loss if slice_corr_loss is not None else "",
                    "mmd_center": _row_value(mmd_center, i),
                    "mmd_priv": _row_value(mmd_priv, i),
                    "mmd_priv_weight": _row_value(mmd_priv_weight, i),
                    "weighted_mmd_total": _row_value(weighted_mmd_total, i),
                    "depth_prior_loss": _row_value(depth_prior_loss, i),
                    "depth_prior_loss_weighted": _row_value(depth_prior_loss_weighted, i),
                }
                writer.writerow(row)
                epoch_slice_attention_records.append(row)

    def summarize_slice_attention_epoch(epoch):
        if not epoch_slice_attention_records:
            return

        epoch_df = pd.DataFrame(epoch_slice_attention_records)
        epoch_df = epoch_df[epoch_df["epoch"] == epoch].copy()
        if epoch_df.empty:
            return

        def parse_vector(series):
            vectors = []
            for value in series.dropna():
                if isinstance(value, str):
                    try:
                        parsed = ast.literal_eval(value)
                    except Exception:
                        continue
                else:
                    parsed = value
                if isinstance(parsed, (list, tuple)):
                    vectors.append([float(x) for x in parsed])
            return vectors

        def parse_scalar_series(series):
            values = []
            for value in series.dropna():
                parsed = value
                if isinstance(value, str):
                    try:
                        parsed = ast.literal_eval(value)
                    except Exception:
                        parsed = value
                while isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                    parsed = parsed[0]
                if isinstance(parsed, (list, tuple)):
                    flat = np.asarray(parsed, dtype=float).reshape(-1)
                    if flat.size == 0:
                        continue
                    parsed = float(flat.mean())
                try:
                    parsed = float(parsed)
                except Exception:
                    continue
                if np.isfinite(parsed):
                    values.append(parsed)
            return np.asarray(values, dtype=float)

        summary = {"epoch": epoch}
        beta_source_col = "final_beta_used_for_fusion" if "final_beta_used_for_fusion" in epoch_df.columns else "beta"
        beta_vectors = parse_vector(epoch_df[beta_source_col])
        if beta_vectors:
            beta_array = np.asarray(beta_vectors, dtype=float)
            center_idx = beta_array.shape[1] // 2
            summary["beta_mean"] = beta_array.mean(axis=0).tolist()
            summary["beta_std"] = beta_array.std(axis=0).tolist()
            summary["beta_center_mean"] = float(beta_array[:, center_idx].mean())
            summary["beta_center_std"] = float(beta_array[:, center_idx].std())
            neighbor_mass = 1.0 - beta_array[:, center_idx]
            near_mask = np.array([abs(i - center_idx) == 1 for i in range(beta_array.shape[1])], dtype=bool)
            far_mask = np.array([abs(i - center_idx) >= 2 for i in range(beta_array.shape[1])], dtype=bool)
            summary["neighbor_mass_mean"] = float(neighbor_mass.mean())
            summary["neighbor_mass_std"] = float(neighbor_mass.std())
            summary["near_neighbor_mass_mean"] = float(beta_array[:, near_mask].sum(axis=1).mean()) if near_mask.any() else 0.0
            summary["near_neighbor_mass_std"] = float(beta_array[:, near_mask].sum(axis=1).std()) if near_mask.any() else 0.0
            summary["far_neighbor_mass_mean"] = float(beta_array[:, far_mask].sum(axis=1).mean()) if far_mask.any() else 0.0
            summary["far_neighbor_mass_std"] = float(beta_array[:, far_mask].sum(axis=1).std()) if far_mask.any() else 0.0

        for diag_col, prefix in [
            ("raw_content_beta", "raw_content_beta"),
            ("position_prior_beta", "position_prior_beta"),
        ]:
            if diag_col in epoch_df.columns:
                diag_vectors = parse_vector(epoch_df[diag_col])
                if diag_vectors:
                    diag_array = np.asarray(diag_vectors, dtype=float)
                    diag_center_idx = diag_array.shape[1] // 2
                    summary[f"{prefix}_mean"] = diag_array.mean(axis=0).tolist()
                    summary[f"{prefix}_center_mean"] = float(diag_array[:, diag_center_idx].mean())
                    summary[f"{prefix}_center_std"] = float(diag_array[:, diag_center_idx].std())

        for key in [
            "entropy",
            "confidence",
            "transition_score",
            "transition_target",
            "transition_label",
            "neighbor_trust",
            "sigma",
            "slice_utility_loss",
            "slice_locality_loss",
            "slice_corr_loss",
            "transition_loss",
            "transition_cls_loss",
            "transition_reg_loss",
            "oracle_best_candidate_dice",
            "relative_depth",
            "rho",
            "u_depth",
            "mmd_center",
            "mmd_priv",
            "mmd_priv_weight",
            "weighted_mmd_total",
            "depth_prior_loss",
            "depth_prior_loss_weighted",
        ]:
            if key in epoch_df.columns:
                values = parse_scalar_series(epoch_df[key])
                if len(values) > 0:
                    summary[f"{key}_mean"] = float(np.mean(values))
                    summary[f"{key}_std"] = float(np.std(values, ddof=0))

        candidate_vectors = parse_vector(epoch_df["candidate_losses"]) if "candidate_losses" in epoch_df.columns else []
        if candidate_vectors:
            candidate_array = np.asarray(candidate_vectors, dtype=float)
            summary["candidate_loss_mean"] = float(candidate_array.mean())
            summary["candidate_loss_std"] = float(candidate_array.std())

        def safe_corr(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() <= 1:
                return np.nan
            a = a[valid]
            b = b[valid]
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                return np.nan
            return float(np.corrcoef(a, b)[0, 1])

        transition_values = pd.to_numeric(epoch_df.get("transition_target"), errors="coerce")
        if beta_vectors and transition_values.notna().sum() > 1:
            transition_np = transition_values.to_numpy(dtype=float)
            center_beta_np = beta_array[:, center_idx]
            near_mass_np = beta_array[:, near_mask].sum(axis=1) if near_mask.any() else np.zeros(beta_array.shape[0])
            far_mass_np = beta_array[:, far_mask].sum(axis=1) if far_mask.any() else np.zeros(beta_array.shape[0])
            summary["corr_beta_center_transition_target"] = safe_corr(center_beta_np, transition_np)
            summary["corr_near_neighbor_mass_transition_target"] = safe_corr(near_mass_np, transition_np)
            summary["corr_far_neighbor_mass_transition_target"] = safe_corr(far_mass_np, transition_np)
            if "confidence_mean" in summary:
                confidence_np = pd.to_numeric(epoch_df["confidence"], errors="coerce").to_numpy(dtype=float)
                summary["corr_confidence_transition_target"] = safe_corr(confidence_np, transition_np)
            neighbor_mass_np = 1.0 - beta_array[:, center_idx]
            summary["corr_neighbor_mass_transition_target"] = safe_corr(neighbor_mass_np, transition_np)
            if "transition_score" in epoch_df.columns:
                score_np = pd.to_numeric(epoch_df["transition_score"], errors="coerce").to_numpy(dtype=float)
                summary["corr_transition_score_transition_target"] = safe_corr(score_np, transition_np)

        relative_depth_np = parse_scalar_series(epoch_df["relative_depth"]) if "relative_depth" in epoch_df.columns else np.asarray([])
        rho_np = parse_scalar_series(epoch_df["rho"]) if "rho" in epoch_df.columns else np.asarray([])
        u_depth_np = parse_scalar_series(epoch_df["u_depth"]) if "u_depth" in epoch_df.columns else np.asarray([])
        if relative_depth_np.size > 1 and rho_np.size == relative_depth_np.size:
            summary["corr_rho_relative_depth"] = safe_corr(rho_np, relative_depth_np)
            if beta_vectors and beta_array.shape[0] == relative_depth_np.size:
                center_beta_np = beta_array[:, center_idx]
                far_mass_np = beta_array[:, far_mask].sum(axis=1) if far_mask.any() else np.zeros(beta_array.shape[0])
                summary["corr_beta_center_relative_depth"] = safe_corr(center_beta_np, relative_depth_np)
                summary["corr_far_neighbor_mass_relative_depth"] = safe_corr(far_mass_np, relative_depth_np)
            if "mmd_priv_weight" in epoch_df.columns:
                mmd_priv_weight_np = parse_scalar_series(epoch_df["mmd_priv_weight"])
                if mmd_priv_weight_np.size == relative_depth_np.size:
                    summary["corr_mmd_priv_weight_relative_depth"] = safe_corr(mmd_priv_weight_np, relative_depth_np)
                    summary["corr_mmd_priv_weight_rho"] = safe_corr(mmd_priv_weight_np, rho_np)
        if rho_np.size > 1 and u_depth_np.size == rho_np.size:
            summary["corr_rho_u_depth"] = safe_corr(rho_np, u_depth_np)
            if beta_vectors and beta_array.shape[0] == rho_np.size:
                center_beta_np = beta_array[:, center_idx]
                far_mass_np = beta_array[:, far_mask].sum(axis=1) if far_mask.any() else np.zeros(beta_array.shape[0])
                summary["corr_beta_center_rho"] = safe_corr(center_beta_np, rho_np)
                summary["corr_far_neighbor_mass_rho"] = safe_corr(far_mass_np, rho_np)

        transition_labels = pd.to_numeric(epoch_df.get("transition_label"), errors="coerce")
        if "transition_score" in epoch_df.columns and transition_labels.notna().sum() > 1:
            score_np = pd.to_numeric(epoch_df["transition_score"], errors="coerce").to_numpy(dtype=float)
            label_np = transition_labels.to_numpy(dtype=float)
            summary["corr_transition_score_transition_label"] = safe_corr(score_np, label_np)

        write_header = not exists(slice_attention_epoch_summary_path)
        with open(slice_attention_epoch_summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

    def current_slice_corr_loss(model):
        extractor = getattr(model, "cross_modal_extractor", None)
        info = getattr(extractor, "last_slice_attention", None) or {}
        loss_value = info.get("slice_corr_loss")
        if loss_value is None:
            return torch.zeros((), device=device)
        return loss_value.to(device) if torch.is_tensor(loss_value) else torch.tensor(loss_value, device=device)

    def compute_distillation_losses(out, gt2D, model):
        """从双路径输出计算总损失。out为模型forward返回的dict(特权蒸馏)或旧式tuple。
        返回 (total_loss, log_dict)。"""
        # 兼容:若关闭蒸馏(旧式tuple),退回原逻辑
        if not isinstance(out, dict):
            logits_pred, iou_pred, mmd_loss = out
            l_seg = seg_loss(logits_pred, gt2D)
            l_ce = ce_loss(logits_pred, gt2D.float())
            mask_loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce
            iou_gt = cal_iou(torch.sigmoid(logits_pred) > 0.5, gt2D.bool())
            l_iou = iou_loss(iou_pred, iou_gt)
            total = mask_loss + iou_loss_weight * l_iou + mmd_loss_weight * mmd_loss
            total = total + args.slice_corr_loss_weight * current_slice_corr_loss(model)
            return total, {"seg": float(l_seg.detach()), "mmd": float(mmd_loss.detach())}

        logits_s = out["logits_student"]
        iou_s = out["iou_student"]
        logits_t = out["logits_teacher"]
        iou_t = out["iou_teacher"]
        f_s = out["feat_student"]
        f_t = out["feat_teacher"]
        mmd_loss = out["mmd_loss"]
        aux_extra = out["aux_extra"]
        gt_f = gt2D.float()

        # --- Student 主分割损失(作用在推理路径上,这是修掉train/test mismatch的核心) ---
        l_seg_s = seg_loss(logits_s, gt2D)
        l_ce_s = ce_loss(logits_s, gt_f)
        iou_gt_s = cal_iou(torch.sigmoid(logits_s) > 0.5, gt2D.bool())
        l_iou_s = iou_loss(iou_s, iou_gt_s)
        loss_student = seg_loss_weight * l_seg_s + ce_loss_weight * l_ce_s + iou_loss_weight * l_iou_s

        # --- Teacher 监督(gamma),保证蒸馏目标质量 ---
        l_seg_t = seg_loss(logits_t, gt2D)
        l_ce_t = ce_loss(logits_t, gt_f)
        iou_gt_t = cal_iou(torch.sigmoid(logits_t) > 0.5, gt2D.bool())
        l_iou_t = iou_loss(iou_t, iou_gt_t)
        loss_teacher = args.teacher_loss_weight * (
            seg_loss_weight * l_seg_t + ce_loss_weight * l_ce_t + iou_loss_weight * l_iou_t
        )

        # --- 特征级蒸馏:F_student 逼近 F_teacher(detach,防止塌缩) ---
        loss_feat = args.distill_feat_weight * F.mse_loss(f_s, f_t.detach())

        # --- logits级KD:二值分割用基于sigmoid的软标签蒸馏,温度T ---
        T = max(args.distill_kd_temp, 1e-4)
        with torch.no_grad():
            soft_target = torch.sigmoid(logits_t.detach() / T)
        loss_kd = args.distill_kd_weight * (T * T) * F.binary_cross_entropy_with_logits(
            logits_s / T, soft_target
        )

        # --- 辅助:纯mmd(乘mmd_weight) + SSCA其余已加权项(直接加,不再二次乘) + slice_corr ---
        loss_aux = mmd_loss_weight * mmd_loss + aux_extra
        loss_aux = loss_aux + args.slice_corr_loss_weight * current_slice_corr_loss(model)

        total = loss_student + loss_teacher + loss_feat + loss_kd + loss_aux
        log = {
            "seg_s": float(l_seg_s.detach()),
            "seg_t": float(l_seg_t.detach()),
            "feat_kd": float(loss_feat.detach()),
            "logit_kd": float(loss_kd.detach()),
            "mmd": float(mmd_loss.detach()) if torch.is_tensor(mmd_loss) else float(mmd_loss),
        }
        return total, log

    for epoch in range(start_epoch + 1, num_epochs):
        # 开始新epoch，重新生成随机样本
        train_dataset.new_epoch()
        ssca_module = getattr(getattr(medsam_lite_model_local, "cross_modal_extractor", None), "ssca", None)
        if ssca_module is not None:
            ssca_module.current_epoch = epoch
        epoch_info = train_dataset.get_epoch_info()
        print(f"Epoch {epoch}: 使用{epoch_info['current_epoch_samples']}个样本 (利用率: {epoch_info['utilization_rate']:.1f}%)")
        
        epoch_loss = [1e10 for _ in range(len(train_loader))]
        epoch_start_time = time()
        pbar = tqdm(train_loader)
        
        for step, batch in enumerate(pbar):
            trus_image = batch["trus_image"]
            mri_image = batch["mri_image"]
            gt2D = batch["gt2D"]
            boxes = batch["bboxes"]
            
            optimizer.zero_grad()
            trus_image = trus_image.to(device)
            mri_image = mri_image.to(device)
            gt2D = gt2D.to(device)
            boxes = boxes.to(device)
            
            # 前向传播 - 支持混合精度
            if args.mixed_precision:
                with autocast('cuda'):
                    model_out = medsam_lite_model_local(
                        trus_image,
                        mri_image,
                        boxes,
                        training=True,
                        mask_gt=gt2D,
                        mri_valid_mask=batch.get("mri_valid_mask"),
                        transition_target=batch.get("transition_target"),
                        transition_label=batch.get("transition_label"),
                        transition_valid=batch.get("transition_valid"),
                        relative_depth=batch.get("relative_depth"),
                    )
                    log_slice_attention(batch, medsam_lite_model_local, epoch, step)
                    loss, _loss_log = compute_distillation_losses(model_out, gt2D, medsam_lite_model_local)

                epoch_loss[step] = loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # 前向传播
                model_out = medsam_lite_model_local(
                    trus_image,
                    mri_image,
                    boxes,
                    training=True,
                    mask_gt=gt2D,
                    mri_valid_mask=batch.get("mri_valid_mask"),
                    transition_target=batch.get("transition_target"),
                    transition_label=batch.get("transition_label"),
                    transition_valid=batch.get("transition_valid"),
                    relative_depth=batch.get("relative_depth"),
                )
                log_slice_attention(batch, medsam_lite_model_local, epoch, step)
                loss, _loss_log = compute_distillation_losses(model_out, gt2D, medsam_lite_model_local)

                epoch_loss[step] = loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        epoch_end_time = time()
        epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
        train_losses.append(epoch_loss_reduced)
        
        # 验证
        val_loss, val_dice, val_iou, val_dice_3d = validate_model(medsam_lite_model_local, val_loader, device, loss_weights)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_dice_3ds.append(val_dice_3d)
        
        # 学习率调度 - 基于验证loss
        if args.lr_scheduler == "plateau":
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()
        
        # 保存检查点
        model_weights = medsam_lite_model_local.state_dict()
        save_obj = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "train_loss": epoch_loss_reduced,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_dice_3d": val_dice_3d,
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_val_dice_3d": best_val_dice_3d,
        }
        torch.save(save_obj, join(work_dir, "dual_modal_latest.pth"))
        if epoch in diagnostic_checkpoint_epochs:
            torch.save(save_obj, join(work_dir, f"dual_modal_epoch_{epoch}.pth"))
        
        # 基于验证loss进行早停，但基于Dice保存最佳模型
        if val_loss < best_val_loss - args.min_delta:
            print(f"New best validation loss: {best_val_loss:.4f} -> {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Dice_3D: {val_dice_3d:.4f}")
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # 基于Dice保存最佳模型（独立于早停逻辑）
        if val_dice > best_val_dice:
            print(f"New best validation Dice: {best_val_dice:.4f} -> {val_dice:.4f}, Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice_3D: {val_dice_3d:.4f}")
            best_val_dice = val_dice
            
            # 保存最佳模型
            save_obj["best_val_loss"] = best_val_loss
            save_obj["best_val_dice"] = best_val_dice
            save_obj["best_val_iou"] = best_val_iou
            save_obj["best_val_dice_3d"] = best_val_dice_3d
            torch.save(save_obj, join(work_dir, "dual_modal_best.pth"))
            
            # 生成最佳Dice时的完整验证结果（基于验证集评估）
            if val_trus_data_root and isdir(val_trus_data_root) and not args.no_best_result_generation:
                print(f"[INFO] 生成最佳验证集Dice ({val_dice:.4f}, epoch {epoch}) 时的完整结果...")
                print(f"       (使用固定目录名，直接覆盖旧结果)")
                generate_validation_results(medsam_lite_model_local, val_trus_data_root, device, work_dir, epoch, val_dice, bbox_shift)
        
        # 更新最佳IoU和3D Dice（仅记录，不保存模型）
        if val_iou > best_val_iou:
            print(f"New best validation IoU: {best_val_iou:.4f} -> {val_iou:.4f}, Dice: {val_dice:.4f}, Dice_3D: {val_dice_3d:.4f}")
            best_val_iou = val_iou
        
        if val_dice_3d > best_val_dice_3d:
            print(f"New best validation 3D Dice: {best_val_dice_3d:.4f} -> {val_dice_3d:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            best_val_dice_3d = val_dice_3d
            save_obj["best_val_loss"] = best_val_loss
            save_obj["best_val_dice"] = best_val_dice
            save_obj["best_val_iou"] = best_val_iou
            save_obj["best_val_dice_3d"] = best_val_dice_3d
            torch.save(save_obj, join(work_dir, "dual_modal_best_3d.pth"))
        
        if early_stopping_counter >= args.early_stopping_patience:
            summarize_slice_attention_epoch(epoch)
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Best validation Dice: {best_val_dice:.4f}")
            print(f"Best validation IoU: {best_val_iou:.4f}")
            print(f"Best validation 3D Dice: {best_val_dice_3d:.4f}")
            break
            
        # 绘制损失曲线
        plt.figure(figsize=(12, 4))
        
        # 训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # 训练和验证Dice
        plt.subplot(1, 2, 2)
        plt.plot(val_dices, label='Val Dice', color='green')
        plt.title("Validation Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(join(work_dir, "training_curves.png"))
        plt.close()
        summarize_slice_attention_epoch(epoch)
        
        print(f"Epoch {epoch}: Train Loss = {epoch_loss_reduced:.4f}, Val Loss = {val_loss:.4f}, Val Dice = {val_dice:.4f}, Val IoU = {val_iou:.4f}, Val Dice_3D = {val_dice_3d:.4f}")
        print(f"  Best so far - Dice: {best_val_dice:.4f}, IoU: {best_val_iou:.4f}, Dice_3D: {best_val_dice_3d:.4f}")
        print(f"Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

    def load_best_checkpoint_for_outputs():
        checkpoint_name = {
            "best_2d": "dual_modal_best.pth",
            "best_3d": "dual_modal_best_3d.pth",
            "latest": "dual_modal_latest.pth",
        }[args.final_results_checkpoint]
        best_path = join(work_dir, checkpoint_name)
        if isfile(best_path):
            print(f"[INFO] Loading best checkpoint for output generation: {best_path}")
            ckpt = torch.load(best_path, map_location=device)
            medsam_lite_model_local.load_state_dict(ckpt["model"], strict=True)
            metric = ckpt.get("val_dice_3d", best_val_dice_3d) if args.final_results_checkpoint == "best_3d" else ckpt.get("val_dice", best_val_dice)
            return ckpt.get("epoch", -1), metric
        else:
            print("[INFO] Best checkpoint not found; using current model for output generation.")
            return -1, best_val_dice

    final_epoch, final_dice = load_best_checkpoint_for_outputs()
    write_validation_attention_log(
        medsam_lite_model_local,
        join(work_dir, "validation_attention_log.csv"),
    )

    if args.generate_final_results and val_trus_data_root and isdir(val_trus_data_root):
        generate_validation_results(
            medsam_lite_model_local,
            val_trus_data_root,
            device,
            work_dir,
            final_epoch,
            final_dice,
            bbox_shift,
        )

if __name__ == '__main__':
    main()
