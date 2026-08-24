from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from segment_anything.modeling import PromptEncoder, TwoWayTransformer
# MaskDecoder会在单模态分支中导入
from tiny_vit_sam import TinyViT, CrossModalFeatureExtractor
from enhanced_dual_modal import EnhancedMaskDecoder, EnhancedDualModalMedSAM_Lite
from prism_checkpoint_utils import remap_legacy_prism_state_dict_keys
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp
from os.path import dirname
import argparse
import SimpleITK as sitk

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-data_root',
    type=str,
    required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    required=True,
    help='directory to save the prediction',
)
parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default="workdir/lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '--dual_modal',
    action='store_true',
    help='whether to use dual-modal model (trained with MRI guidance)',
)
parser.add_argument(
    '--prism_infer_mode',
    type=str,
    default='student',
    choices=['student', 'student_blend', 'train_consistent', 'self_attn', 'zero_teacher_attn'],
    help='PRISM inference path: student (推荐) uses TRUS + student_adapter, '
         '与特权蒸馏训练的Student路径完全一致; '
         'train_consistent uses only TRUS encoder output; '
         'self_attn applies cross_modal_extractor(trus_feat, trus_feat); '
         'zero_teacher_attn applies cross-modal attention with teacher encoder '
         'fed by a zero image.',
)
parser.add_argument(
    '--student_blend_alpha',
    type=float,
    default=0.5,
    help='blend weight for --prism_infer_mode student_blend: 0 uses raw TRUS encoder features, 1 uses student_adapter features'
)
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay/CT_Abd',
    help='directory to save the overlay image'
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='whether to overwrite the existing prediction'
)
parser.add_argument(
    '--save_nii',
    action='store_true',
    help='whether to save the prediction as NIfTI format (.nii.gz)'
)
parser.add_argument(
    '--mask_threshold',
    type=float,
    default=0.5,
    help='probability threshold used to binarize sigmoid mask predictions; default keeps legacy behavior'
)
parser.add_argument(
    '--keep_largest_component',
    action='store_true',
    help='keep only the largest connected component after thresholding'
)
parser.add_argument(
    '--min_component_area',
    type=int,
    default=0,
    help='remove connected components smaller than this pixel area after thresholding'
)
parser.add_argument(
    '--max_mask_box_ratio',
    type=float,
    default=0.0,
    help='optional upper bound for predicted mask area divided by prompt box area; 0 disables it'
)
parser.add_argument(
    '--min_mask_box_ratio',
    type=float,
    default=0.0,
    help='optional lower bound for predicted mask area divided by prompt box area; 0 disables it'
)
parser.add_argument(
    '--box_mode',
    type=str,
    default='gt',
    choices=['gt', 'full_image'],
    help='box prompt mode: gt uses ground-truth-derived box; full_image uses the whole padded 256x256 image box'
)

args = parser.parse_args()

data_root = args.data_root
pred_save_dir = args.pred_save_dir
save_overlay = args.save_overlay
save_nii = args.save_nii
num_workers = args.num_workers
overwrite = args.overwrite
use_dual_modal = args.dual_modal
prism_infer_mode = args.prism_infer_mode
student_blend_alpha = args.student_blend_alpha
mask_threshold = args.mask_threshold
keep_largest_component = args.keep_largest_component
min_component_area = args.min_component_area
max_mask_box_ratio = args.max_mask_box_ratio
min_mask_box_ratio = args.min_mask_box_ratio
box_mode = args.box_mode
if not 0.0 <= student_blend_alpha <= 1.0:
    raise ValueError(f'--student_blend_alpha must be in [0, 1], got {student_blend_alpha}')
if not 0.0 <= mask_threshold <= 1.0:
    raise ValueError(f'--mask_threshold must be in [0, 1], got {mask_threshold}')
if max_mask_box_ratio < 0.0:
    raise ValueError(f'--max_mask_box_ratio must be non-negative, got {max_mask_box_ratio}')
if min_mask_box_ratio < 0.0:
    raise ValueError(f'--min_mask_box_ratio must be non-negative, got {min_mask_box_ratio}')
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
makedirs(pred_save_dir, exist_ok=True)
bbox_shift = 5
device = torch.device(args.device)
gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
print(f"找到 {len(gt_path_files)} 个.npz文件")
if len(gt_path_files) == 0:
    print(f"警告: 在 {data_root} 中没有找到.npz文件")
    print("请检查数据路径是否正确")
else:
    print(f"前3个文件: {gt_path_files[:3]}")
image_size = 256

def resize_longest_side(image, target_length):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


class DualModalMedSAM_Lite(nn.Module):
    """
    双模态MedSAM模型 - 推理版本
    推理时仅使用TRUS数据
    """
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                use_cross_modal=True
                ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        
        # 双编码器
        self.trus_encoder = image_encoder
        self.mri_encoder = deepcopy(image_encoder) if use_cross_modal else None
        
        # 跨模态特征提取器
        if use_cross_modal:
            self.cross_modal_extractor = CrossModalFeatureExtractor(
                in_channels=256, 
                num_heads=8, 
                mmd_weight=0.1
            )
        
        # SAM组件
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, trus_image, mri_image, boxes, training=True):
        """
        推理时只使用TRUS图像
        """
        # TRUS特征编码
        trus_feat = self.trus_encoder(trus_image)  # (B, 256, 64, 64)
        
        if training and self.use_cross_modal and mri_image is not None:
            # 训练时: MRI引导TRUS特征学习
            mri_feat = self.mri_encoder(mri_image)  # (B, 256, 64, 64)
            enhanced_trus_feat, mmd_loss = self.cross_modal_extractor(trus_feat, mri_feat)
            image_embedding = enhanced_trus_feat
        elif self.use_cross_modal:
            # 推理时: 使用训练好的跨模态特征提取器(无MRI输入)
            # 使用TRUS特征作为Query和Key/Value,利用训练好的注意力机制
            enhanced_trus_feat, _ = self.cross_modal_extractor(trus_feat, trus_feat)
            image_embedding = enhanced_trus_feat
            mmd_loss = None
        else:
            # 单模态推理
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


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """显示掩码（填充方式，已弃用，改用show_contour）"""
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_contour(mask, ax, edgecolor='red', linewidth=2):
    """显示掩码轮廓（描线方式，便于直观对比）"""
    if mask.sum() == 0:
        return
    
    # 确保mask是浮点数类型（matplotlib.contour需要）
    if mask.dtype != np.float32 and mask.dtype != np.float64:
        mask = (mask > 0).astype(np.float32)
    
    # 使用matplotlib的contour绘制轮廓（更简单高效）
    # 在0.5处绘制轮廓线（二值图像在0和1之间）
    ax.contour(mask, levels=[0.5], colors=[edgecolor], linewidths=linewidth)


def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_box(box, new_size, original_size):
    """
    Revert box coordinates from scale at 256 to original scale

    Parameters
    ----------
    box : np.ndarray
        box coordinates at 256 scale
    new_size : tuple
        Image shape with the longest edge resized to 256
    original_size : tuple
        Original image shape

    Returns
    -------
    np.ndarray
        box coordinates at original scale
    """
    new_box = np.zeros_like(box)
    ratio = max(original_size) / max(new_size)
    for i in range(len(box)):
       new_box[i] = int(box[i] * ratio)

    return new_box


def apply_box_area_constraint(prob_map, mask, box_256, new_size, original_size):
    if max_mask_box_ratio <= 0.0 and min_mask_box_ratio <= 0.0:
        return mask

    H, W = prob_map.shape[:2]
    box = resize_box(box_256.astype(np.int32), new_size, original_size)
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    box_area = max(1, (x1 - x0) * (y1 - y0))

    current_area = int(mask.sum())
    target_area = None
    if max_mask_box_ratio > 0.0:
        max_area = max(1, int(round(max_mask_box_ratio * box_area)))
        if current_area > max_area:
            target_area = max_area
    if target_area is None and min_mask_box_ratio > 0.0:
        min_area = max(1, int(round(min_mask_box_ratio * box_area)))
        if current_area < min_area:
            target_area = min_area

    if target_area is None:
        return mask

    if current_area > target_area:
        pool = mask.astype(bool)
    else:
        pool = np.zeros_like(mask, dtype=bool)
        pool[y0:y1, x0:x1] = True

    pool_indices = np.flatnonzero(pool.reshape(-1))
    if len(pool_indices) == 0:
        return mask
    target_area = min(target_area, len(pool_indices))
    scores = prob_map.reshape(-1)[pool_indices]
    if target_area <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    top_local = np.argpartition(scores, -target_area)[-target_area:]
    selected = pool_indices[top_local]
    constrained = np.zeros_like(mask, dtype=np.uint8).reshape(-1)
    constrained[selected] = 1
    return constrained.reshape(mask.shape)


def postprocess_binary_mask(mask):
    """Optional deployment-time cleanup; no-op unless explicitly enabled."""
    if not keep_largest_component and min_component_area <= 0:
        return mask
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    component_ids = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            component_ids.append((label_id, area))

    if not component_ids:
        return np.zeros_like(mask, dtype=np.uint8)

    if keep_largest_component:
        largest_id = max(component_ids, key=lambda x: x[1])[0]
        return (labels == largest_id).astype(np.uint8)

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_id, _ in component_ids:
        cleaned[labels == label_id] = 1
    return cleaned


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > mask_threshold).astype(np.uint8)
    medsam_seg = apply_box_area_constraint(low_res_pred, medsam_seg, box_256, new_size, original_size)
    medsam_seg = postprocess_binary_mask(medsam_seg)

    return medsam_seg

def get_bbox(gt2D, bbox_shift=5):
    assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
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

# 根据是否使用双模态模型选择不同的mask decoder
if use_dual_modal:
    # 双模态模型使用EnhancedMaskDecoder
    print("使用双模态模型进行推理 (仅TRUS输入)")
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
    medsam_lite_model = EnhancedDualModalMedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
        use_cross_modal=True,
        use_src_enhancement=True
    )
else:
    # 单模态模型使用原始的MaskDecoder（与train_one_gpu.py一致）
    print("使用单模态模型进行推理")
    from segment_anything.modeling import MaskDecoder
    medsam_lite_mask_decoder = MaskDecoder(
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
    )
    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder
    )

# PyTorch 2.6+ 默认weights_only=True，需要设置为False以加载包含numpy对象的checkpoint
medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu', weights_only=False)

# 处理检查点格式
if 'model' in medsam_lite_checkpoint:
    # 如果是训练保存的检查点格式
    model_state_dict = medsam_lite_checkpoint['model']
    print(f"加载训练检查点: Epoch {medsam_lite_checkpoint.get('epoch', 'unknown')}")
else:
    # 如果是直接的模型权重
    model_state_dict = medsam_lite_checkpoint

# 兼容旧版 PRISM checkpoint 命名:
# cross_modal_extractor.adaptive_fusion.* -> cross_modal_extractor.fusion.*
if use_dual_modal:
    model_state_dict = remap_legacy_prism_state_dict_keys(model_state_dict)

# 尝试加载，如果失败则使用strict=False（允许缺失某些键，如BatchNorm的running stats）
try:
    medsam_lite_model.load_state_dict(model_state_dict, strict=True)
except RuntimeError as e:
    print(f"警告: 严格加载失败，尝试宽松加载: {e}")
    print("使用strict=False加载（忽略缺失的键）...")
    medsam_lite_model.load_state_dict(model_state_dict, strict=False)
medsam_lite_model.to(device)
medsam_lite_model.eval()


def MedSAM_infer_npz(gt_path_file):
    """
    简单推理版本：从真实标签生成边界框，不使用渐进式推理
    对每个切片独立处理，边界框从该切片的真实标签生成
    """
    npz_name = basename(gt_path_file)
    task_folder = basename(dirname(gt_path_file))
    makedirs(join(pred_save_dir, task_folder), exist_ok=True)
    if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
        npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
        img_3D = npz_data['imgs'] # (Num, H, W)
        gt_3D = npz_data['gts'] # (Num, H, W)
        spacing = npz_data['spacing']
        seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, H, W)
        box_list = [dict() for _ in range(img_3D.shape[0])]

        # 遍历所有切片，独立处理每个切片（不使用前一个切片的预测结果）
        num_slices = img_3D.shape[0]
        print(f"  处理 {npz_name}: 共 {num_slices} 个切片")
        
        for i in range(num_slices):
            img_2d = img_3D[i,:,:] # (H, W)
            H, W = img_2d.shape[:2]
            img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1) # (H, W, 3)

            ## MedSAM Lite preprocessing（与官方代码和train_one_gpu.py保持一致）
            img_256 = resize_longest_side(img_3c, 256)
            newh, neww = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                if use_dual_modal:
                    trus_feat = medsam_lite_model.trus_encoder(img_256_tensor)
                    if prism_infer_mode == 'student':
                        # 特权蒸馏Student路径(推荐):与训练Student完全一致,推理只需TRUS。
                        image_embedding = medsam_lite_model._student_feature(trus_feat)
                    elif prism_infer_mode == 'student_blend':
                        student_feat = medsam_lite_model._student_feature(trus_feat)
                        image_embedding = (1.0 - student_blend_alpha) * trus_feat + student_blend_alpha * student_feat
                    elif prism_infer_mode == 'train_consistent':
                        image_embedding = trus_feat
                    elif prism_infer_mode == 'self_attn':
                        enhanced_trus_feat, _ = medsam_lite_model.cross_modal_extractor(trus_feat, trus_feat)
                        image_embedding = enhanced_trus_feat
                    elif prism_infer_mode == 'zero_teacher_attn':
                        zero_teacher_image = torch.zeros_like(img_256_tensor)
                        mri_feat = medsam_lite_model.mri_encoder(zero_teacher_image)
                        enhanced_trus_feat, _ = medsam_lite_model.cross_modal_extractor(trus_feat, mri_feat)
                        image_embedding = enhanced_trus_feat
                    else:
                        raise ValueError(f'Unknown prism_infer_mode: {prism_infer_mode}')
                else:
                    # 单模态模型推理
                    image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

            # 从真实标签生成边界框（与官方代码保持一致）
            gt = gt_3D[i,:,:] # (H, W)
            label_ids = [1] if box_mode == 'full_image' else np.unique(gt)[1:]
            for label_id in label_ids:
                gt2D = np.uint8(gt == label_id) # only one label, (H, W)
                if gt2D.shape != (newh, neww):
                    gt2D_resize = cv2.resize(
                        gt2D.astype(np.uint8), (neww, newh),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                else:
                    gt2D_resize = gt2D.astype(np.uint8)
                gt2D_padded = pad_image(gt2D_resize, 256) ## (256, 256)
                if box_mode == 'full_image':
                    box = np.array([0, 0, 255, 255], dtype=np.int32)
                elif np.sum(gt2D_padded) > 0:
                    box = get_bbox(gt2D_padded, bbox_shift) # (4,)
                else:
                    continue
                sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
                seg_3D[i, sam_mask>0] = label_id
                box_list[i][label_id] = box

        # 保存所有切片的预测结果
        label_ids = np.unique(gt_3D)[1:]
        output_path = join(pred_save_dir, task_folder, npz_name)
        np.savez_compressed(
            output_path,
            segs=seg_3D,  # 所有切片的预测结果 (Num, H, W)
            gts=gt_3D,   # 所有切片的真实标签 (Num, H, W)
            spacing=spacing
        )
        print(f"  [OK] 已保存: {npz_name} (形状: {seg_3D.shape}, 切片数: {seg_3D.shape[0]})")

        # 保存为NIfTI格式（如果启用）
        if save_nii:
            nii_dir = join(pred_save_dir, task_folder, "nii")
            makedirs(nii_dir, exist_ok=True)
            
            # 确保spacing是正确的格式
            if isinstance(spacing, np.ndarray):
                spacing_array = spacing.flatten()
            else:
                spacing_array = np.array([spacing] if np.isscalar(spacing) else spacing)
            
            # 如果spacing只有2个元素，添加第3个（切片间距）
            if len(spacing_array) == 2:
                spacing_array = np.append(spacing_array, 1.0)  # 默认切片间距为1.0
            elif len(spacing_array) == 1:
                spacing_array = np.array([spacing_array[0], spacing_array[0], 1.0])
            
            # 确保spacing有3个元素
            if len(spacing_array) >= 3:
                spacing_3d = tuple(spacing_array[:3])
            else:
                spacing_3d = (1.0, 1.0, 1.0)  # 默认值
            
            # 将seg_3D从(Num, H, W)转换为SimpleITK格式
            # SimpleITK期望的格式是(Z, H, W)，即(Num, H, W)
            seg_3d_sitk = sitk.GetImageFromArray(seg_3D.astype(np.uint8))
            seg_3d_sitk.SetSpacing(spacing_3d)
            
            # 保存预测结果
            nii_name = npz_name.replace('.npz', '_pred.nii.gz')
            nii_path = join(nii_dir, nii_name)
            sitk.WriteImage(seg_3d_sitk, nii_path)
            
            # 同时保存真实标签（可选，用于对比）
            gt_3d_sitk = sitk.GetImageFromArray(gt_3D.astype(np.uint8))
            gt_3d_sitk.SetSpacing(spacing_3d)
            nii_gt_name = npz_name.replace('.npz', '_gt.nii.gz')
            nii_gt_path = join(nii_dir, nii_gt_name)
            sitk.WriteImage(gt_3d_sitk, nii_gt_path)
            
            print(f"  [OK] 已保存NIfTI格式: {nii_name} 和 {nii_gt_name} (spacing: {spacing_3d})")

        # 生成所有切片的overlay图像（使用轮廓描线）
        if save_overlay:
            case_name = npz_name.split(".")[0]
            case_overlay_dir = join(png_save_dir, case_name)
            makedirs(case_overlay_dir, exist_ok=True)
            
            print(f"  生成 {num_slices} 个切片的overlay图像...")
            
            # 为每个切片生成overlay
            for slice_idx in range(num_slices):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原始图像
                axes[0].imshow(img_3D[slice_idx], cmap='gray')
                axes[0].set_title(f"Image (Slice {slice_idx})")
                axes[0].axis('off')
                
                # Ground Truth（红色轮廓）
                axes[1].imshow(img_3D[slice_idx], cmap='gray')
                gt_slice = gt_3D[slice_idx]
                gt_binary = (gt_slice > 0).astype(np.uint8)
                if np.sum(gt_binary) > 0:
                    show_contour(gt_binary, axes[1], edgecolor='red', linewidth=2)
                axes[1].set_title(f"Ground Truth (Slice {slice_idx})")
                axes[1].axis('off')
                
                # Prediction（绿色轮廓）
                axes[2].imshow(img_3D[slice_idx], cmap='gray')
                pred_slice = seg_3D[slice_idx]
                pred_binary = (pred_slice > 0).astype(np.uint8)
                if np.sum(pred_binary) > 0:
                    show_contour(pred_binary, axes[2], edgecolor='green', linewidth=2)
                axes[2].set_title(f"Prediction (Slice {slice_idx})")
                axes[2].axis('off')
                
                plt.tight_layout()
                overlay_filename = join(case_overlay_dir, f"slice_{slice_idx:03d}.png")
                plt.savefig(overlay_filename, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"  [OK] 已生成 {num_slices} 个切片的overlay: {case_overlay_dir}")

if __name__ == '__main__':
    num_workers = num_workers
    if num_workers <= 1:
        with tqdm(total=len(gt_path_files)) as pbar:
            for gt_path_file in gt_path_files:
                MedSAM_infer_npz(gt_path_file)
                pbar.update()
    else:
        mp.set_start_method('spawn')
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(gt_path_files)) as pbar:
                for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
                    pbar.update()
