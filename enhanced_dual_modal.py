import torch
import torch.nn as nn
from segment_anything.modeling import MaskDecoder
from tiny_vit_sam import CrossModalFeatureExtractor
from copy import deepcopy

class EnhancedMaskDecoder(MaskDecoder):
    """
    继承自 SAM 的 MaskDecoder，增加了源域增强 (Source Enhancement) 的接口支持。
    在推理阶段，逻辑通常与原始 MaskDecoder 一致，主要是在训练阶段会有差异。
    """
    def __init__(self, use_src_enhancement=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_src_enhancement = use_src_enhancement

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        # 直接调用父类的 forward，保持 SAM 的标准解码逻辑
        return super().forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

class EnhancedDualModalMedSAM_Lite(nn.Module):
    """
    双模态 MedSAM 模型结构
    训练时使用MRI和TRUS双模态,推理时仅需TRUS
    """
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        use_cross_modal=True,
        use_src_enhancement=True,
        use_adaptive_fusion=True,
        use_fusion=True,
        slice_attention_mode="index_pairing",
        ssca_descriptor_dim=128,
        ssca_beta_temperature=0.1,
        ssca_min_confidence=0.2,
        ssca_position_prior_weight=0.0,
        ssca_position_prior_sigma=0.75,
        ssca_max_window_size=7,
        slice_corr_loss_weight=0.0,
        slice_corr_prior_sigma=0.75,
        ssca_use_box_aware_pooling=False,
        ssca_boundary_ring_width=3,
        slice_utility_loss_weight=0.0,
        slice_utility_temperature=0.5,
        use_transition_aware_beta=False,
        transition_loss_weight=0.0,
        jump_logit_penalty=1.0,
        use_reliability_gate=False,
        reliability_use_candidate_agreement=True,
        transition_cls_loss_weight=0.05,
        transition_reg_loss_weight=0.01,
        use_dynamic_bandwidth_beta=False,
        sigma_min=0.30,
        sigma_max=1.25,
        dynamic_bandwidth_use_gt_transition_prob=0.0,
        dynamic_bandwidth_warmup_epochs=0,
        dynamic_beta_mode="content_plus_prior",
        dynamic_prior_weight=1.0,
        use_neighbor_residual_fusion=False,
        use_gt_transition_for_beta=False,
        use_gt_transition_for_neighbor_trust=False,
        depth_prior_enabled=False,
        depth_embed_dim=32,
        depth_prior_hidden_dim=64,
        lambda_depth_prior=0.01,
        depth_prior_use_u_shape_regularization=True,
        depth_gate_enabled=False,
        depth_gate_alpha_min=0.05,
        depth_gate_alpha_max=0.60,
        use_depth_gated_mmd=False,
        dg_mmd_project_dim=128,
        dg_mmd_lambda_center=0.005,
        dg_mmd_lambda_priv=0.005,
        dg_mmd_min_priv_weight=0.2,
        beta_modulation_enabled=False,
        beta_modulation_mix_max=0.0,
        use_privileged_distillation=True,
        student_adapter_hidden=256,
    ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        self.use_src_enhancement = use_src_enhancement
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_fusion = use_fusion
        self.slice_attention_mode = slice_attention_mode
        # 特权蒸馏(LUPI):训练时Teacher用TRUS+MRI走SSCA,Student只用TRUS走轻量adapter,
        # 蒸馏让Student逼近Teacher;推理只跑Student,彻底消除原来的train/test路径不一致。
        self.use_privileged_distillation = bool(use_privileged_distillation)
        
        # 1. 图像编码器
        self.trus_encoder = image_encoder
        # 如果使用跨模态，则复制一份作为 MRI 编码器
        self.mri_encoder = deepcopy(image_encoder) if use_cross_modal else None
        
        # 2. 跨模态特征提取器 (MMD Loss & Feature Enhancement)
        if use_cross_modal:
            # 注意：mmd_weight 在训练时从外部传入，这里使用默认值
            # 实际使用时会在 train_dual_modal.py 中通过 CrossModalFeatureExtractor 的参数控制
            self.cross_modal_extractor = CrossModalFeatureExtractor(
                in_channels=256,
                num_heads=8,
                mmd_weight=0.1,  # 默认值，训练时会根据配置调整
                use_adaptive_fusion=use_adaptive_fusion,
                use_fusion=use_fusion,
                ssca_descriptor_dim=ssca_descriptor_dim,
                ssca_beta_temperature=ssca_beta_temperature,
                ssca_min_confidence=ssca_min_confidence,
                ssca_position_prior_weight=ssca_position_prior_weight,
                ssca_position_prior_sigma=ssca_position_prior_sigma,
                ssca_max_window_size=ssca_max_window_size,
                slice_corr_loss_weight=slice_corr_loss_weight,
                slice_corr_prior_sigma=slice_corr_prior_sigma,
                ssca_use_box_aware_pooling=ssca_use_box_aware_pooling,
                ssca_boundary_ring_width=ssca_boundary_ring_width,
                slice_utility_loss_weight=slice_utility_loss_weight,
                slice_utility_temperature=slice_utility_temperature,
                use_transition_aware_beta=use_transition_aware_beta,
                transition_loss_weight=transition_loss_weight,
                jump_logit_penalty=jump_logit_penalty,
                use_reliability_gate=use_reliability_gate,
                reliability_use_candidate_agreement=reliability_use_candidate_agreement,
                transition_cls_loss_weight=transition_cls_loss_weight,
                transition_reg_loss_weight=transition_reg_loss_weight,
                use_dynamic_bandwidth_beta=use_dynamic_bandwidth_beta,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                dynamic_bandwidth_use_gt_transition_prob=dynamic_bandwidth_use_gt_transition_prob,
                dynamic_bandwidth_warmup_epochs=dynamic_bandwidth_warmup_epochs,
                dynamic_beta_mode=dynamic_beta_mode,
                dynamic_prior_weight=dynamic_prior_weight,
                use_neighbor_residual_fusion=use_neighbor_residual_fusion,
                use_gt_transition_for_beta=use_gt_transition_for_beta,
                use_gt_transition_for_neighbor_trust=use_gt_transition_for_neighbor_trust,
                depth_prior_enabled=depth_prior_enabled,
                depth_embed_dim=depth_embed_dim,
                depth_prior_hidden_dim=depth_prior_hidden_dim,
                lambda_depth_prior=lambda_depth_prior,
                depth_prior_use_u_shape_regularization=depth_prior_use_u_shape_regularization,
                depth_gate_enabled=depth_gate_enabled,
                depth_gate_alpha_min=depth_gate_alpha_min,
                depth_gate_alpha_max=depth_gate_alpha_max,
                use_depth_gated_mmd=use_depth_gated_mmd,
                dg_mmd_project_dim=dg_mmd_project_dim,
                dg_mmd_lambda_center=dg_mmd_lambda_center,
                dg_mmd_lambda_priv=dg_mmd_lambda_priv,
                dg_mmd_min_priv_weight=dg_mmd_min_priv_weight,
                beta_modulation_enabled=beta_modulation_enabled,
                beta_modulation_mix_max=beta_modulation_mix_max,
            )
            
        # 3. SAM 组件
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # 4. Student adapter:推理路径专用。残差形式 F_student = trus_feat + adapter(trus_feat)。
        # 让adapter承担"模仿被MRI增强后特征"的容量,trus_feat保持原始语义供Teacher的SSCA使用,
        # 避免编码器被两个目标互相拉扯。推理时只多两层conv,几乎零开销。
        self.student_adapter = nn.Sequential(
            nn.Conv2d(256, student_adapter_hidden, 3, padding=1),
            nn.GroupNorm(32, student_adapter_hidden),
            nn.GELU(),
            nn.Conv2d(student_adapter_hidden, 256, 3, padding=1),
        )

    def _student_feature(self, trus_feat):
        return trus_feat + self.student_adapter(trus_feat)

    def _decode(self, image_embedding, boxes):
        if boxes is not None:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=boxes, masks=None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            return low_res_masks, iou_predictions
        return None, None

    def forward(
        self,
        trus_image,
        mri_image=None,
        boxes=None,
        training=False,
        mask_gt=None,
        mri_valid_mask=None,
        transition_target=None,
        transition_label=None,
        transition_valid=None,
        relative_depth=None,
    ):
        """
        Args:
            trus_image: TRUS图像 (B, 3, H, W)
            mri_image: MRI图像 (B, 3, H, W) - 训练时使用,推理时可为None
            boxes: 边界框 (B, 1, 4) - 可选
            training: 是否为训练模式
        """
        # 1. 获取TRUS特征
        def _is_encoded_feature(tensor):
            return (
                tensor is not None
                and tensor.dim() == 4
                and tensor.shape[1:] == (256, 64, 64)
            )

        def _is_encoded_feature_set(tensor):
            return (
                tensor is not None
                and tensor.dim() == 5
                and tensor.shape[2:] == (256, 64, 64)
            )

        if _is_encoded_feature(trus_image):
            trus_feat = trus_image
        else:
            trus_feat = self.trus_encoder(trus_image)  # (B, 256, 64, 64)

        # ===== 推理 / 验证:只走 Student 路径(纯 TRUS),与训练 Student 完全一致 =====
        if not (training and self.use_cross_modal and mri_image is not None):
            f_student = self._student_feature(trus_feat)
            low_res_masks, iou_predictions = self._decode(f_student, boxes)
            return low_res_masks, iou_predictions

        # ===== 训练:Teacher(TRUS+MRI 走 SSCA) + Student(纯 TRUS 走 adapter) =====
        # Teacher 编码 MRI
        if _is_encoded_feature_set(mri_image) or _is_encoded_feature(mri_image):
            mri_feat = mri_image
        elif mri_image.dim() == 5:
            B, S, C, H, W = mri_image.shape
            mri_feat = self.mri_encoder(mri_image.reshape(B * S, C, H, W))
            _, feat_C, feat_H, feat_W = mri_feat.shape
            mri_feat = mri_feat.reshape(B, S, feat_C, feat_H, feat_W)
        else:
            mri_feat = self.mri_encoder(mri_image)  # (B, 256, 64, 64)

        # Teacher 特征(经 SSCA + fusion)。cross_modal_extractor 在 5D+return_loss 下返回
        # (fused, mmd, aux_weighted_extra);4D 旧路径返回 (fused, mmd)。两种都兼容。
        cm_out = self.cross_modal_extractor(
            trus_feat,
            mri_feat,
            slice_attention_mode=self.slice_attention_mode,
            boxes=boxes,
            image_hw=(256, 256),
            mask_gt=mask_gt,
            mri_valid_mask=mri_valid_mask,
            transition_target=transition_target,
            transition_label=transition_label,
            transition_valid=transition_valid,
            relative_depth=relative_depth,
        )
        if isinstance(cm_out, (tuple, list)) and len(cm_out) == 3:
            f_teacher, mmd_loss, aux_extra = cm_out
        else:
            f_teacher, mmd_loss = cm_out
            aux_extra = trus_feat.new_zeros(())

        # Student 特征(纯 TRUS,推理时唯一路径)
        f_student = self._student_feature(trus_feat)

        # 两路共享同一 decoder/prompt_encoder
        logits_t, iou_t = self._decode(f_teacher, boxes)
        logits_s, iou_s = self._decode(f_student, boxes)

        # 返回 student 主输出 + teacher 输出 + 蒸馏所需特征 + 辅助损失
        return {
            "logits_student": logits_s,
            "iou_student": iou_s,
            "logits_teacher": logits_t,
            "iou_teacher": iou_t,
            "feat_student": f_student,
            "feat_teacher": f_teacher,
            "mmd_loss": mmd_loss,
            "aux_extra": aux_extra,
        }

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """Do cropping and resizing"""
        masks = masks[:, :, :new_size[0], :new_size[1]]
        masks = torch.nn.functional.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks
