import torch
import torch.nn as nn
from segment_anything.modeling import MaskDecoder
from tiny_vit_sam import CrossModalFeatureExtractor
from copy import deepcopy

class EnhancedMaskDecoder(MaskDecoder):
    """SAM mask decoder with the interface used by the MIRAUS model."""
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
        # Preserve the standard SAM decoding path.
        return super().forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

class EnhancedDualModalMedSAM_Lite(nn.Module):
    """MRI-privileged teacher and deployable TRUS-only student."""
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
        candidate_seg_loss_weight=1.0,
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
        use_gated_student_adapter=False,
        student_gate_init_bias=-1.5,
        use_student_refinement_adapter=False,
        use_spatial_fusion_gate=False,
        spatial_gate_conf_floor=0.5,
        use_foreground_mmd=False,
        foreground_mmd_min_tokens=8,
        trus_context_mix=0.0,
    ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        self.use_src_enhancement = use_src_enhancement
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_fusion = use_fusion
        self.slice_attention_mode = slice_attention_mode
        # The teacher uses TRUS and MRI; the deployable student uses TRUS only.
        self.use_privileged_distillation = bool(use_privileged_distillation)
        self.use_gated_student_adapter = bool(use_gated_student_adapter)
        self.use_student_refinement_adapter = bool(use_student_refinement_adapter)
        
        self.trus_encoder = image_encoder
        self.mri_encoder = deepcopy(image_encoder) if use_cross_modal else None
        
        # Cross-modal feature extraction and utility-aware aggregation.
        if use_cross_modal:
            self.cross_modal_extractor = CrossModalFeatureExtractor(
                in_channels=256,
                num_heads=8,
                mmd_weight=0.1,
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
                candidate_seg_loss_weight=candidate_seg_loss_weight,
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
                use_spatial_fusion_gate=use_spatial_fusion_gate,
                spatial_gate_conf_floor=spatial_gate_conf_floor,
                use_foreground_mmd=use_foreground_mmd,
                foreground_mmd_min_tokens=foreground_mmd_min_tokens,
            )
            
        # Shared SAM prompt encoder and mask decoder.
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # The student predicts an MRI-conditioned residual from TRUS features.
        self.student_adapter = nn.Sequential(
            nn.Conv2d(256, student_adapter_hidden, 3, padding=1),
            nn.GroupNorm(32, student_adapter_hidden),
            nn.GELU(),
            nn.Conv2d(student_adapter_hidden, 256, 3, padding=1),
        )
        if self.use_student_refinement_adapter:
            self.student_refinement_adapter = nn.Sequential(
                nn.Conv2d(256, student_adapter_hidden, 3, padding=1),
                nn.GroupNorm(32, student_adapter_hidden),
                nn.GELU(),
                nn.Conv2d(student_adapter_hidden, 256, 3, padding=1),
            )
            nn.init.zeros_(self.student_refinement_adapter[-1].weight)
            nn.init.zeros_(self.student_refinement_adapter[-1].bias)
        else:
            self.student_refinement_adapter = None
        if self.use_gated_student_adapter:
            gate_hidden = max(32, student_adapter_hidden // 4)
            self.student_gate = nn.Sequential(
                nn.Conv2d(256, gate_hidden, 1),
                nn.GELU(),
                nn.Conv2d(gate_hidden, 1, 1),
            )
            nn.init.zeros_(self.student_gate[-1].weight)
            nn.init.constant_(self.student_gate[-1].bias, float(student_gate_init_bias))
        else:
            self.student_gate = None

    def _student_feature(self, trus_feat, return_details=False):
        base_residual = self.student_adapter(trus_feat)
        raw_residual = (
            self.student_refinement_adapter(trus_feat)
            if self.student_refinement_adapter is not None
            else base_residual
        )
        if self.student_gate is None:
            gate = torch.ones(
                (trus_feat.shape[0], 1, *trus_feat.shape[-2:]),
                device=trus_feat.device,
                dtype=trus_feat.dtype,
            )
        else:
            gate = torch.sigmoid(self.student_gate(trus_feat))
        gated_residual = gate * raw_residual
        predicted_residual = (
            base_residual + gated_residual
            if self.student_refinement_adapter is not None
            else gated_residual
        )
        student_feat = trus_feat + predicted_residual
        if return_details:
            return student_feat, raw_residual, gate, predicted_residual
        return student_feat

    @staticmethod
    def _is_encoded_feature(tensor):
        return (
            tensor is not None
            and tensor.dim() == 4
            and tensor.shape[1:] == (256, 64, 64)
        )

    @staticmethod
    def _is_encoded_feature_set(tensor):
        return (
            tensor is not None
            and tensor.dim() == 5
            and tensor.shape[2:] == (256, 64, 64)
        )

    def _encode_trus(self, trus_image):
        if self._is_encoded_feature(trus_image):
            return trus_image
        return self.trus_encoder(trus_image)

    def _encode_mri(self, mri_image):
        if self._is_encoded_feature_set(mri_image) or self._is_encoded_feature(mri_image):
            return mri_image
        if mri_image.dim() == 5:
            batch, slices, channels, height, width = mri_image.shape
            features = self.mri_encoder(
                mri_image.reshape(batch * slices, channels, height, width)
            )
            return features.reshape(batch, slices, *features.shape[1:])
        return self.mri_encoder(mri_image)

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

    def forward_student(self, trus_image, boxes=None, **_ignored):
        trus_feat = self._encode_trus(trus_image)
        student_feat, raw_residual, gate, residual = self._student_feature(
            trus_feat, return_details=True
        )
        logits, iou = self._decode(student_feat, boxes)
        return {
            "logits_student": logits,
            "iou_student": iou,
            "feat_student": student_feat,
            "feat_trus": trus_feat,
            "student_raw_residual": raw_residual,
            "student_residual": residual,
            "student_gate": gate,
        }

    def forward_teacher(
        self,
        trus_image,
        mri_image,
        boxes=None,
        mask_gt=None,
        mri_valid_mask=None,
        transition_target=None,
        transition_label=None,
        transition_valid=None,
        relative_depth=None,
        **_ignored,
    ):
        if mri_image is None:
            raise ValueError("forward_teacher requires MRI input")
        trus_feat = self._encode_trus(trus_image)
        mri_feat = self._encode_mri(mri_image)
        cross_modal = self.cross_modal_extractor(
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
        if isinstance(cross_modal, (tuple, list)) and len(cross_modal) == 3:
            teacher_feat, mmd_loss, aux_extra = cross_modal
        else:
            teacher_feat, mmd_loss = cross_modal
            aux_extra = trus_feat.new_zeros(())
        logits, iou = self._decode(teacher_feat, boxes)
        return {
            "logits_teacher": logits,
            "iou_teacher": iou,
            "feat_teacher": teacher_feat,
            "feat_trus": trus_feat,
            "mmd_loss": mmd_loss,
            "aux_extra": aux_extra,
        }

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
        **_ignored,
    ):
        """Run the teacher/student training path or the TRUS-only student path."""
        # 1. 鑾峰彇TRUS鐗瑰緛
        if training and self.use_cross_modal and mri_image is not None and self.use_privileged_distillation:
            teacher_output = self.forward_teacher(
                trus_image,
                mri_image,
                boxes=boxes,
                mask_gt=mask_gt,
                mri_valid_mask=mri_valid_mask,
                transition_target=transition_target,
                transition_label=transition_label,
                transition_valid=transition_valid,
                relative_depth=relative_depth,
            )
            student_output = self.forward_student(trus_image, boxes=boxes)
            return {**teacher_output, **student_output}

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

        # Distillation checkpoints infer through the TRUS student. Legacy
        # ablations instead reuse the trained cross-modal block as self-attention.
        if not (training and self.use_cross_modal and mri_image is not None):
            if self.use_privileged_distillation:
                inference_feat = self._student_feature(trus_feat)
            elif self.use_cross_modal:
                cm_out = self.cross_modal_extractor(trus_feat, trus_feat)
                inference_feat = cm_out[0] if isinstance(cm_out, tuple) else cm_out
            else:
                inference_feat = trus_feat
            low_res_masks, iou_predictions = self._decode(inference_feat, boxes)
            return low_res_masks, iou_predictions

        # Joint compatibility path: MRI-privileged teacher and TRUS-only student.
        # Teacher 缂栫爜 MRI
        if _is_encoded_feature_set(mri_image) or _is_encoded_feature(mri_image):
            mri_feat = mri_image
        elif mri_image.dim() == 5:
            B, S, C, H, W = mri_image.shape
            mri_feat = self.mri_encoder(mri_image.reshape(B * S, C, H, W))
            _, feat_C, feat_H, feat_W = mri_feat.shape
            mri_feat = mri_feat.reshape(B, S, feat_C, feat_H, feat_W)
        else:
            mri_feat = self.mri_encoder(mri_image)  # (B, 256, 64, 64)

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

        # Pre-distillation ablations supervise only the MRI-guided path.
        if not self.use_privileged_distillation:
            logits_t, iou_t = self._decode(f_teacher, boxes)
            return logits_t, iou_t, mmd_loss

        # Student representation from TRUS alone.
        f_student, student_raw_residual, student_gate, student_residual = self._student_feature(
            trus_feat, return_details=True
        )

        # 涓よ矾鍏变韩鍚屼竴 decoder/prompt_encoder
        logits_t, iou_t = self._decode(f_teacher, boxes)
        logits_s, iou_s = self._decode(f_student, boxes)

        # Return both paths and the features required for privileged transfer.
        return {
            "logits_student": logits_s,
            "iou_student": iou_s,
            "logits_teacher": logits_t,
            "iou_teacher": iou_t,
            "feat_student": f_student,
            "feat_teacher": f_teacher,
            "feat_trus": trus_feat,
            "student_residual": student_residual,
            "student_gate": student_gate,
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
