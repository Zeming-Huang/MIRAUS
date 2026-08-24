# -*- coding: utf-8 -*-
"""
配对MRI-TRUS数据加载器
用于双模态训练,确保MRI和TRUS的切片能够一一对应加载
"""

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
from os.path import join, basename, isfile, normpath

class PairedNpyDataset(Dataset):
    """
    配对MRI-TRUS数据集
    确保MRI和TRUS的切片能够同步加载,用于跨模态训练
    """
    
    @staticmethod
    def _parse_case_slice(path):
        name = basename(str(path))
        case_match = re.search(r"(case\d+)", name)
        case_id = case_match.group(1) if case_match else os.path.splitext(name)[0]
        slice_match = re.search(r"case\d+[-_](\d+)", name)
        slice_index = int(slice_match.group(1)) if slice_match else 0
        return case_id, slice_index

    def _build_trus_rank_index(self):
        grouped = {}
        for trus_gt_file in self.trus_gt_files:
            case_id, slice_index = self._parse_case_slice(trus_gt_file)
            grouped.setdefault(case_id, []).append((slice_index, normpath(trus_gt_file)))

        self.trus_rank_by_file = {}
        self.trus_count_by_case = {}
        self.trus_entries_by_case = {}
        for case_id, entries in grouped.items():
            entries = sorted(entries, key=lambda x: (x[0], x[1]))
            self.trus_count_by_case[case_id] = len(entries)
            self.trus_entries_by_case[case_id] = [file_path for _, file_path in entries]
            for rank, (_, file_path) in enumerate(entries):
                self.trus_rank_by_file[normpath(file_path)] = rank

    def _build_trus_transition_targets(self):
        self.trus_mask_area_by_file = {}
        self.transition_target_by_file = {}
        self.transition_label_by_file = {}
        self.transition_valid_by_file = {}

        for case_id, file_paths in self.trus_entries_by_case.items():
            areas = []
            for file_path in file_paths:
                gt = np.load(file_path, 'r', allow_pickle=True)
                area = float(np.uint8(gt > 0).sum())
                self.trus_mask_area_by_file[file_path] = area
                areas.append(area)

            raw_jumps = []
            for idx, file_path in enumerate(file_paths):
                area = areas[idx]
                deltas = []
                if idx > 0:
                    deltas.append(abs(area - areas[idx - 1]))
                if idx + 1 < len(areas):
                    deltas.append(abs(areas[idx + 1] - area))

                if not deltas:
                    raw_jumps.append(0.0)
                    self.transition_target_by_file[file_path] = 0.0
                    self.transition_label_by_file[file_path] = 0
                    self.transition_valid_by_file[file_path] = False
                    continue

                raw_jump = max(deltas) / max(area, 1.0)
                raw_jumps.append(float(raw_jump))
                self.transition_valid_by_file[file_path] = True

            raw_jumps = np.asarray(raw_jumps, dtype=np.float32)
            valid_mask = np.asarray([
                self.transition_valid_by_file.get(file_path, False)
                for file_path in file_paths
            ], dtype=bool)
            if valid_mask.any():
                valid_jumps = raw_jumps[valid_mask]
                p20 = float(np.percentile(valid_jumps, 20))
                p80 = float(np.percentile(valid_jumps, 80))
                p90 = float(np.percentile(valid_jumps, 90))
            else:
                p20 = p80 = p90 = 0.0

            denom = p90 - p20 + 1e-6
            for idx, file_path in enumerate(file_paths):
                if not self.transition_valid_by_file.get(file_path, False):
                    self.transition_target_by_file[file_path] = 0.0
                    self.transition_label_by_file[file_path] = 0
                    continue
                jump_norm = np.clip((raw_jumps[idx] - p20) / denom, 0.0, 1.0)
                self.transition_target_by_file[file_path] = float(jump_norm)
                self.transition_label_by_file[file_path] = int(raw_jumps[idx] >= p80)

    def _build_mri_slice_index(self):
        self.mri_slices_by_case = {}
        for mri_img_file in sorted(glob(join(self.mri_img_path, "*.npy"))):
            case_id, slice_index = self._parse_case_slice(mri_img_file)
            mri_gt_file = normpath(join(self.mri_gt_path, basename(mri_img_file)))
            if not isfile(mri_gt_file):
                continue
            self.mri_slices_by_case.setdefault(case_id, []).append({
                "img": normpath(mri_img_file),
                "gt": mri_gt_file,
                "case_id": case_id,
                "slice_index": slice_index,
            })

        for case_id, entries in self.mri_slices_by_case.items():
            self.mri_slices_by_case[case_id] = sorted(
                entries,
                key=lambda item: (item["slice_index"], item["img"]),
            )

    def _normalized_mri_center_rank(self, case_id, trus_gt_file):
        mri_entries = self.mri_slices_by_case.get(case_id, [])
        if not mri_entries:
            return None

        trus_rank = self.trus_rank_by_file.get(normpath(trus_gt_file), 0)
        trus_count = self.trus_count_by_case.get(case_id, 1)
        mri_count = len(mri_entries)
        if trus_count <= 1 or mri_count <= 1:
            return 0

        normalized_depth = trus_rank / float(trus_count - 1)
        return int(round(normalized_depth * (mri_count - 1)))

    def _mri_window_entries(self, pair):
        center_rank, selected_case_id = self._pairing_center_rank(pair)
        mri_entries = self.mri_slices_by_case.get(selected_case_id, [])
        if not mri_entries:
            return ([{
                "img": pair["mri_img"],
                "gt": pair["mri_gt"],
                "case_id": pair["case_id"],
                "slice_index": pair["mri_center_index"],
            }], [True])

        radius = self.mri_window_radius if self.return_mri_window else 0
        window_entries = []
        valid_mask = []
        for offset in range(-radius, radius + 1):
            raw_rank = center_rank + offset
            valid = 0 <= raw_rank < len(mri_entries)
            rank = min(max(raw_rank, 0), len(mri_entries) - 1)
            window_entries.append(mri_entries[rank])
            valid_mask.append(bool(valid))
        return window_entries, valid_mask

    def _pairing_center_rank(self, pair):
        case_id = pair["case_id"]
        center_rank = int(pair.get("mri_center_rank", 0))
        mode = self.pairing_mode

        if mode == "normal":
            selected_case_id = case_id
        elif mode.startswith("shift_plus_"):
            selected_case_id = case_id
            center_rank += int(mode.rsplit("_", 1)[-1])
        elif mode.startswith("shift_minus_"):
            selected_case_id = case_id
            center_rank -= int(mode.rsplit("_", 1)[-1])
        elif mode == "same_patient_random":
            selected_case_id = case_id
            entries = self.mri_slices_by_case.get(selected_case_id, [])
            center_rank = random.randint(0, max(len(entries) - 1, 0))
        elif mode == "different_patient_random":
            candidate_cases = [c for c in self.mri_slices_by_case.keys() if c != case_id]
            if not candidate_cases:
                raise ValueError("different_patient_random requires at least two patients.")
            selected_case_id = random.choice(candidate_cases)
            entries = self.mri_slices_by_case.get(selected_case_id, [])
            center_rank = random.randint(0, max(len(entries) - 1, 0))
        else:
            raise ValueError(f"Unknown pairing_mode: {mode}")

        entries = self.mri_slices_by_case.get(selected_case_id, [])
        if entries:
            center_rank = min(max(center_rank, 0), len(entries) - 1)
        else:
            center_rank = 0
        return center_rank, selected_case_id

    def __init__(
        self,
        trus_root,
        mri_root,
        image_size=256,
        bbox_shift=5,
        data_aug=False,
        mri_window_radius=0,
        slice_attention_mode="index_pairing",
        pairing_mode="normal",
        trus_feature_cache_dir=None,
        mri_feature_cache_dir=None,
        box_mode="gt",
    ):
        """
        Args:
            trus_root: TRUS数据根目录 (已经是具体的channel路径，如.../trus/channel_0)
            mri_root: MRI数据根目录 (已经是具体的channel路径，如.../mri/channel_0)
            image_size: 图像尺寸
            bbox_shift: bounding box扰动范围
            data_aug: 是否使用数据增强
        """
        # 规范化路径（处理Windows路径混合问题）
        # 如果是相对路径，先转换为绝对路径再规范化
        if not os.path.isabs(trus_root):
            trus_root = os.path.abspath(trus_root)
        if not os.path.isabs(mri_root):
            mri_root = os.path.abspath(mri_root)
        self.trus_root = normpath(trus_root)
        self.mri_root = normpath(mri_root)
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        if box_mode not in {"gt", "full_image"}:
            raise ValueError(f"Unknown box_mode: {box_mode}")
        self.box_mode = box_mode
        self.data_aug = data_aug
        self.mri_window_radius = int(mri_window_radius)
        self.slice_attention_mode = slice_attention_mode
        self.pairing_mode = pairing_mode
        self.return_mri_window = self.mri_window_radius > 0 or slice_attention_mode != "index_pairing"
        self.trus_feature_cache_dir = normpath(trus_feature_cache_dir) if trus_feature_cache_dir else None
        self.mri_feature_cache_dir = normpath(mri_feature_cache_dir) if mri_feature_cache_dir else None
        self.use_feature_cache = bool(self.trus_feature_cache_dir and self.mri_feature_cache_dir)
        if self.use_feature_cache and self.data_aug:
            raise ValueError("Feature cache cannot be used with data augmentation.")
        
        # 直接使用传入的路径，不重复添加通道
        trus_gt_path = normpath(join(self.trus_root, 'gts'))
        trus_img_path = normpath(join(self.trus_root, 'imgs'))
        
        if not os.path.exists(trus_gt_path) or not os.path.exists(trus_img_path):
            raise ValueError(f"TRUS数据路径不存在: {trus_gt_path} 或 {trus_img_path}")
            
        mri_gt_path = normpath(join(self.mri_root, 'gts'))
        mri_img_path = normpath(join(self.mri_root, 'imgs'))
        if not os.path.exists(mri_gt_path) or not os.path.exists(mri_img_path):
            raise ValueError(f"MRI数据路径不存在: {mri_gt_path} 或 {mri_img_path}")
        
        # 获取所有TRUS标签文件
        self.trus_gt_files = sorted(glob(join(trus_gt_path, '*.npy'), recursive=True))
        # 使用规范化后的路径变量
        self.trus_gt_path = trus_gt_path
        self.trus_img_path = trus_img_path
        self.mri_gt_path = mri_gt_path
        self.mri_img_path = mri_img_path
        self.trus_gt_files = [
            file for file in self.trus_gt_files
            if isfile(normpath(join(self.trus_img_path, basename(file))))
        ]
        self._build_trus_rank_index()
        self._build_trus_transition_targets()
        self._build_mri_slice_index()
        
        # 验证配对关系
        self.valid_pairs = []
        for trus_gt_file in self.trus_gt_files:
            trus_img_file = normpath(join(self.trus_img_path, basename(trus_gt_file)))
            
            # 通过去除前缀来匹配MRI文件
            # TRUS_Prostate_case000000-000.npy -> case000000-000.npy
            trus_basename = basename(trus_gt_file)
            if trus_basename.startswith('TRUS_Prostate_'):
                mri_basename = trus_basename.replace('TRUS_Prostate_', 'MRI_Prostate_')
            else:
                # 如果格式不同,尝试其他匹配方式
                mri_basename = trus_basename
            
            mri_gt_file = normpath(join(self.mri_gt_path, mri_basename))
            mri_img_file = normpath(join(self.mri_img_path, mri_basename))
            case_id, trus_slice_index = self._parse_case_slice(trus_gt_file)
            mri_center_rank = self._normalized_mri_center_rank(case_id, trus_gt_file)
            mapped_mri_entry = None
            if mri_center_rank is not None:
                mapped_mri_entry = self.mri_slices_by_case[case_id][mri_center_rank]
            
            # 检查所有对应文件是否存在（使用规范化路径）
            if (isfile(normpath(trus_img_file)) and isfile(normpath(mri_gt_file)) and isfile(normpath(mri_img_file))):
                _, mri_slice_index = self._parse_case_slice(mri_img_file)
                self.valid_pairs.append({
                    'trus_img': trus_img_file,
                    'trus_gt': trus_gt_file,
                    'mri_img': mri_img_file,
                    'mri_gt': mri_gt_file,
                    'case_id': case_id,
                    'trus_slice_index': trus_slice_index,
                    'mri_center_rank': mri_center_rank if mri_center_rank is not None else 0,
                    'mri_center_index': (
                        mapped_mri_entry["slice_index"]
                        if mapped_mri_entry is not None
                        else mri_slice_index
                    ),
                })
            elif isfile(normpath(trus_img_file)) and mapped_mri_entry is not None:
                self.valid_pairs.append({
                    'trus_img': trus_img_file,
                    'trus_gt': trus_gt_file,
                    'mri_img': mapped_mri_entry["img"],
                    'mri_gt': mapped_mri_entry["gt"],
                    'case_id': case_id,
                    'trus_slice_index': trus_slice_index,
                    'mri_center_rank': mri_center_rank,
                    'mri_center_index': mapped_mri_entry["slice_index"],
                })
            else:
                print(f"警告: 找不到配对文件 {trus_basename} -> {mri_basename}")
        
        print(f"找到 {len(self.valid_pairs)} 个有效的MRI-TRUS配对")
        if len(self.valid_pairs) == 0:
            raise ValueError("没有找到有效的配对数据!")
    
    def __len__(self):
        return len(self.valid_pairs)

    @staticmethod
    def _feature_cache_path(cache_dir, image_path):
        stem = os.path.splitext(basename(image_path))[0]
        return normpath(join(cache_dir, stem + ".pt"))

    def _load_cached_feature(self, cache_dir, image_path):
        feature_path = self._feature_cache_path(cache_dir, image_path)
        if not isfile(feature_path):
            raise FileNotFoundError(f"Missing cached feature: {feature_path}")
        feature = torch.load(feature_path, map_location="cpu")
        if isinstance(feature, dict):
            feature = feature.get("feature", feature)
        return feature.float()
    
    def __getitem__(self, index):
        pair = self.valid_pairs[index]
        num_slices = int(self.trus_count_by_case.get(pair["case_id"], 1))
        trus_rank = int(self.trus_rank_by_file.get(normpath(pair["trus_gt"]), 0))
        relative_depth = trus_rank / float(max(num_slices - 1, 1))
        
        # 加载TRUS数据
        trus_img_3c = np.load(pair['trus_img'], 'r', allow_pickle=True)  # (H, W, 3)
        trus_gt = np.load(pair['trus_gt'], 'r', allow_pickle=True)  # (H, W)
        
        # 加载MRI数据
        mri_window_entries, mri_valid_mask = self._mri_window_entries(pair)
        if self.use_feature_cache:
            trus_gt = np.uint8(trus_gt > 0)
            trus_img_tensor = self._load_cached_feature(self.trus_feature_cache_dir, pair["trus_img"])
            mri_img_window = [
                self._load_cached_feature(self.mri_feature_cache_dir, entry["img"])
                for entry in mri_window_entries
            ]
            mri_img_tensor = mri_img_window[len(mri_img_window) // 2]
            if self.return_mri_window:
                mri_img_tensor = torch.stack(mri_img_window, dim=0)

            trus_gt_tensor = torch.tensor(trus_gt[None, :, :]).long()
            y_indices, x_indices = np.where(trus_gt > 0)
            H, W = trus_gt.shape
            if self.box_mode == "full_image":
                bboxes = np.array([0, 0, W - 1, H - 1])
            elif len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
                bboxes = np.array([x_min, y_min, x_max, y_max])
            else:
                bboxes = np.array([0, 0, W-1, H-1])

            return {
                "trus_image": trus_img_tensor,
                "mri_image": mri_img_tensor,
                "gt2D": trus_gt_tensor,
                "bboxes": torch.tensor(bboxes[None, None, ...]).float(),
                "image_name": basename(pair['trus_gt']),
                "case_id": pair["case_id"],
                "trus_slice_index": torch.tensor(pair["trus_slice_index"]).long(),
                "slice_index": torch.tensor(trus_rank).long(),
                "num_slices": torch.tensor(num_slices).long(),
                "relative_depth": torch.tensor(relative_depth).float(),
                "mri_center_index": torch.tensor(mri_window_entries[len(mri_window_entries) // 2]["slice_index"]).long(),
                "mri_pairing_case_id": mri_window_entries[len(mri_window_entries) // 2].get("case_id", pair["case_id"]),
                "mri_window_indices": torch.tensor([
                    entry["slice_index"] for entry in mri_window_entries
                ]).long(),
                "mri_valid_mask": torch.tensor(mri_valid_mask).bool(),
                "transition_target": torch.tensor(
                    self.transition_target_by_file.get(normpath(pair["trus_gt"]), 0.0)
                ).float(),
                "transition_label": torch.tensor(
                    self.transition_label_by_file.get(normpath(pair["trus_gt"]), 0)
                ).float(),
                "transition_valid": torch.tensor(
                    self.transition_valid_by_file.get(normpath(pair["trus_gt"]), False)
                ).bool(),
                "mask_area": torch.tensor(
                    self.trus_mask_area_by_file.get(normpath(pair["trus_gt"]), float(trus_gt.sum()))
                ).float(),
                "new_size": torch.tensor(np.array([self.image_size, self.image_size])).long(),
                "original_size": torch.tensor(np.array([self.image_size, self.image_size])).long()
            }
        mri_img_window = [
            np.load(entry["img"], 'r', allow_pickle=True)
            for entry in mri_window_entries
        ]
        mri_img_3c = mri_img_window[len(mri_img_window) // 2]  # (H, W, 3)
        mri_gt = np.load(mri_window_entries[len(mri_window_entries) // 2]["gt"], 'r', allow_pickle=True)  # (H, W)
        
        # 验证数据形状
        assert trus_img_3c.shape[:2] == trus_gt.shape, f"TRUS图像和标签形状不匹配: {trus_img_3c.shape} vs {trus_gt.shape}"
        assert mri_img_3c.shape[:2] == mri_gt.shape, f"MRI图像和标签形状不匹配: {mri_img_3c.shape} vs {mri_gt.shape}"
        
        # 确保标签是二值的
        trus_gt = np.uint8(trus_gt > 0)
        mri_gt = np.uint8(mri_gt > 0)
        
        # 数据增强 - 同时对MRI和TRUS应用相同的变换 (关闭数据增强)
        if self.data_aug:
            # 随机水平翻转
            if random.random() > 0.5:
                trus_img_3c = np.ascontiguousarray(np.flip(trus_img_3c, axis=1))
                trus_gt = np.ascontiguousarray(np.flip(trus_gt, axis=1))
                mri_img_3c = np.ascontiguousarray(np.flip(mri_img_3c, axis=1))
                mri_img_window = [
                    np.ascontiguousarray(np.flip(mri_img, axis=1))
                    for mri_img in mri_img_window
                ]
                mri_gt = np.ascontiguousarray(np.flip(mri_gt, axis=1))
            
            # 随机垂直翻转
            if random.random() > 0.5:
                trus_img_3c = np.ascontiguousarray(np.flip(trus_img_3c, axis=0))
                trus_gt = np.ascontiguousarray(np.flip(trus_gt, axis=0))
                mri_img_3c = np.ascontiguousarray(np.flip(mri_img_3c, axis=0))
                mri_img_window = [
                    np.ascontiguousarray(np.flip(mri_img, axis=0))
                    for mri_img in mri_img_window
                ]
                mri_gt = np.ascontiguousarray(np.flip(mri_gt, axis=0))
        
        # 转换为tensor格式 (C, H, W) - 与原始NpyDataset保持一致
        trus_img_tensor = torch.tensor(trus_img_3c).permute(2, 0, 1).float()
        mri_img_tensor = torch.tensor(mri_img_3c).permute(2, 0, 1).float()
        if self.return_mri_window:
            mri_img_tensor = torch.stack([
                torch.tensor(mri_img).permute(2, 0, 1).float()
                for mri_img in mri_img_window
            ], dim=0)
        
        # 验证图像归一化 - 与原始NpyDataset保持一致
        assert torch.max(trus_img_tensor) <= 1.0 and torch.min(trus_img_tensor) >= 0.0, \
            f"TRUS图像未正确归一化: [{torch.min(trus_img_tensor):.3f}, {torch.max(trus_img_tensor):.3f}]"
        assert torch.max(mri_img_tensor) <= 1.0 and torch.min(mri_img_tensor) >= 0.0, \
            f"MRI图像未正确归一化: [{torch.min(mri_img_tensor):.3f}, {torch.max(mri_img_tensor):.3f}]"
        
        # 生成bounding box (MedSAM需要的提示信息)
        trus_gt_tensor = torch.tensor(trus_gt[None, :, :]).long()  # (1, H, W)
        y_indices, x_indices = np.where(trus_gt > 0)
        
        H, W = trus_gt.shape
        if self.box_mode == "full_image":
            bboxes = np.array([0, 0, W - 1, H - 1])
        elif len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # 添加扰动 (训练时增加鲁棒性)
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W-1, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H-1, y_max + random.randint(0, self.bbox_shift))
            
            bboxes = np.array([x_min, y_min, x_max, y_max])
        else:
            # 如果没有前景,使用整个图像作为bbox
            bboxes = np.array([0, 0, W-1, H-1])
        
        bboxes_tensor = torch.tensor(bboxes[None, None, ...]).float()  # (1, 1, 4)
        
        return {
            "trus_image": trus_img_tensor,
            "mri_image": mri_img_tensor,
            "gt2D": trus_gt_tensor,  # 使用TRUS的标签作为ground truth
            "bboxes": bboxes_tensor,
            "image_name": basename(pair['trus_gt']),
            "case_id": pair["case_id"],
            "trus_slice_index": torch.tensor(pair["trus_slice_index"]).long(),
            "slice_index": torch.tensor(trus_rank).long(),
            "num_slices": torch.tensor(num_slices).long(),
            "relative_depth": torch.tensor(relative_depth).float(),
            "mri_center_index": torch.tensor(mri_window_entries[len(mri_window_entries) // 2]["slice_index"]).long(),
            "mri_pairing_case_id": mri_window_entries[len(mri_window_entries) // 2].get("case_id", pair["case_id"]),
            "mri_window_indices": torch.tensor([
                entry["slice_index"] for entry in mri_window_entries
            ]).long(),
            "mri_valid_mask": torch.tensor(mri_valid_mask).bool(),
            "transition_target": torch.tensor(
                self.transition_target_by_file.get(normpath(pair["trus_gt"]), 0.0)
            ).float(),
            "transition_label": torch.tensor(
                self.transition_label_by_file.get(normpath(pair["trus_gt"]), 0)
            ).float(),
            "transition_valid": torch.tensor(
                self.transition_valid_by_file.get(normpath(pair["trus_gt"]), False)
            ).bool(),
            "mask_area": torch.tensor(
                self.trus_mask_area_by_file.get(normpath(pair["trus_gt"]), float(trus_gt.sum()))
            ).float(),
            "new_size": torch.tensor(np.array([self.image_size, self.image_size])).long(),
            "original_size": torch.tensor(np.array([self.image_size, self.image_size])).long()
        }
    
    def get_sample_info(self, index):
        """获取样本信息,用于调试"""
        pair = self.valid_pairs[index]
        return {
            'trus_img': pair['trus_img'],
            'mri_img': pair['mri_img'],
            'trus_gt': pair['trus_gt'],
            'mri_gt': pair['mri_gt']
        }

