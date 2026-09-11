#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效的双模态数据集 - 每个epoch只使用指定数量的样本
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .paired_dataset import PairedNpyDataset

class EfficientPairedNpyDataset(Dataset):
    """
    高效的双模态数据集
    每个epoch随机选择指定数量的样本，提高训练效率
    """
    def __init__(
        self,
        trus_root,
        mri_root,
        image_size=256,
        bbox_shift=5,
        data_aug=False,
        samples_per_epoch=64,
        mri_window_radius=0,
        slice_attention_mode="index_pairing",
        pairing_mode="normal",
        trus_feature_cache_dir=None,
        mri_feature_cache_dir=None,
        include_cases=None,
        exclude_cases=None,
        foreground_only=False,
        seed=2026,
        box_mode="gt",
        trus_window_radius=0,
    ):
        """
        samples_per_epoch: 如果为None，使用所有样本
        """
        """
        初始化高效数据集
        
        Args:
            trus_root: TRUS数据根目录
            mri_root: MRI数据根目录  
            image_size: 图像尺寸
            bbox_shift: 边界框扰动
            data_aug: 是否使用数据增强
            samples_per_epoch: 每个epoch的样本数量
        """
        # 如果samples_per_epoch为None，使用所有样本
        if samples_per_epoch is None:
            samples_per_epoch = float('inf')  # 表示使用所有样本
        
        self.samples_per_epoch = samples_per_epoch
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        self.mri_window_radius = int(mri_window_radius)
        self.slice_attention_mode = slice_attention_mode
        self.pairing_mode = pairing_mode
        self.trus_feature_cache_dir = trus_feature_cache_dir
        self.mri_feature_cache_dir = mri_feature_cache_dir
        self.include_cases = include_cases
        self.exclude_cases = exclude_cases
        self.foreground_only = bool(foreground_only)
        self.seed = int(seed)
        self.epoch = 0
        self.box_mode = str(box_mode)
        self.trus_window_radius = int(trus_window_radius)
        
        # 使用原始数据集获取所有文件列表
        self.full_dataset = PairedNpyDataset(
            trus_root,
            mri_root,
            image_size,
            bbox_shift,
            data_aug,
            mri_window_radius=self.mri_window_radius,
            slice_attention_mode=self.slice_attention_mode,
            pairing_mode=self.pairing_mode,
            trus_feature_cache_dir=self.trus_feature_cache_dir,
            mri_feature_cache_dir=self.mri_feature_cache_dir,
            include_cases=self.include_cases,
            exclude_cases=self.exclude_cases,
            foreground_only=self.foreground_only,
            seed=self.seed,
            box_mode=self.box_mode,
            trus_window_radius=self.trus_window_radius,
        )
        self.total_samples = len(self.full_dataset)
        
        # 确定实际使用的样本数
        if samples_per_epoch == float('inf') or samples_per_epoch >= self.total_samples:
            actual_samples = self.total_samples
            utilization = 100.0
        else:
            actual_samples = samples_per_epoch
            utilization = samples_per_epoch/self.total_samples*100.0
        
        print(f"EfficientPairedNpyDataset初始化:")
        print(f"  - 总样本数: {self.total_samples}")
        print(f"  - 每epoch样本数: {actual_samples} ({'全部' if actual_samples == self.total_samples else f'{utilization:.1f}%'})")
        
        # 为当前epoch生成随机索引
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.full_dataset.set_epoch(self.epoch)
        self.current_epoch_indices = self._generate_epoch_indices()
        
    def _generate_epoch_indices(self):
        rng = random.Random(self.seed + 1000003 * self.epoch)
        """为当前epoch生成随机索引"""
        if self.samples_per_epoch == float('inf') or self.samples_per_epoch >= self.total_samples:
            # 如果需要的样本数大于等于总样本数，使用所有样本并打乱
            indices = list(range(self.total_samples))
            rng.shuffle(indices)
        else:
            # 随机选择指定数量的样本（不排序，保持随机顺序）
            indices = rng.sample(range(self.total_samples), self.samples_per_epoch)
        
        return indices  # 不排序，保持随机顺序以提高数据多样性
    
    def __len__(self):
        """返回当前epoch的样本数量"""
        return len(self.current_epoch_indices)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 使用当前epoch的索引映射
        actual_idx = self.current_epoch_indices[idx]
        return self.full_dataset[actual_idx]
    
    def new_epoch(self):
        """开始新的epoch，重新生成随机索引"""
        self.current_epoch_indices = self._generate_epoch_indices()
        print(f"新epoch开始: 使用{len(self.current_epoch_indices)}个样本")
    
    def get_epoch_info(self):
        """获取当前epoch信息"""
        return {
            'total_samples': self.total_samples,
            'current_epoch_samples': len(self.current_epoch_indices),
            'utilization_rate': len(self.current_epoch_indices) / self.total_samples * 100,
            'indices': self.current_epoch_indices[:10] if len(self.current_epoch_indices) > 10 else self.current_epoch_indices  # 只显示前10个索引
        }
