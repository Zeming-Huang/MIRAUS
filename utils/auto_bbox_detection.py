#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动边界框检测工具
用于在推理时自动生成边界框（不使用真实标签）
"""

import numpy as np
import cv2
from scipy.ndimage import label, binary_erosion


def auto_detect_bbox(image, method='intensity', padding=10):
    """
    自动检测边界框
    
    参数:
        image: (H, W) 输入图像，可以是归一化后的[0,1]或原始值
        method: 检测方法
            - 'intensity': 基于强度阈值（推荐）
            - 'contour': 基于轮廓检测
            - 'full_image': 使用整个图像
            - 'otsu': 使用Otsu阈值
        padding: 边界框的padding（像素）
    
    返回:
        bbox: (4,) [x_min, y_min, x_max, y_max]
    """
    H, W = image.shape
    
    if method == 'intensity':
        # 基于强度阈值检测
        # 使用中位数或百分位数作为阈值
        threshold = np.percentile(image, 40)  # 40%分位数
        binary = (image > threshold).astype(np.uint8)
        
        # 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 找到最大连通组件
        labeled, num_features = label(binary)
        if num_features > 0:
            # 找到最大的连通组件
            component_sizes = np.bincount(labeled.flat)[1:]  # 排除背景
            if len(component_sizes) > 0:
                largest_idx = np.argmax(component_sizes) + 1
                mask = (labeled == largest_idx).astype(np.uint8)
            else:
                mask = np.ones_like(image, dtype=np.uint8)
        else:
            mask = np.ones_like(image, dtype=np.uint8)
    
    elif method == 'otsu':
        # 使用Otsu阈值
        # 转换为8位图像
        if image.max() <= 1.0:
            img_8bit = (image * 255).astype(np.uint8)
        else:
            img_8bit = image.astype(np.uint8)
        
        _, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)
        
        # 找到最大连通组件
        labeled, num_features = label(binary)
        if num_features > 0:
            component_sizes = np.bincount(labeled.flat)[1:]
            if len(component_sizes) > 0:
                largest_idx = np.argmax(component_sizes) + 1
                mask = (labeled == largest_idx).astype(np.uint8)
            else:
                mask = np.ones_like(image, dtype=np.uint8)
        else:
            mask = np.ones_like(image, dtype=np.uint8)
    
    elif method == 'contour':
        # 使用轮廓检测
        # 转换为8位图像
        if image.max() <= 1.0:
            img_8bit = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            img_8bit = image.astype(np.uint8)
        
        # Otsu阈值化
        _, binary = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 轮廓检测
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 1)
        else:
            mask = np.ones_like(image, dtype=np.uint8)
    
    elif method == 'full_image':
        # 使用整个图像
        mask = np.ones_like(image, dtype=np.uint8)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 生成边界框
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 添加padding
        x_min = max(0, x_min - padding)
        x_max = min(W-1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(H-1, y_max + padding)
        
        return np.array([x_min, y_min, x_max, y_max])
    else:
        # 如果没找到，返回整个图像
        return np.array([0, 0, W-1, H-1])


def get_full_image_bbox(image):
    """使用整个图像作为边界框"""
    H, W = image.shape[:2]
    return np.array([0, 0, W-1, H-1])


def get_bbox_from_previous_slice(prev_seg, current_shape, padding=10):
    """
    从前一个切片的预测结果生成当前切片的边界框
    
    参数:
        prev_seg: (H, W) 前一个切片的预测分割结果
        current_shape: (H, W) 当前切片的形状
        padding: padding大小
    
    返回:
        bbox: (4,) [x_min, y_min, x_max, y_max]
    """
    # 找到前一个切片的边界框
    y_indices, x_indices = np.where(prev_seg > 0)
    
    if len(y_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 添加padding
        H, W = current_shape
        x_min = max(0, x_min - padding)
        x_max = min(W-1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(H-1, y_max + padding)
        
        return np.array([x_min, y_min, x_max, y_max])
    else:
        # 如果前一个切片没有预测，使用整个图像
        H, W = current_shape
        return np.array([0, 0, W-1, H-1])


def get_bbox_from_3d_projection(seg_3d, slice_idx, current_shape, padding=10):
    """
    从3D分割结果投影生成边界框
    
    参数:
        seg_3d: (Num, H, W) 已处理的3D分割结果
        slice_idx: 当前切片索引
        current_shape: (H, W) 当前切片的形状
        padding: padding大小
    
    返回:
        bbox: (4,) [x_min, y_min, x_max, y_max]
    """
    # 使用相邻切片的信息
    if slice_idx > 0:
        # 使用前一个切片
        prev_seg = seg_3d[slice_idx - 1]
        return get_bbox_from_previous_slice(prev_seg, current_shape, padding)
    elif slice_idx < seg_3d.shape[0] - 1:
        # 使用后一个切片（如果这是第一个切片）
        next_seg = seg_3d[slice_idx + 1]
        return get_bbox_from_previous_slice(next_seg, current_shape, padding)
    else:
        # 如果只有一张切片，使用整个图像
        H, W = current_shape
        return np.array([0, 0, W-1, H-1])






