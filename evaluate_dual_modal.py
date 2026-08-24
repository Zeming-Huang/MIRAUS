#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模态MedSAM评估脚本
计算Dice分数、Hausdorff距离等指标
"""

import numpy as np
import os
from os.path import join, basename
from glob import glob
import argparse
from tqdm import tqdm
import pandas as pd

def compute_dice_score(pred, gt):
    """计算Dice分数"""
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union

def compute_hausdorff_distance(pred, gt):
    """计算Hausdorff距离"""
    from scipy.spatial.distance import directed_hausdorff
    
    # 获取边界点
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    # 计算双向Hausdorff距离
    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(d1, d2)

def evaluate_predictions(pred_dir, gt_dir):
    """评估预测结果"""
    results = []
    
    # 获取所有预测文件
    pred_files = sorted(glob(join(pred_dir, '**', '*.npz'), recursive=True))
    
    print(f"找到 {len(pred_files)} 个预测文件")
    
    for pred_file in tqdm(pred_files, desc="评估中"):
        # 加载预测结果
        pred_data = np.load(pred_file)
        pred_segs = pred_data['segs']  # (Num, H, W)
        pred_gts = pred_data['gts']    # (Num, H, W)
        
        # 找到对应的真实标签文件
        file_name = basename(pred_file)
        gt_file = join(gt_dir, file_name)
        
        if not os.path.exists(gt_file):
            print(f"警告: 找不到对应的真实标签文件 {gt_file}")
            continue
            
        # 加载真实标签
        gt_data = np.load(gt_file)
        gt_segs = gt_data['gts']  # (Num, H, W)
        
        # 确保形状一致
        if pred_segs.shape != gt_segs.shape:
            print(f"警告: 形状不匹配 {pred_segs.shape} vs {gt_segs.shape}")
            continue
        
        # 计算每个切片的指标
        case_dices = []
        case_hausdorffs = []
        
        for i in range(pred_segs.shape[0]):
            pred_slice = pred_segs[i]
            gt_slice = gt_segs[i]
            
            # 二值化
            pred_binary = (pred_slice > 0).astype(np.uint8)
            gt_binary = (gt_slice > 0).astype(np.uint8)
            
            # 计算Dice分数
            dice = compute_dice_score(pred_binary, gt_binary)
            case_dices.append(dice)
            
            # 计算Hausdorff距离
            hausdorff = compute_hausdorff_distance(pred_binary, gt_binary)
            case_hausdorffs.append(hausdorff)
        
        # 计算病例级别的指标
        case_dice = np.mean(case_dices)
        case_hausdorff = np.mean([h for h in case_hausdorffs if h != float('inf')])
        
        results.append({
            'case': file_name,
            'dice': case_dice,
            'hausdorff': case_hausdorff,
            'num_slices': pred_segs.shape[0]
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='评估双模态MedSAM结果')
    parser.add_argument('-pred_dir', type=str, required=True, help='预测结果目录')
    parser.add_argument('-gt_dir', type=str, required=True, help='真实标签目录')
    parser.add_argument('-output_csv', type=str, default='evaluation_results.csv', help='输出CSV文件')
    
    args = parser.parse_args()
    
    # 评估预测结果
    results = evaluate_predictions(args.pred_dir, args.gt_dir)
    
    if not results:
        print("没有找到有效的预测结果")
        return
    
    # 计算总体指标
    dices = [r['dice'] for r in results]
    hausdorffs = [r['hausdorff'] for r in results if r['hausdorff'] != float('inf')]
    
    print(f"\n=== 评估结果 ===")
    print(f"病例数: {len(results)}")
    print(f"平均Dice分数: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"中位数Dice分数: {np.median(dices):.4f}")
    print(f"最高Dice分数: {np.max(dices):.4f}")
    print(f"最低Dice分数: {np.min(dices):.4f}")
    
    if hausdorffs:
        print(f"平均Hausdorff距离: {np.mean(hausdorffs):.4f} ± {np.std(hausdorffs):.4f}")
    
    # 保存详细结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n详细结果已保存到: {args.output_csv}")
    
    # 按Dice分数排序显示前5个和后5个病例
    df_sorted = df.sort_values('dice', ascending=False)
    print(f"\n=== 最佳5个病例 ===")
    print(df_sorted.head()[['case', 'dice', 'hausdorff']].to_string(index=False))
    
    print(f"\n=== 最差5个病例 ===")
    print(df_sorted.tail()[['case', 'dice', 'hausdorff']].to_string(index=False))

if __name__ == '__main__':
    main()
