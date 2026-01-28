#!/usr/bin/env python3
"""
Evaluate Mauri Segmentation by Resampling Manual Labels to Mauri Space

Instead of resampling Mauri to manual space, this resamples manual labels
to Mauri's native 0.35mm resolution for comparison.

Note: Mauri uses labels 138 (left) and 139 (right) for claustrum
      Manual ground truth uses label 1 for claustrum

Author: Aaron
Date: January 2026
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
import json
import os
import glob
import re
import pandas as pd

def resample_to_target(img_nii, reference_nii):
    """Resample image to match reference image space exactly using nibabel"""
    from nibabel.processing import resample_from_to
    # Nearest neighbor (order=0) is critical for categorical labels
    return resample_from_to(img_nii, reference_nii, order=0)


def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient for binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def compute_hausdorff(pred_mask, gt_mask, voxel_size):
    """Compute 95th percentile Hausdorff distance in mm"""
    if not np.any(pred_mask) or not np.any(gt_mask):
        return np.inf

    # Get surfaces
    p_surf = pred_mask ^ binary_erosion(pred_mask)
    g_surf = gt_mask ^ binary_erosion(gt_mask)

    if not np.any(p_surf) or not np.any(g_surf):
        return 0.0

    d_p_to_g = distance_transform_edt(~g_surf)
    d_g_to_p = distance_transform_edt(~p_surf)

    # 95th percentile
    hd = max(np.percentile(d_p_to_g[p_surf], 95),
             np.percentile(d_g_to_p[g_surf], 95))

    return hd * np.mean(voxel_size)


def evaluate(mauri_path, manual_path, subject_id=None, hemi=None,
             save_resampled=False, output_dir=None, save_output=False, group='vcfs', model=None, prediction_path=None):
    """
    Evaluates by resampling the high-res Mauri prediction down to the
    Manual Ground Truth space.
    """
    # 1. Load images
    mauri_nii = nib.load(mauri_path)
    manual_nii = nib.load(manual_path)

    # 2. Resample PREDICTION to MANUAL space
    # This direction is more stable for small, thin structures
    print(f"  Resampling Mauri (0.35mm) to Manual space...")
    mauri_resampled_nii = resample_to_target(mauri_nii, manual_nii)

    # 3. Save resampled file if requested
    if save_resampled and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{subject_id}_resampled_to_manual.nii.gz")
        nib.save(mauri_resampled_nii, out_path)
        print(f"  Saved resampled prediction: {out_path}")

    # 4. Extract data and force to Integer (prevents label mismatch)
    pred_data = np.round(mauri_resampled_nii.get_fdata()).astype(np.int16)
    gt_data = np.round(manual_nii.get_fdata()).astype(np.int16)

    # 5. Define Labels (Using your specific IDs)
    MAURI_L, MAURI_R = 138, 139

    # 6. Create Binary Masks
    pred_combined = np.isin(pred_data, [138, 139])
    manual_present = np.unique(gt_data)
    # Check for specific hemisphere labels first
    if 138 in manual_present or 139 in manual_present:
        # Grabs both if present, or just one if only one exists
        gt_combined = np.isin(gt_data, [138, 139])
    elif 1 in manual_present:
        gt_combined = (gt_data == 1)
    else:
        # Fallback: if no standard labels found, treat all non-zero as claustrum
        gt_combined = (gt_data > 0)

    intersection = np.logical_and(pred_combined, gt_combined).sum()
    print(f"  DEBUG: Intersection voxels: {intersection}")

    # 7. Compute Metrics
    manual_pixdim = nib.affines.voxel_sizes(manual_nii.affine)
    voxel_vol = np.prod(manual_pixdim)

    # Calculate Dice
    dice_combined = compute_dice(pred_combined, gt_combined)

    # Calculate Volume
    vol_pred = pred_combined.sum() * voxel_vol
    vol_gt = gt_combined.sum() * voxel_vol
    vol_diff_pct = ((vol_pred - vol_gt) / vol_gt * 100) if vol_gt > 0 else 0

    results = {
        'subject_id': subject_id,
        'hemisphere': hemi,
        'group': group,
        'comparison_space': 'manual',
        'dice': float(dice_combined),
        'hausdorff_95_mm': float(compute_hausdorff(pred_combined, gt_combined, manual_pixdim)),
        'volume_pred_mm3': float(vol_pred),
        'volume_gt_mm3': float(vol_gt),
        'volume_diff_percent': float(vol_diff_pct),
        'left_clau_volume_mm3': float(np.sum(pred_data == MAURI_L) * voxel_vol),
        'right__clau_volume_mm3': float(np.sum(pred_data == MAURI_R) * voxel_vol),
        'model': model,
        'prediction_path': prediction_path
    }

    # Save if requested
    if save_output:
        df = pd.DataFrame(results, index=[0])
        headers = ['subject_id', 'hemisphere', 'group', 'comparison_space',
                  'dice', 'hausdorff_95_mm', 'volume_pred_mm3',
                  'volume_gt_mm3', 'volume_diff_percent', 'left_clau_volume_mm3',
                  'right__clau_volume_mm3', 'model', 'prediction_path']
        file = os.path.join(output_dir, 'dice.csv')
        file_exists = os.path.isfile(file)
        if not file_exists:
            df.to_csv(file, mode='w', index=False, header=headers)
        else:
            df.to_csv(file, mode='a', index=False, header=False)
        print(f"\nResults saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()