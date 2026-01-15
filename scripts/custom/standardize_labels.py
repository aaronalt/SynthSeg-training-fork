"""
standardize_labels.py

Standardize training labels to consistent size and resolution.
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import glob
import os


def standardize_training_labels(input_dir, output_dir, target_shape=(160, 160, 160), target_res=0.35):
    """
    Standardize all training labels to same size and resolution.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing original training labels
    output_dir : str
        Directory to save standardized labels
    target_shape : tuple
        Target shape for all labels (H, W, D)
    target_res : float
        Target isotropic resolution in mm
        
    Returns:
    --------
    output_dir : str
        Path to standardized labels directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    labels = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))
    
    print(f"\nStandardizing {len(labels)} labels to {target_shape} at {target_res}mm")
    
    for label_file in labels:
        basename = os.path.basename(label_file)
        
        # Load
        img = nib.load(label_file)
        data = img.get_fdata()
        affine = img.affine
        
        # Get current resolution
        voxel_size = np.abs(np.diag(affine)[:3])
        
        # Resample to target resolution
        zoom_factors = voxel_size / target_res
        data_resampled = zoom(data, zoom_factors, order=0)
        
        # Crop or pad to target shape
        output = np.zeros(target_shape, dtype=data_resampled.dtype)
        
        # Calculate crop/pad for each dimension
        for dim in range(3):
            if data_resampled.shape[dim] >= target_shape[dim]:
                start = (data_resampled.shape[dim] - target_shape[dim]) // 2
                end = start + target_shape[dim]
                if dim == 0:
                    data_resampled = data_resampled[start:end, :, :]
                elif dim == 1:
                    data_resampled = data_resampled[:, start:end, :]
                else:
                    data_resampled = data_resampled[:, :, start:end]
        
        # Center in output (handles padding)
        start_x = (target_shape[0] - data_resampled.shape[0]) // 2
        start_y = (target_shape[1] - data_resampled.shape[1]) // 2
        start_z = (target_shape[2] - data_resampled.shape[2]) // 2
        
        output[start_x:start_x+data_resampled.shape[0],
               start_y:start_y+data_resampled.shape[1],
               start_z:start_z+data_resampled.shape[2]] = data_resampled
        
        # Create new affine
        new_affine = np.eye(4)
        new_affine[:3, :3] = np.diag([target_res, target_res, target_res])
        
        # Save
        output_file = os.path.join(output_dir, basename)
        nib.save(nib.Nifti1Image(output.astype(data.dtype), new_affine), output_file)
        print(f"  {basename}: {img.shape} -> {output.shape}")
    
    print(f"✓ Saved to {output_dir}\n")
    return output_dir
