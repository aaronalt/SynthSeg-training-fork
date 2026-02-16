import nibabel as nib
import numpy as np

# --- Edit this ---
label_path = '/home/althause/data/claustrum_gt/7T/T1_CONTROL/sub-7806_ses-T3_t1w_claustrum_label_rh.nii.gz'
label_value = 139  # 138=left, 139=right


# -----------------

def get_label_morphometrics(label_path, label_value):
    """
    Load a NIfTI label map and compute basic morphometrics for a given label.

    Returns dict with:
      - volume_mm3: total volume in mm³
      - voxel_count: number of voxels
      - bbox_extent_mm: bounding box size along x, y, z in mm
      - center_of_mass_vox: center of mass in voxel coordinates
    """
    img = nib.load(label_path)
    data = img.get_fdata()
    voxel_sizes = img.header.get_zooms()[:3]

    mask = (data == label_value)
    voxel_count = int(mask.sum())

    if voxel_count == 0:
        return None

    voxel_volume = float(np.prod(voxel_sizes))
    volume_mm3 = voxel_count * voxel_volume

    coords = np.argwhere(mask)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_extent_vox = bbox_max - bbox_min + 1
    bbox_extent_mm = bbox_extent_vox * np.array(voxel_sizes)

    center_of_mass_vox = coords.mean(axis=0)

    return {
        'volume_mm3': volume_mm3,
        'voxel_count': voxel_count,
        'bbox_extent_mm': tuple(bbox_extent_mm),
        'center_of_mass_vox': tuple(center_of_mass_vox),
        'voxel_sizes': tuple(voxel_sizes),
    }


result = get_label_morphometrics(label_path, label_value)
print(result)