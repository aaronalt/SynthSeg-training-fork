import os
import nibabel as nib
import numpy as np

label_dir = '/home/althause/data/7T/training_labels_native/t1w/'


def get_label_morphometrics(label_path, label_value):
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


results = []
for fname in sorted(os.listdir(label_dir)):
    if not fname.endswith('.nii.gz'):
        continue

    hemi = 'lh' if '_lh_' in fname else 'rh'
    label_value = 138 if hemi == 'lh' else 139
    sub = fname.split('_ses')[0]  # e.g. 'sub-5166'

    metrics = get_label_morphometrics(os.path.join(label_dir, fname), label_value)
    if metrics:
        metrics['subject'] = sub
        metrics['hemi'] = hemi
        metrics['filename'] = fname
        results.append(metrics)
    else:
        print(f"WARNING: no label {label_value} found in {fname}")

for r in results:
    print(
        f"{r['subject']} {r['hemi']}  vol={r['volume_mm3']:.1f} mm³  bbox={tuple(round(x, 1) for x in r['bbox_extent_mm'])}")