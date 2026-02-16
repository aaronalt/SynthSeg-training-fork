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

# --- Asymmetry ratio per subject ---
from collections import defaultdict

subject_vols = defaultdict(dict)
for r in results:
    subject_vols[r['subject']][r['hemi']] = r['volume_mm3']

print("\nAsymmetry index (LH - RH) / (LH + RH):")
print("  0 = symmetric, positive = LH larger, negative = RH larger\n")
for sub in sorted(subject_vols):
    if 'lh' in subject_vols[sub] and 'rh' in subject_vols[sub]:
        lh = subject_vols[sub]['lh']
        rh = subject_vols[sub]['rh']
        ai = (lh - rh) / (lh + rh)
        print(f"  {sub}  LH={lh:.0f}  RH={rh:.0f}  AI={ai:+.3f}")

# --- Volume z-scores (separate for lh and rh) ---
lh_vols = [(r['subject'], r['volume_mm3']) for r in results if r['hemi'] == 'lh']
rh_vols = [(r['subject'], r['volume_mm3']) for r in results if r['hemi'] == 'rh']

def print_zscores(vols, hemi_label):
    subs, vals = zip(*vols)
    vals = np.array(vals)
    mean = vals.mean()
    std = vals.std()
    print(f"\n{hemi_label}:  mean={mean:.0f} mm³,  std={std:.0f} mm³")
    for sub, vol in sorted(zip(subs, vals), key=lambda x: abs((x[1]-mean)/std), reverse=True):
        z = (vol - mean) / std
        flag = '  ***' if abs(z) > 2 else '  *' if abs(z) > 1.5 else ''
        print(f"  {sub}  vol={vol:.0f}  z={z:+.2f}{flag}")

print_zscores(lh_vols, 'LH')
print_zscores(rh_vols, 'RH')