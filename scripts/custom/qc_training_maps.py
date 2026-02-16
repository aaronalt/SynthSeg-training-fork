import os
import nibabel as nib
import numpy as np

label_dir = '/home/althause/data/7T/training_labels_native/t2w-tra/'


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

from scipy import ndimage


def get_surface_area(label_path, label_value):
    """
    Estimate surface area by counting exposed voxel faces.
    Returns surface area in mm².
    """
    img = nib.load(label_path)
    data = img.get_fdata()
    voxel_sizes = img.header.get_zooms()[:3]

    mask = (data == label_value).astype(np.uint8)

    # count exposed faces along each axis
    face_areas = [
        voxel_sizes[1] * voxel_sizes[2],  # x-faces
        voxel_sizes[0] * voxel_sizes[2],  # y-faces
        voxel_sizes[0] * voxel_sizes[1],  # z-faces
    ]

    sa = 0.0
    for axis in range(3):
        diff = np.diff(mask, axis=axis)
        sa += np.count_nonzero(diff) * face_areas[axis]

    # add boundary faces (voxels touching image edge)
    for axis in range(3):
        first = np.take(mask, 0, axis=axis)
        last = np.take(mask, mask.shape[axis] - 1, axis=axis)
        sa += (first.sum() + last.sum()) * face_areas[axis]

    return sa

print("\nSA:V ratio (higher = thinner/more complex, lower = more blob-like):\n")
sav_results = []
for r in results:
    path = os.path.join(label_dir, r['filename'])
    label_value = 138 if r['hemi'] == 'lh' else 139
    sa = get_surface_area(path, label_value)
    sav = sa / r['volume_mm3']
    sav_results.append((r['subject'], r['hemi'], sav, sa, r['volume_mm3']))

# sort by SA:V
sav_results.sort(key=lambda x: x[2])
for sub, hemi, sav, sa, vol in sav_results:
    print(f"  {sub} {hemi}  SA:V={sav:.3f}  SA={sa:.0f} mm²  vol={vol:.0f} mm³")


# --- Summary of all flags ---
from collections import defaultdict

flags = defaultdict(list)

# 1. Bbox x-extent > 40mm
for r in results:
    x_extent = r['bbox_extent_mm'][0]
    if x_extent > 40:
        flags[(r['subject'], r['hemi'])].append(f"bbox_x={x_extent:.0f}mm")

# 2. Asymmetry |AI| > 0.15
for sub in sorted(subject_vols):
    if 'lh' in subject_vols[sub] and 'rh' in subject_vols[sub]:
        lh = subject_vols[sub]['lh']
        rh = subject_vols[sub]['rh']
        ai = (lh - rh) / (lh + rh)
        if abs(ai) > 0.15:
            flags[(sub, 'lh')].append(f"AI={ai:+.3f}")
            flags[(sub, 'rh')].append(f"AI={ai:+.3f}")

# 3. Volume |z| > 1.5
for hemi_label, vols in [('lh', lh_vols), ('rh', rh_vols)]:
    subs, vals = zip(*vols)
    vals = np.array(vals)
    mean, std = vals.mean(), vals.std()
    for sub, vol in zip(subs, vals):
        z = (vol - mean) / std
        if abs(z) > 1.5:
            flags[(sub, hemi_label)].append(f"vol_z={z:+.2f}")

# 4. SA:V in bottom or top 15%
sav_vals = [x[2] for x in sav_results]
low_thresh = np.percentile(sav_vals, 15)
high_thresh = np.percentile(sav_vals, 85)
for sub, hemi, sav, sa, vol in sav_results:
    if sav < low_thresh:
        flags[(sub, hemi)].append(f"SA:V_low={sav:.3f}")
    elif sav > high_thresh:
        flags[(sub, hemi)].append(f"SA:V_high={sav:.3f}")

# Print sorted by number of flags
print("Labels with flags (sorted by flag count):\n")
for key, flag_list in sorted(flags.items(), key=lambda x: -len(x[1])):
    sub, hemi = key
    print(f"  {sub} {hemi}  [{len(flag_list)} flags]  {', '.join(flag_list)}")

print(f"\nClean labels: {42 - len(flags)}/{42}")