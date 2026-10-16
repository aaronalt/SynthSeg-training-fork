"""
Evaluate an existing folder of segmentations against a GT folder, without running prediction.
Mirrors the evaluation in SynthSeg/predict.py (each segmentation is resampled onto its GT grid with
nearest neighbour, then evaluate.evaluation), but orders subjects by (subject, hemisphere) key, so the
.npy outputs of different seg folders evaluated against the same GT have identical column order and
can be compared with `collect_metrics.py --compare`.

Usage:
    python eval_seg_vs_gt.py SEG_DIR GT_DIR OUT_DIR

Example (Mauri's model vs manual 3T GT, then your model vs the same GT):
    python eval_seg_vs_gt.py /home/althause/data/claustrum_gt/mauri_3t /home/althause/data/claustrum_gt/3T \
        /home/althause/data/seg/eval_3T/mauri_model
    python eval_seg_vs_gt.py /home/althause/data/seg/<exp>/<checkpoint> /home/althause/data/claustrum_gt/3T \
        /home/althause/data/seg/eval_3T/<checkpoint>
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from SynthSeg import evaluate
from SynthSeg.predict import best_gt_match

EVAL_LABELS = [138, 139]
HEMI_LABEL = {'lh': 138, 'rh': 139}


def list_niftis(folder):
    """Label maps directly in folder or in subfolders (e.g. sub-XXXX/ses-*/anat/claustrum.lh.nii.gz),
    skipping SynthSeg side outputs (posteriors/, resampled/)."""
    folder = Path(folder)
    return sorted(p for p in folder.rglob('*.nii*')
                  if not {'posteriors', 'resampled', 'mauri'} & set(p.relative_to(folder).parts[:-1]))


def subject_hemi(path, root):
    """(subject_id, 'lh'|'rh') from the path relative to root, e.g. sub-6568/.../claustrum.lh.nii.gz."""
    rel = str(Path(path).relative_to(root)).replace('_', '.')
    # prefer an explicit sub-<id>, so digits in folder names (e.g. 7T/T1/...) are not taken as the subject
    match = (re.search(r'sub-0*(\d+).*?\b(lh|rh)\b', rel, re.IGNORECASE)
             or re.search(r'(\d+).*?\b(lh|rh)\b', rel, re.IGNORECASE))
    return (str(int(match.group(1))), match.group(2).lower()) if match else None


def match_segs_to_gt(seg_dir, gt_dir):
    """[(subject, hemi, gt_relpath, seg_path, gt_path)] sorted by GT, one row per matched segmentation.
    Segmentations are matched to GT by (subject, hemisphere), then by contrast/session/orientation when a
    subject and hemisphere have several GT files (e.g. 7T t1w / t2w-cor / t2w-tra)."""
    gts = {}
    for p in list_niftis(gt_dir):
        key = subject_hemi(p, gt_dir)
        if key is not None:
            gts.setdefault(key, []).append((p, str(Path(p).relative_to(gt_dir))))
    rows, unmatched = [], []
    for seg in list_niftis(seg_dir):
        key = subject_hemi(seg, seg_dir)
        if key not in gts:
            unmatched.append(seg)
            continue
        gt = best_gt_match(str(Path(seg).relative_to(seg_dir)), gts[key])
        rows.append((key[0], key[1], str(Path(gt).relative_to(gt_dir)), seg, gt))
    used = [r[2] for r in rows]
    duplicates = sorted({g for g in used if used.count(g) > 1})
    if duplicates:
        raise ValueError(f'Several segmentations in {seg_dir} matched the same GT files {duplicates}')
    if unmatched:
        print(f'WARNING: no GT for {len(unmatched)} segmentation(s), e.g. {unmatched[0]}')
    missing = sorted(set(rel for v in gts.values() for _, rel in v) - set(used))
    if missing:
        print(f'WARNING: no segmentation for {len(missing)} GT file(s): {missing}')
    return sorted(rows, key=lambda r: (int(r[0]), r[1], r[2]))


def eval_seg_vs_gt(seg_dir, gt_dir, out_dir, gt_subset=None):
    """Evaluate seg_dir against gt_dir into out_dir (metric .npy files + subjects.txt). gt_subset: optional
    collection of GT relative paths to restrict to (e.g. the GT files two models have in common)."""
    rows = match_segs_to_gt(seg_dir, gt_dir)
    if gt_subset is not None:
        rows = [r for r in rows if r[2] in set(gt_subset)]
    if not rows:
        raise ValueError(f'No segmentation matched a GT file between {seg_dir} and {gt_dir}')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='eval_seg_vs_gt_') as temp_dir:
        aligned_seg_dir = os.path.join(temp_dir, 'segmentations')
        os.makedirs(aligned_seg_dir)
        aligned_gt_paths = []
        with open(out_dir / 'subjects.txt', 'w') as f:
            for index, (subject, hemi, gt_rel, seg_path, gt_path) in enumerate(rows):
                seg_image, gt_image = nib.load(seg_path), nib.load(gt_path)

                # binary {0,1} maps (no 138/139 values) get the hemisphere's label
                seg_data = np.asarray(seg_image.dataobj)
                if seg_data.max() <= 1:
                    seg_image = nib.Nifti1Image((seg_data > 0).astype(np.int32) * HEMI_LABEL[hemi],
                                                seg_image.affine, seg_image.header)

                aligned_seg = resample_from_to(seg_image, (gt_image.shape, gt_image.affine), order=0)
                name = f'{index:06d}_sub-{subject}_{hemi}'
                nib.save(aligned_seg, os.path.join(aligned_seg_dir, f'{name}_seg.nii.gz'))
                aligned_gt_path = os.path.join(temp_dir, f'{name}_gt.nii.gz')
                nib.save(gt_image, aligned_gt_path)
                aligned_gt_paths.append(aligned_gt_path)

                f.write(f'sub-{subject}\t{hemi}\t{gt_rel}\t{seg_path}\t{gt_path}\n')
                print(f'{name}: seg voxel size {seg_image.header.get_zooms()[:3]}, '
                      f'GT voxel size {gt_image.header.get_zooms()[:3]}')

        evaluate.evaluation(aligned_gt_paths,
                            aligned_seg_dir,
                            label_list=EVAL_LABELS,
                            path_dice=str(out_dir / 'dice.npy'),
                            path_hausdorff=str(out_dir / 'hausdorff.npy'),
                            path_hausdorff_99=str(out_dir / 'hausdorff_99.npy'),
                            path_hausdorff_95=str(out_dir / 'hausdorff_95.npy'),
                            path_mean_distance=str(out_dir / 'mean_distance.npy'),
                            path_precision=str(out_dir / 'precision.npy'),
                            path_recall=str(out_dir / 'recall.npy'),
                            recompute=True)
    print(f'\nWrote metrics for {len(rows)} segmentations to {out_dir} (order in subjects.txt)')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    eval_seg_vs_gt(*sys.argv[1:])
