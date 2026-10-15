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
    rel = str(Path(path).relative_to(root))
    match = re.search(r'(\d+).*?\b(lh|rh)\b', rel.replace('_', '.'), re.IGNORECASE)
    return (str(int(match.group(1))), match.group(2).lower()) if match else None


def eval_seg_vs_gt(seg_dir, gt_dir, out_dir):
    segs = {subject_hemi(p, seg_dir): p for p in list_niftis(seg_dir)}
    gts = {subject_hemi(p, gt_dir): p for p in list_niftis(gt_dir)}
    segs.pop(None, None), gts.pop(None, None)

    keys = sorted(set(segs) & set(gts), key=lambda k: (int(k[0]), k[1]))
    missing = sorted(set(gts) - set(segs))
    if missing:
        print(f'WARNING: no segmentation for GT keys {missing}')
    if not keys:
        raise ValueError(f'No (subject, hemisphere) pairs matched between {seg_dir} and {gt_dir}')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='eval_seg_vs_gt_') as temp_dir:
        aligned_seg_dir = os.path.join(temp_dir, 'segmentations')
        os.makedirs(aligned_seg_dir)
        aligned_gt_paths = []
        with open(out_dir / 'subjects.txt', 'w') as f:
            for index, key in enumerate(keys):
                seg_image, gt_image = nib.load(segs[key]), nib.load(gts[key])

                # binary {0,1} maps (no 138/139 values) get the hemisphere's label
                seg_data = np.asarray(seg_image.dataobj)
                if seg_data.max() <= 1:
                    seg_image = nib.Nifti1Image((seg_data > 0).astype(np.int32) * HEMI_LABEL[key[1]],
                                                seg_image.affine, seg_image.header)

                aligned_seg = resample_from_to(seg_image, (gt_image.shape, gt_image.affine), order=0)
                name = f'{index:06d}_sub-{key[0]}_{key[1]}'
                nib.save(aligned_seg, os.path.join(aligned_seg_dir, f'{name}_seg.nii.gz'))
                aligned_gt_path = os.path.join(temp_dir, f'{name}_gt.nii.gz')
                nib.save(gt_image, aligned_gt_path)
                aligned_gt_paths.append(aligned_gt_path)

                f.write(f'sub-{key[0]}\t{key[1]}\t{segs[key]}\t{gts[key]}\n')
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
    print(f'\nWrote metrics for {len(keys)} segmentations to {out_dir} (order in subjects.txt)')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    eval_seg_vs_gt(*sys.argv[1:])
