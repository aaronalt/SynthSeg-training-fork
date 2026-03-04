"""
GT Comparison: Evaluate pretrained checkpoint against all GT directories.

Runs prediction once per subject using the pretrained model, then evaluates
segmentations against every matching GT file (T1w, T2w, etc.) to compare
how well each GT type agrees with the model's output.

No training required — uses the existing checkpoint directly.

Usage:
    python scripts/custom/gt_comparison.py
    python scripts/custom/gt_comparison.py --output_dir /path/to/output
"""

import os
import sys
import re
import json
import argparse
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

from SynthSeg.predict import predict
import compare_dice_batch as evaluate

# =====================================================================
# CONFIG
# =====================================================================
MRI_IMAGES_DIR   = '/home/althause/data/TEST/trainingset'
GT_DIRS          = {
    'T1_CONTROL_7T': '/home/althause/data/claustrum_gt/7T/T1_CONTROL',
    'T2_7T':         '/home/althause/data/claustrum_gt/7T/T2',
    'T1_CONTROL_3T': '/home/althause/data/claustrum_gt/3T/T1_CONTROL',
    'T1_VCFS_3T':    '/home/althause/data/claustrum_gt/3T/T1_VCFS',
}
CHECKPOINT       = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260218_211654/dice_finetune_068_100.h5'
OUTPUT_DIR       = '/home/althause/data/gt_comparison'

SEGMENTATION_LABELS = np.array([0, 14, 15, 16, 24,
                                  2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                  41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
N_NEUTRAL_LABELS = 5
# =====================================================================


def find_all_gts(subject_id, hemisphere):
    """Return list of (label, gt_path) for all GT dirs that have a match."""
    patterns = [
        f'*{subject_id}*{hemisphere}*.nii.gz',
        f'sub-{subject_id}*{hemisphere}*.nii.gz',
        f'*{subject_id}*.nii.gz',
    ]
    found = []
    for label, gt_dir in GT_DIRS.items():
        p = Path(gt_dir)
        if not p.exists():
            continue
        for pat in patterns:
            matches = list(p.glob(pat))
            if matches:
                found.append((label, matches[0]))
                break
    return found


def get_subjects():
    """Extract subject IDs from MRI image filenames."""
    files = sorted(glob(os.path.join(MRI_IMAGES_DIR, '*.nii.gz')))
    subjects = set()
    for f in files:
        name = Path(f).stem.replace('.nii', '')
        m = re.match(r'(sub-\d+|\d+)', name)
        if m:
            sid = m.group(1)
            if not sid.startswith('sub-'):
                sid = f'sub-{sid}'
            subjects.add(sid)
    return sorted(subjects)


def run_prediction(subject_id, seg_dir):
    """Run predict for one subject, return list of output seg paths."""
    sid = subject_id.replace('sub-', '')

    # Find MRI image
    mri_matches = sorted(glob(os.path.join(MRI_IMAGES_DIR, f'*{sid}*.nii.gz')))
    if not mri_matches:
        mri_matches = sorted(glob(os.path.join(MRI_IMAGES_DIR, f'*{subject_id}*.nii.gz')))
    if not mri_matches:
        print(f"  [{subject_id}] ERROR: No MRI image found.")
        return []

    # Symlink to a temp input dir (predict() takes a directory)
    input_dir = os.path.join(seg_dir, f'_input_{subject_id}')
    os.makedirs(input_dir, exist_ok=True)
    for mri in mri_matches:
        dest = os.path.join(input_dir, os.path.basename(mri))
        if not os.path.exists(dest):
            os.symlink(os.path.abspath(mri), dest)

    subject_seg_dir = os.path.join(seg_dir, subject_id)
    os.makedirs(subject_seg_dir, exist_ok=True)

    predict(
        input_dir,
        subject_seg_dir,
        CHECKPOINT,
        SEGMENTATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        target_res=0.50,
        flip=True,
        sigma_smoothing=0.5,
        keep_biggest_component=False,
        n_levels=5,
        nb_conv_per_level=2,
        conv_size=3,
        unet_feat_count=24,
        feat_multiplier=2,
        activation='elu',
    )

    shutil.rmtree(input_dir, ignore_errors=True)
    return sorted(glob(os.path.join(subject_seg_dir, '*.nii.gz')))


def evaluate_subject(subject_id, seg_files):
    """Evaluate all seg files for a subject against all available GTs."""
    rows = []
    for seg_path in seg_files:
        name = Path(seg_path).stem.replace('.nii', '')
        m = re.search(r'(\d+).*?(lh|rh)', name, re.IGNORECASE)
        if not m:
            continue
        sid = m.group(1)
        hemisphere = m.group(2).lower()

        gt_matches = find_all_gts(sid, hemisphere)
        if not gt_matches:
            print(f"  [{subject_id}] No GT found for {sid} {hemisphere}")
            continue

        for gt_label, gt_path in gt_matches:
            res = evaluate.evaluate(
                seg_path, str(gt_path),
                subject_id=sid,
                hemi=hemisphere,
                save_output=False,
            )
            if res:
                rows.append({
                    'subject_id': subject_id,
                    'hemisphere': hemisphere,
                    'gt_type':    gt_label,
                    'dice':       res['dice'],
                    'precision':  res['precision'],
                    'recall':     res['recall'],
                    'iou':        res['iou'],
                    'hausdorff':  res.get('hausdorff_95_mm'),
                    'vol_pred':   res.get('volume_pred_mm3'),
                    'vol_gt':     res.get('volume_gt_mm3'),
                })
                print(f"  [{subject_id}] {hemisphere} vs {gt_label}: "
                      f"Dice={res['dice']:.4f}  Prec={res['precision']:.4f}  "
                      f"Recall={res['recall']:.4f}  IoU={res['iou']:.4f}")
    return rows


def print_report(df):
    """Print summary table grouped by GT type."""
    print(f"\n{'='*80}")
    print("RESULTS BY GT TYPE")
    print(f"{'='*80}")
    for gt_type, grp in df.groupby('gt_type'):
        print(f"\n  {gt_type}  (n={len(grp)})")
        print(f"    Dice:      mean={grp['dice'].mean():.4f}  std={grp['dice'].std():.4f}  "
              f"min={grp['dice'].min():.4f}  max={grp['dice'].max():.4f}")
        print(f"    Recall:    mean={grp['recall'].mean():.4f}  std={grp['recall'].std():.4f}")
        print(f"    Precision: mean={grp['precision'].mean():.4f}  std={grp['precision'].std():.4f}")

    print(f"\n{'='*80}")
    print("PER-SUBJECT SUMMARY (mean across hemispheres and GT types)")
    print(f"{'='*80}")
    subj_summary = df.groupby(['subject_id', 'gt_type'])['dice'].mean().unstack(fill_value=float('nan'))
    print(subj_summary.round(4).to_string())


def create_plots(df, output_dir):
    """Create comparison plots across GT types."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    gt_types = sorted(df['gt_type'].unique())
    colors = plt.cm.tab10.colors

    # 1. Dice per GT type — boxplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data_by_type = [df[df['gt_type'] == t]['dice'].dropna().values for t in gt_types]
    axes[0].boxplot(data_by_type, labels=gt_types, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[0].set_ylabel('Dice')
    axes[0].set_title('Dice by GT Type')
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha='right')

    data_by_type_recall = [df[df['gt_type'] == t]['recall'].dropna().values for t in gt_types]
    axes[1].boxplot(data_by_type_recall, labels=gt_types, patch_artist=True,
                    boxprops=dict(facecolor='coral', alpha=0.6))
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall by GT Type')
    axes[1].axhline(y=0.45, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_dice_recall_by_gt_type.png'), dpi=150)
    plt.close()

    # 2. Per-subject Dice heatmap across GT types
    pivot = df.groupby(['subject_id', 'gt_type'])['dice'].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, len(gt_types) * 2), max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title('Per-Subject Dice by GT Type')
    plt.colorbar(im, ax=ax, label='Dice')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if 0.3 < val < 0.8 else 'white')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_subject_dice_heatmap.png'), dpi=150)
    plt.close()

    # 3. Scatter: T1w Dice vs T2w Dice per subject (if both exist)
    t1_types = [t for t in gt_types if 'T1' in t]
    t2_types = [t for t in gt_types if 'T2' in t]
    if t1_types and t2_types:
        t1_mean = df[df['gt_type'].isin(t1_types)].groupby('subject_id')['dice'].mean()
        t2_mean = df[df['gt_type'].isin(t2_types)].groupby('subject_id')['dice'].mean()
        common = t1_mean.index.intersection(t2_mean.index)
        if len(common) > 1:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(t1_mean[common], t2_mean[common], s=60, color='steelblue', alpha=0.8)
            for sid in common:
                ax.annotate(sid, (t1_mean[sid], t2_mean[sid]), fontsize=7,
                            xytext=(4, 4), textcoords='offset points')
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.4, label='y=x')
            ax.set_xlabel('T1w GT Dice (mean)')
            ax.set_ylabel('T2w GT Dice (mean)')
            ax.set_title('T1w vs T2w GT Agreement per Subject')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, '03_t1w_vs_t2w_dice.png'), dpi=150)
            plt.close()

    print(f"Plots saved to {plots_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=OUTPUT_DIR)
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Limit to specific subject IDs (e.g. sub-5166 sub-8399)')
    parser.add_argument('--repredict', action='store_true',
                        help='Re-run prediction even if segmentation already exists')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seg_dir = os.path.join(args.output_dir, 'segmentations')
    os.makedirs(seg_dir, exist_ok=True)

    subjects = args.subjects if args.subjects else get_subjects()
    print(f"Found {len(subjects)} subjects: {subjects}")
    print(f"GT types: {list(GT_DIRS.keys())}")
    print(f"Output: {args.output_dir}\n")

    all_rows = []
    for subject_id in subjects:
        print(f"\n[{subject_id}] Running prediction...")
        subject_seg_dir = os.path.join(seg_dir, subject_id)
        existing_segs = sorted(glob(os.path.join(subject_seg_dir, '*.nii.gz')))

        if existing_segs and not args.repredict:
            print(f"  [{subject_id}] Using existing segmentations ({len(existing_segs)} files).")
            seg_files = existing_segs
        else:
            seg_files = run_prediction(subject_id, seg_dir)

        if not seg_files:
            print(f"  [{subject_id}] No segmentations — skipping evaluation.")
            continue

        rows = evaluate_subject(subject_id, seg_files)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo results collected.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, 'gt_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print_report(df)
    create_plots(df, args.output_dir)
    print("\nDone.")
