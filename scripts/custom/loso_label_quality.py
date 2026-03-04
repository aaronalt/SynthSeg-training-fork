"""
LOSO Label Quality Assessment

Runs Leave-One-Subject-Out cross-validation using quick SynthSeg models to
identify outlier labels before using them as training data.

Each fold: train on N-1 subjects → predict on held-out → evaluate vs GT
Folds are distributed across 2 GPUs in parallel.

Usage:
    python loso_label_quality.py

"""

import os
import sys

# Resolve paths relative to this file's location, not the calling directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_SCRIPT_DIR, '../..')))

import json
import subprocess
import time
from glob import glob
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd

# =====================================================================
# CONFIG
# =====================================================================
LABELS_DIR          = '/home/althause/data/7T/training_labels_native/t1w'
GT_DIRS             = [
    '/home/althause/data/claustrum_gt/7T/T1_CONTROL',
    '/home/althause/data/claustrum_gt/7T/T2',
    '/home/althause/data/claustrum_gt/3T/T1_CONTROL',
    '/home/althause/data/claustrum_gt/3T/T1_VCFS',
]
OUTPUT_DIR          = '/home/althause/data/loso_label_quality'
CHECKPOINT          = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260218_211654/dice_finetune_068_100.h5'
INTENSITY_PRIORS    = '/home/althause/data/intensity_estimation/images'

# Quick training settings — increase for more accurate LOO estimates
EPOCHS_PER_FOLD     = 10    # 5-10 is enough to identify outliers
STEPS_PER_EPOCH     = 1000  # 1000 steps × 10 epochs = 10k steps total per fold

# Quality thresholds — subjects below these are flagged for review
DICE_THRESHOLD      = 0.50
RECALL_THRESHOLD    = 0.45

# GPUs to use (indices into CUDA_VISIBLE_DEVICES)
GPUS                = [0, 1, 2, 3]  # one fold at a time per GPU
# =====================================================================

WORKER_SCRIPT = os.path.join(_SCRIPT_DIR, 'loso_worker.py')
PYTHON        = sys.executable


def get_subjects(labels_dir):
    """Extract unique subject IDs from label filenames."""
    files = sorted(glob(os.path.join(labels_dir, '*.nii.gz')))
    subjects = set()
    for f in files:
        name = Path(f).stem.replace('.nii', '')
        import re
        m = re.match(r'(sub-\d+|\d+)', name)
        if m:
            sid = m.group(1)
            if not sid.startswith('sub-'):
                sid = f'sub-{sid}'
            subjects.add(sid)
    return sorted(subjects)


def run_fold_subprocess(subject, gpu_id):
    """Launch loso_worker.py for one fold on a specific GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        PYTHON, WORKER_SCRIPT,
        '--held_out',             subject,
        '--labels_dir',           LABELS_DIR,
        '--gt_dirs',              *GT_DIRS,
        '--output_dir',           OUTPUT_DIR,
        '--checkpoint',           CHECKPOINT,
        '--intensity_priors_dir', INTENSITY_PRIORS,
        '--epochs',               str(EPOCHS_PER_FOLD),
        '--steps_per_epoch',      str(STEPS_PER_EPOCH),
    ]

    log_path = os.path.join(OUTPUT_DIR, f'fold_{subject}.log')
    print(f"  [GPU {gpu_id}] Starting fold: {subject} → log: {log_path}")

    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc


def gpu_worker(gpu_id, job_queue, done_list):
    """Thread that pulls subjects from the queue and runs them on one GPU."""
    while True:
        subject = job_queue.get()
        if subject is None:
            break
        proc = run_fold_subprocess(subject, gpu_id)
        proc.wait()
        rc = proc.returncode
        status = 'OK' if rc == 0 else f'FAILED (rc={rc})'
        print(f"  [GPU {gpu_id}] Finished fold: {subject} — {status}")
        done_list.append((subject, rc))
        job_queue.task_done()


def collect_results(output_dir, subjects):
    """Aggregate per-fold JSON results into a summary DataFrame."""
    rows = []
    for subject in subjects:
        results_path = os.path.join(output_dir, f'fold_{subject}', 'results.json')
        if not os.path.exists(results_path):
            print(f"  WARNING: No results for {subject}")
            continue
        with open(results_path) as f:
            fold_results = json.load(f)
        for r in fold_results:
            gt_dir = r.get('gt_dir', '')
            rows.append({
                'subject':    r.get('held_out_subject', subject),
                'subject_id': r.get('subject_id'),
                'hemisphere': r.get('hemisphere'),
                'gt_dir':     gt_dir,
                'gt_type':    os.path.basename(gt_dir) if gt_dir else '',
                'dice':       r.get('dice'),
                'precision':  r.get('precision'),
                'recall':     r.get('recall'),
                'iou':        r.get('iou'),
                'hausdorff':  r.get('hausdorff_95_mm'),
                'vol_pred':   r.get('volume_pred_mm3'),
                'vol_gt':     r.get('volume_gt_mm3'),
            })
    return pd.DataFrame(rows)


def print_quality_report(df, dice_thresh, recall_thresh):
    """Print ranked quality report and flag outliers."""
    # Print per-GT-type breakdown if multiple GT types present
    if 'gt_type' in df.columns and df['gt_type'].nunique() > 1:
        print(f"\n{'='*70}")
        print("PER GT-TYPE SUMMARY")
        print(f"{'='*70}")
        gt_summary = df.groupby('gt_type').agg(
            n=('dice', 'count'),
            dice_mean=('dice', 'mean'),
            dice_std=('dice', 'std'),
            recall_mean=('recall', 'mean'),
        ).reset_index()
        for _, row in gt_summary.iterrows():
            print(f"  {row['gt_type']:<20} n={int(row['n']):>3}  "
                  f"Dice={row['dice_mean']:.4f}±{row['dice_std']:.4f}  "
                  f"Recall={row['recall_mean']:.4f}")

    # Average LH + RH per subject (over all GT types)
    summary = df.groupby('subject').agg(
        dice_mean=('dice', 'mean'),
        dice_min=('dice', 'min'),
        precision_mean=('precision', 'mean'),
        recall_mean=('recall', 'mean'),
        iou_mean=('iou', 'mean'),
    ).reset_index().sort_values('dice_mean')

    print(f"\n{'='*70}")
    print("LABEL QUALITY REPORT (ranked by mean Dice, lowest first)")
    print(f"{'='*70}")
    print(f"{'Subject':<15} {'Dice':>7} {'Min':>7} {'Prec':>7} {'Recall':>7} {'IoU':>7}  Flag")
    print("-" * 70)
    for _, row in summary.iterrows():
        flagged = []
        if row['dice_mean'] < dice_thresh:
            flagged.append(f'Dice<{dice_thresh}')
        if row['recall_mean'] < recall_thresh:
            flagged.append(f'Recall<{recall_thresh}')
        flag_str = ' ⚠ ' + ', '.join(flagged) if flagged else ''
        print(f"{row['subject']:<15} {row['dice_mean']:7.4f} {row['dice_min']:7.4f} "
              f"{row['precision_mean']:7.4f} {row['recall_mean']:7.4f} "
              f"{row['iou_mean']:7.4f}{flag_str}")

    flagged_subjects = summary[
        (summary['dice_mean'] < dice_thresh) |
        (summary['recall_mean'] < recall_thresh)
    ]['subject'].tolist()

    print(f"\n{'='*70}")
    print(f"Flagged for review ({len(flagged_subjects)}/{len(summary)}): {flagged_subjects}")
    print(f"Overall mean Dice: {summary['dice_mean'].mean():.4f} ± {summary['dice_mean'].std():.4f}")
    print(f"{'='*70}")
    return summary, flagged_subjects


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = get_subjects(LABELS_DIR)
    print(f"Found {len(subjects)} subjects for LOSO: {subjects}")
    print(f"Using {len(GPUS)} GPUs: {GPUS} (1 fold per GPU)")
    print(f"Epochs per fold: {EPOCHS_PER_FOLD}, Steps: {STEPS_PER_EPOCH}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Skip folds that already have results
    pending = [s for s in subjects
               if not os.path.exists(os.path.join(OUTPUT_DIR, f'fold_{s}', 'results.json'))]
    already_done = len(subjects) - len(pending)
    if already_done:
        print(f"Skipping {already_done} already-completed folds.")
    print(f"Running {len(pending)} folds...\n")

    # Fill job queue, then add one sentinel per GPU at the end
    job_queue = Queue()
    for s in pending:
        job_queue.put(s)
    for _ in GPUS:
        job_queue.put(None)  # one sentinel per thread, added after all jobs

    # Launch GPU worker threads
    done_list = []
    threads = []
    for gpu_id in GPUS:
        t = Thread(target=gpu_worker, args=(gpu_id, job_queue, done_list), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\nAll folds complete. Collecting results...")

    # Aggregate results
    df = collect_results(OUTPUT_DIR, subjects)
    if df.empty:
        print("No results found. Check fold logs for errors.")
        sys.exit(1)

    # Save full results CSV
    csv_path = os.path.join(OUTPUT_DIR, 'loso_quality_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}")

    # Print quality report
    summary, flagged = print_quality_report(df, DICE_THRESHOLD, RECALL_THRESHOLD)

    # Save summary CSV
    summary_path = os.path.join(OUTPUT_DIR, 'loso_quality_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
