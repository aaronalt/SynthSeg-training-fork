"""
Collect evaluation metrics (.npy: dice, hausdorff, hausdorff_95/99, mean_distance) into
metrics.csv (wide) and metrics_tidy.csv (long) WITHOUT re-running prediction or evaluation.

Usage:
    python collect_metrics.py [--models MODEL_DIR] [path ...]

Each path argument can be:
  - a checkpoint seg folder containing .npy files (processed directly), or
  - a parent folder, which is swept recursively for subfolders containing .npy files
    (e.g. /home/althause/data/seg/<experiment> sweeps all its checkpoint folders).
With no path arguments, sweeps the default segmentation root /home/althause/data/seg.

--models MODEL_DIR additionally plots training/validation loss curves from the
TensorBoard event files under MODEL_DIR (e.g. the experiment's model folder containing
logs/train and logs/validation) and writes loss_curves.png there.

If segmentations were deleted (DELETE_TMP_PREDICTIONS=True), subject columns fall back to
subject_0..N (the .npy files alone don't record file names).
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_merge import merge_metrics_npy, summarize_per_epoch, plot_loss_curves

DEFAULT_SEG_ROOT = '/home/althause/data/seg'


def find_seg_folders(path):
    """Return [path] if it directly contains .npy files, else its subfolders that do."""
    path = Path(path)
    if list(path.glob('*.npy')):
        return [path]
    return sorted(d for d in path.rglob('*') if d.is_dir() and list(d.glob('*.npy')))


def compute_extended_summary(folders, out_dir=None):
    """Computes mean, std, median, Q25, Q75, and IQR across subjects for all metrics in each folder."""
    summary_rows = []

    for folder in folders:
        folder = Path(folder)
        csv_path = folder / 'metrics.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        # Exclude non-numeric subject identifiers
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'subject']

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            mean_val = series.mean()
            std_val = series.std(ddof=1)  # Sample standard deviation
            median_val = series.median()
            q25_val = series.quantile(0.25)
            q75_val = series.quantile(0.75)
            iqr_val = q75_val - q25_val

            summary_rows.append({
                'folder': folder.name,
                'path': str(folder),
                'metric': col,
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'q25': q25_val,
                'q75': q75_val,
                'iqr': iqr_val,
                'mean_std_str': f"{mean_val:.4f} ± {std_val:.4f}",
                'median_iqr_str': f"{median_val:.4f} [{q25_val:.4f}, {q75_val:.4f}]"
            })

    if not summary_rows:
        return None

    summary_df = pd.DataFrame(summary_rows)

    if out_dir is None:
        out_dir = os.path.commonpath([str(f) for f in folders]) if len(folders) > 1 else str(folders[0])

    out_path = Path(out_dir) / 'metrics_summary_stats.csv'
    summary_df.to_csv(out_path, index=False)
    print(f'\nWrote extended statistical summary to: {out_path}')
    return out_path


if __name__ == '__main__':
    argv = sys.argv[1:]
    models_dir = None
    if '--models' in argv:
        i = argv.index('--models')
        models_dir = argv[i + 1]
        del argv[i:i + 2]
    args = argv or [DEFAULT_SEG_ROOT]
    n_written = 0
    loss_plotted = False
    for arg in args:
        folders = find_seg_folders(arg)
        if not folders:
            print(f'No .npy files found under {arg}')
        for folder in folders:
            print(f'\n=== {folder} ===')
            csv_path, _ = merge_metrics_npy(folder)
            n_written += csv_path is not None
        if folders:
            summarize_per_epoch(folders)
            compute_extended_summary(folders)
            if models_dir and not loss_plotted:
                plot_loss_curves(models_dir,
                                 out_dir=os.path.commonpath([str(f) for f in folders]))
                loss_plotted = True
    if models_dir and not loss_plotted:
        plot_loss_curves(models_dir)
    print(f'\nDone: wrote metrics CSVs for {n_written} folder(s).')
