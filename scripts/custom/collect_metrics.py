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
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_merge import merge_metrics_npy, summarize_per_epoch, plot_loss_curves

DEFAULT_SEG_ROOT = '/home/althause/data/seg'
IGNORE_METADATA_COLS = {'subject', 'label', 'structure', 'id', 'unnamed: 0'}


def find_seg_folders(path):
    """Return [path] if it directly contains .npy files, else its subfolders that do."""
    path = Path(path)
    if list(path.glob('*.npy')):
        return [path]
    return sorted(d for d in path.rglob('*') if d.is_dir() and list(d.glob('*.npy')))


def filter_hemisphere_mismatches(df):
    """
    Filters out cross-hemisphere evaluation artifacts:
    - If subject filename contains '.lh.', keep ONLY label 138 (Left)
    - If subject filename contains '.rh.', keep ONLY label 139 (Right)
    Handles both Tidy/Long format and Wide column formats.
    """
    df = df.copy()

    # Find the subject column (case-insensitive)
    subj_col = next((c for c in df.columns if c.lower() == 'subject'), None)
    if subj_col is None:
        return df

    subjects = df[subj_col].astype(str)
    is_lh = subjects.str.contains(r'\.lh\.', case=False, regex=True)
    is_rh = subjects.str.contains(r'\.rh\.', case=False, regex=True)

    # --- Scenario 1: Row-based 'label' or 'structure' column ---
    label_col = next((c for c in df.columns if c.lower() in ('label', 'structure')), None)
    if label_col is not None:
        labels = df[label_col].astype(str).str.replace(r'\.0$', '', regex=True)
        invalid_mask = (is_lh & labels.isin(['139', 'rh', 'right'])) | \
                       (is_rh & labels.isin(['138', 'lh', 'left']))
        df = df[~invalid_mask].copy()

    # --- Scenario 2: Wide format columns (e.g., hd95_138, hd95_139) ---
    for col in df.columns:
        if col.lower() in IGNORE_METADATA_COLS:
            continue
        # Set 139 metrics to NaN for LH images
        if any(col.endswith(suf) for suf in ['_139', '_rh', '_right']):
            df.loc[is_lh, col] = np.nan
        # Set 138 metrics to NaN for RH images
        elif any(col.endswith(suf) for suf in ['_138', '_lh', '_left']):
            df.loc[is_rh, col] = np.nan

    return df


def compute_extended_summary(folders, out_dir=None, hd_threshold=30.0):
    """
    Computes extended statistics (Mean, SD, Median, Q25, Q75, IQR) across test cases 
    for each epoch while ignoring hemisphere-mismatched evaluations and distance artifacts.
    """
    summary_rows = []

    for folder in folders:
        folder = Path(folder)
        
        # Prefer tidy CSV if available, fall back to standard metrics.csv
        csv_path = folder / 'metrics_tidy.csv'
        if not csv_path.exists():
            csv_path = folder / 'metrics.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        
        # 1. Filter out hemisphere mismatches (.lh. evaluating label 139 or .rh. evaluating label 138)
        df = filter_hemisphere_mismatches(df)

        folder_info = {
            'folder': folder.name,
            'path': str(folder)
        }

        # --- Branch A: Tidy Long Format (metric, label, subject, value) ---
        if {'metric', 'value'}.issubset(set(df.columns)):
            label_col = 'label' if 'label' in df.columns else None
            group_cols = ['metric', label_col] if label_col else ['metric']
            
            for group_keys, group_df in df.groupby(group_cols):
                metric_name = group_keys[0] if isinstance(group_keys, tuple) else group_keys
                label_val = group_keys[1] if isinstance(group_keys, tuple) else 'all'

                series = group_df['value'].dropna()

                # Filter distance artifacts (> 30mm cutoff)
                if any(hd_key in str(metric_name).lower() for hd_key in ['hd', 'hausdorff', 'distance']):
                    series = series[series <= hd_threshold]

                if series.empty:
                    continue

                mean_val = series.mean()
                std_val = series.std(ddof=1) if len(series) > 1 else 0.0
                median_val = series.median()
                q25_val = series.quantile(0.25)
                q75_val = series.quantile(0.75)
                iqr_val = q75_val - q25_val

                row = dict(folder_info)
                row.update({
                    'metric': metric_name,
                    'label': label_val,
                    'n_valid': len(series),
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'q25': q25_val,
                    'q75': q75_val,
                    'iqr': iqr_val,
                    'mean_std_str': f"{mean_val:.4f} ± {std_val:.4f}",
                    'median_iqr_str': f"{median_val:.4f} [{q25_val:.4f}, {q75_val:.4f}]"
                })
                summary_rows.append(row)

        # --- Branch B: Wide Format (each metric is a column) ---
        else:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns 
                if c.lower().strip() not in IGNORE_METADATA_COLS
            ]

            for col in numeric_cols:
                series = df[col].dropna()

                # Filter distance artifacts (> 30mm cutoff)
                if any(hd_key in col.lower() for hd_key in ['hd', 'hausdorff', 'distance']):
                    series = series[series <= hd_threshold]

                if series.empty:
                    continue

                mean_val = series.mean()
                std_val = series.std(ddof=1) if len(series) > 1 else 0.0
                median_val = series.median()
                q25_val = series.quantile(0.25)
                q75_val = series.quantile(0.75)
                iqr_val = q75_val - q25_val

                row = dict(folder_info)
                row.update({
                    'metric': col,
                    'n_valid': len(series),
                    'mean': mean_val,
                    'std': std_val,
                    'median': median_val,
                    'q25': q25_val,
                    'q75': q75_val,
                    'iqr': iqr_val,
                    'mean_std_str': f"{mean_val:.4f} ± {std_val:.4f}",
                    'median_iqr_str': f"{median_val:.4f} [{q25_val:.4f}, {q75_val:.4f}]"
                })
                summary_rows.append(row)

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
