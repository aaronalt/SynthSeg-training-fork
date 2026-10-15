"""
Collect evaluation metrics (.npy: dice, hausdorff, hausdorff_95/99, mean_distance) into
metrics.csv (wide) and metrics_tidy.csv (long) WITHOUT re-running prediction or evaluation.

Usage:
    python collect_metrics.py [--models MODEL_DIR] [--baseline SEG_DIR] [--gt GT_DIR]
                              [--best-metric METRIC] [--no-compare] [path ...]
    python collect_metrics.py --compare REF_DIR CMP_DIR

Each path argument can be:
  - a checkpoint seg folder containing .npy files (processed directly), or
  - a parent folder, which is swept recursively for subfolders containing .npy files
    (e.g. /home/althause/data/seg/<experiment> sweeps all its checkpoint folders).
With no path arguments, sweeps the default segmentation root /home/althause/data/seg.

For each path, after collecting all checkpoints:
  1. merges metrics.csv / metrics_tidy.csv per checkpoint, per-epoch summary, extended summary;
  2. picks the best checkpoint (highest mean --best-metric, default dice);
  3. evaluates the best checkpoint's segmentations and the baseline segmentations (--baseline,
     default Mauri's model) against the same GT (--gt) with identical resampling and subject order
     (eval_seg_vs_gt.py), into <path>/comparison/{best,baseline};
  4. runs paired Wilcoxon tests + Cohen's d for every metric (baseline = reference) and writes
     <path>/comparison/comparison_summary.csv (plus one compare_<metric>.csv per metric).
--no-compare skips steps 2-4 (they are also skipped with --mauri).

--models MODEL_DIR additionally plots training/validation loss curves from the
TensorBoard event files under MODEL_DIR (e.g. the experiment's model folder containing
logs/train and logs/validation) and writes loss_curves.png there.

--compare REF_DIR CMP_DIR only compares two folders that already hold metric .npy files, then exits.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_merge import (merge_metrics_npy, summarize_per_epoch, plot_loss_curves,
                           MAXIMIZE_METRICS, MINIMIZE_METRICS)
from eval_seg_vs_gt import eval_seg_vs_gt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from SynthSeg.evaluate import compute_non_parametric_paired_test, cohens_d


DEFAULT_SEG_ROOT = '/home/althause/data/seg'
DEFAULT_BASELINE_SEG = '/home/althause/data/claustrum_gt/mauri_3t'  # Mauri's model segmentations
DEFAULT_GT = '/home/althause/data/claustrum_gt/3T/T1_all'           # manual 3T labels
COMPARISON_DIRNAME = 'comparison'
IGNORE_METADATA_COLS = {'subject', 'label', 'structure', 'id', 'unnamed: 0'}


def find_seg_folders(path):
    """Return [path] if it directly contains .npy files, else its subfolders that do
    (skipping comparison/ outputs written by this script)."""
    path = Path(path)
    if list(path.glob('*.npy')):
        return [path]
    return sorted(d for d in path.rglob('*') if d.is_dir() and list(d.glob('*.npy'))
                  and COMPARISON_DIRNAME not in d.relative_to(path).parts)


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


def compare_segmentations(folder_ref, folder_cmp, metric='dice', eval_label_names=('138', '139'),
                          alternative='two-sided', out_dir=None):
    """Paired Wilcoxon test + Cohen's d between two seg folders evaluated on the same subjects."""
    ref = np.load(Path(folder_ref) / f'{metric}.npy')
    cmp = np.load(Path(folder_cmp) / f'{metric}.npy')
    assert ref.shape == cmp.shape, f'shape mismatch {ref.shape} vs {cmp.shape} (same subjects/order?)'

    # each subject has only one hemisphere: the absent label has dice == 0 in both runs
    dice_ref = np.load(Path(folder_ref) / 'dice.npy')
    dice_cmp = np.load(Path(folder_cmp) / 'dice.npy')
    present = (dice_ref != 0) | (dice_cmp != 0)

    rows = []
    for i, name in enumerate(eval_label_names):
        x, y = ref[i, present[i]], cmp[i, present[i]]
        p = compute_non_parametric_paired_test(x, y, alternative=alternative)[0]  # 1d input -> single p
        d = cohens_d(x, y)
        rows.append({'label': name, 'n': len(x), 'mean_ref': x.mean(), 'mean_cmp': y.mean(),
                     'p_wilcoxon': p, 'cohens_d': d})

    # pooled: one value per subject (its present hemisphere)
    x = np.where(present, ref, 0).sum(axis=0)
    y = np.where(present, cmp, 0).sum(axis=0)
    rows.append({'label': 'pooled', 'n': len(x), 'mean_ref': x.mean(), 'mean_cmp': y.mean(),
                 'p_wilcoxon': compute_non_parametric_paired_test(x, y, alternative=alternative)[0],
                 'cohens_d': cohens_d(x, y)})

    df = pd.DataFrame(rows)
    print(f'\n{metric}: {folder_ref}  vs  {folder_cmp}')
    print(df.to_string(index=False))

    out_dir = Path(out_dir) if out_dir else Path(folder_cmp)
    out_path = out_dir / f'compare_{metric}.csv'
    df.insert(0, 'folder_cmp', str(folder_cmp))
    df.insert(0, 'folder_ref', str(folder_ref))
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')
    return df


ALL_METRICS = MAXIMIZE_METRICS + MINIMIZE_METRICS


def run_comparison(ref_dir, cmp_dir, out_dir=None):
    """compare_segmentations for every metric present in both folders. Writes compare_<metric>.csv
    files and comparison_summary.csv (all metrics, Bonferroni-adjusted p over the metrics compared,
    and which side is better) into out_dir (default cmp_dir). Returns the summary DataFrame or None."""
    out_dir = Path(out_dir) if out_dir else Path(cmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for metric in ALL_METRICS:
        missing = [d for d in (ref_dir, cmp_dir) if not (Path(d) / f'{metric}.npy').exists()]
        if missing:
            print(f'Skipping {metric}: no {metric}.npy in {", ".join(map(str, missing))}')
            continue
        df = compare_segmentations(ref_dir, cmp_dir, metric=metric, out_dir=out_dir)
        df.insert(2, 'metric', metric)
        tables.append(df)
    if not tables:
        for d in (ref_dir, cmp_dir):
            print(f'\n.npy files under {d}:')
            for f in sorted(Path(d).rglob('*.npy')):
                print(f'  {f.relative_to(d)}')
        return None

    summary = pd.concat(tables, ignore_index=True)
    summary['p_bonferroni'] = (summary['p_wilcoxon'] * len(tables)).clip(upper=1.0)
    higher_is_better = summary['metric'].isin(MAXIMIZE_METRICS)
    cmp_wins = np.where(higher_is_better, summary['mean_cmp'] > summary['mean_ref'],
                        summary['mean_cmp'] < summary['mean_ref'])
    summary['better'] = np.where(summary['mean_cmp'] == summary['mean_ref'], 'tie',
                                 np.where(cmp_wins, 'cmp', 'ref'))
    out_path = out_dir / 'comparison_summary.csv'
    summary.to_csv(out_path, index=False)

    pooled = summary[summary['label'] == 'pooled']
    print(f'\n=== Comparison summary (pooled; ref = {ref_dir}, cmp = {cmp_dir}) ===')
    print(pooled[['metric', 'n', 'mean_ref', 'mean_cmp', 'p_wilcoxon', 'p_bonferroni', 'cohens_d', 'better']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Wrote {out_path}')
    return summary


def pick_best_folder(folders, metric='dice'):
    """Checkpoint folder with the best mean `metric` (present labels only, averaged over labels).
    Returns (folder, score) or (None, None)."""
    scores = {}
    for folder in folders:
        metric_file, dice_file = Path(folder) / f'{metric}.npy', Path(folder) / 'dice.npy'
        if not (metric_file.exists() and dice_file.exists()):
            continue
        arr, present = np.load(metric_file), np.load(dice_file) != 0
        if arr.ndim != 2 or arr.shape != present.shape:
            continue
        per_label = [arr[i][present[i]].mean() for i in range(arr.shape[0]) if present[i].any()]
        if per_label:
            scores[str(folder)] = float(np.mean(per_label))
    if not scores:
        return None, None
    best = (min if metric in MINIMIZE_METRICS else max)(scores, key=scores.get)
    return best, scores[best]


def _subject_keys(eval_dir):
    with open(Path(eval_dir) / 'subjects.txt') as f:
        return [tuple(line.split('\t')[:2]) for line in f if line.strip()]


def compare_best_to_baseline(sweep_path, folders, baseline_seg, gt_dir, best_metric='dice'):
    """Pick the best checkpoint among folders, evaluate it and the baseline against gt_dir with identical
    resampling/ordering, then run_comparison(baseline, best). Outputs go to <sweep_path>/comparison/."""
    for name, d in (('baseline', baseline_seg), ('GT', gt_dir)):
        if not Path(d).is_dir():
            print(f'\nSkipping comparison: {name} folder {d} not found (set it with --{name.lower()}).')
            return None

    best, score = pick_best_folder(folders, best_metric)
    if best is None:
        print(f'\nSkipping comparison: no {best_metric}.npy found to pick a best checkpoint.')
        return None
    print(f'\nBest checkpoint by mean {best_metric}: {best} ({score:.4f})')

    sweep_path = Path(sweep_path)
    root = (sweep_path.parent if sweep_path in map(Path, folders) else sweep_path) / COMPARISON_DIRNAME
    best_eval, baseline_eval = root / Path(best).name, root / 'baseline'

    print(f'\n--- Evaluating best checkpoint against {gt_dir} ---')
    eval_seg_vs_gt(best, gt_dir, best_eval)
    print(f'\n--- Evaluating baseline {baseline_seg} against {gt_dir} ---')
    eval_seg_vs_gt(baseline_seg, gt_dir, baseline_eval)

    if _subject_keys(best_eval) != _subject_keys(baseline_eval):
        print(f'\nSkipping statistics: subjects differ between {best_eval}/subjects.txt and '
              f'{baseline_eval}/subjects.txt, so the paired tests would be invalid.')
        return None
    with open(root / 'selection.txt', 'w') as f:
        f.write(f'best_checkpoint\t{best}\nselected_by\tmean {best_metric} = {score:.6f}\n'
                f'baseline\t{baseline_seg}\ngt\t{gt_dir}\n')
    return run_comparison(baseline_eval, best_eval, out_dir=root)


def _pop_flag(argv, flag, n_values=1, default=None):
    """Remove `flag` and its values from argv; returns the value(s), True for n_values=0, or default."""
    if flag not in argv:
        return default
    i = argv.index(flag)
    values = argv[i + 1:i + 1 + n_values]
    del argv[i:i + 1 + n_values]
    return True if n_values == 0 else (values[0] if n_values == 1 else values)


if __name__ == '__main__':
    argv = sys.argv[1:]

    # --compare REF_DIR CMP_DIR: paired stats between two folders that already hold metric .npy files
    compare_dirs = _pop_flag(argv, '--compare', n_values=2)
    if compare_dirs:
        run_comparison(*compare_dirs)
        sys.exit(0)

    models_dir = _pop_flag(argv, '--models')
    use_mauri = _pop_flag(argv, '--mauri', n_values=0, default=False)  # sweep <checkpoint>/mauri subfolders
    baseline_seg = _pop_flag(argv, '--baseline', default=DEFAULT_BASELINE_SEG)
    gt_dir = _pop_flag(argv, '--gt', default=DEFAULT_GT)
    best_metric = _pop_flag(argv, '--best-metric', default='dice')
    do_compare = not _pop_flag(argv, '--no-compare', n_values=0, default=False) and not use_mauri

    args = argv or [DEFAULT_SEG_ROOT]
    n_written = 0
    loss_plotted = False
    for arg in args:
        raw_folders = find_seg_folders(arg)

        # Toggle between 'mauri' subfolders and original folders
        if use_mauri:
            folders = [os.path.join(f, 'mauri') for f in raw_folders if os.path.isdir(os.path.join(f, 'mauri'))]
        else:
            folders = raw_folders

        if not folders:
            target_str = '"mauri" subfolders' if use_mauri else '.npy files'
            print(f'No {target_str} found under {arg}')
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
            if do_compare:
                compare_best_to_baseline(arg, folders, baseline_seg, gt_dir, best_metric)
    if models_dir and not loss_plotted:
        plot_loss_curves(models_dir)
    print(f'\nDone: wrote metrics CSVs for {n_written} folder(s).')
