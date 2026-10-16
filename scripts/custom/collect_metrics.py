"""
Collect evaluation metrics (.npy: dice, hausdorff, hausdorff_95/99, mean_distance) into
metrics.csv (wide) and metrics_tidy.csv (long) WITHOUT re-running prediction or evaluation.

Usage:
    python collect_metrics.py --models MODEL_DIR [--baseline SEG_DIR] [--gt GT_DIR] [--min-epoch N]
                              [--seg-7t DIR] [--gt-7t DIR] [--baseline-7t DIR] [--no-sensitivity]
                              [--select val_loss|test] [--best-metric METRIC] [--no-compare] [path ...]
    python collect_metrics.py --compare REF_DIR CMP_DIR

Each path argument can be:
  - a checkpoint seg folder containing .npy files (processed directly), or
  - a parent folder, which is swept recursively for subfolders containing .npy files
    (e.g. /home/althause/data/seg/<experiment> sweeps all its checkpoint folders).
With no path arguments, sweeps the default segmentation root /home/althause/data/seg.

For each path (the 3T test predictions, prediction.py SPLIT='test3t'), after collecting all checkpoints:
  1. merges metrics.csv / metrics_tidy.csv per checkpoint, per-epoch summary, extended summary;
  2. selects the final checkpoint WITHOUT real data: lowest synthetic validation loss (VAL_2 label maps, the
     quantity EarlyStopping monitors) in the TensorBoard logs under --models, among epochs >= --min-epoch
     (default 40, after the HD95 weight warm-up). --select test instead takes the best mean --best-metric
     on the test results (biased, for exploration only);
  3. 3T: evaluates the selected checkpoint and the baseline (--baseline, default Mauri's model) against
     --gt on the same cases and writes paired Wilcoxon + Cohen's d for every metric to
     <path>/comparison/comparison_summary.csv (plus compare_<metric>.csv);
  4. 3T sensitivity: the same comparison (pooled Dice, mean distance) at every checkpoint with epoch
     >= --min-epoch, in <path>/comparison/sensitivity_3t.csv (--no-sensitivity skips it);
  5. 7T: if the selected checkpoint was predicted on the 7T scans (prediction.py SPLIT='test7t', under
     --seg-7t, default /home/althause/data/seg_7t/<experiment name>), evaluates it against --gt-7t and
     writes descriptive results to <path>/comparison_7t/; with --baseline-7t also compares with the
     baseline's 7T segmentations.
--no-compare skips steps 2-5 (they are also skipped with --mauri).

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

from metrics_merge import (merge_metrics_npy, summarize_per_epoch, plot_loss_curves, _read_tb_scalars,
                           _epoch_of, MAXIMIZE_METRICS, MINIMIZE_METRICS)
from eval_seg_vs_gt import eval_seg_vs_gt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from SynthSeg.evaluate import compute_non_parametric_paired_test, cohens_d


DEFAULT_SEG_ROOT = '/home/althause/data/seg'
DEFAULT_BASELINE_SEG = '/home/althause/data/claustrum_gt/mauri_3t'  # Mauri's model segmentations
DEFAULT_GT = '/home/althause/data/claustrum_gt/3T/T1_all'           # manual 3T labels
DEFAULT_SEG_7T_ROOT = '/home/althause/data/seg_7t'  # prediction.py SPLIT='test7t' output: <root>/<experiment>/<ckpt>
DEFAULT_GT_7T = '/home/althause/data/claustrum_gt/7T'  # manual 7T labels
COMPARISON_DIRNAME = 'comparison'
IGNORE_METADATA_COLS = {'subject', 'label', 'structure', 'id', 'unnamed: 0'}


def find_seg_folders(path):
    """Return [path] if it directly contains .npy files, else its subfolders that do
    (skipping comparison*/ outputs written by this script)."""
    path = Path(path)
    if list(path.glob('*.npy')):
        return [path]
    return sorted(d for d in path.rglob('*') if d.is_dir() and list(d.glob('*.npy'))
                  and not any(part.startswith(COMPARISON_DIRNAME) for part in d.relative_to(path).parts))


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


def _paired_p(x, y, alternative='two-sided', what=''):
    """Wilcoxon p-value via compute_non_parametric_paired_test. NaN (with a warning) when every paired
    difference is zero, which scipy cannot test and which for two different models usually means the
    arrays are wrong (e.g. unfilled zeros), and NaN for fewer than 2 pairs."""
    if len(x) < 2:
        return np.nan
    if np.all(x == y):
        print(f'WARNING: {what}: identical values for all {len(x)} pairs ({np.unique(x)[:5]}...); '
              f'p set to NaN. Check the .npy files, this is unlikely for two different segmentations.')
        return np.nan
    return compute_non_parametric_paired_test(x, y, alternative=alternative)[0]  # 1d input -> single p


def _compare_table(folder_ref, folder_cmp, metric='dice', eval_label_names=('138', '139'),
                   alternative='two-sided'):
    """Per-label and pooled paired Wilcoxon + Cohen's d between two folders evaluated on the same cases
    (same column order), as a DataFrame. Nothing is printed or written."""
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
        p = _paired_p(x, y, alternative, f'{metric} label {name}')
        d = cohens_d(x, y)
        rows.append({'label': name, 'n': len(x), 'mean_ref': x.mean(), 'mean_cmp': y.mean(),
                     'p_wilcoxon': p, 'cohens_d': d, 'n_identical': int(np.sum(x == y))})

    # pooled: one value per subject (its present hemisphere)
    x = np.where(present, ref, 0).sum(axis=0)
    y = np.where(present, cmp, 0).sum(axis=0)
    rows.append({'label': 'pooled', 'n': len(x), 'mean_ref': x.mean(), 'mean_cmp': y.mean(),
                 'p_wilcoxon': _paired_p(x, y, alternative, f'{metric} pooled'),
                 'cohens_d': cohens_d(x, y), 'n_identical': int(np.sum(x == y))})
    return pd.DataFrame(rows)


def compare_segmentations(folder_ref, folder_cmp, metric='dice', eval_label_names=('138', '139'),
                          alternative='two-sided', out_dir=None):
    """Paired Wilcoxon test + Cohen's d between two seg folders evaluated on the same subjects; prints the
    table and writes compare_<metric>.csv into out_dir (default folder_cmp)."""
    df = _compare_table(folder_ref, folder_cmp, metric, eval_label_names, alternative)
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


VAL_LOSS_TAGS = ('epoch_dice_loss', 'epoch_loss', 'val_dice_loss', 'val_loss', 'loss')


def pick_best_by_val_loss(folders, models_dir, min_epoch=0):
    """Checkpoint folder (epoch >= min_epoch) with the lowest validation loss in the TensorBoard logs under
    models_dir (synthetic images generated from the VAL label maps, the quantity EarlyStopping monitors).
    min_epoch skips epochs whose loss is not comparable (e.g. during the HD95 weight warm-up).
    Keras checkpoint names use 1-based epochs ({epoch:03d}), TensorBoard steps are 0-based.
    Returns (folder, loss, tag) or (None, None, reason)."""
    if not models_dir or not Path(models_dir).is_dir():
        return None, None, f'--models directory not given or not found ({models_dir})'
    event_files = [f for f in Path(models_dir).rglob('events.out.tfevents.*') if 'val' in f.parent.name.lower()]
    if not event_files:
        return None, None, f'no validation TensorBoard logs (logs/validation/) under {models_dir}'
    series, seen = _read_tb_scalars(event_files, tags=VAL_LOSS_TAGS)
    tag = next((t for t in VAL_LOSS_TAGS if t in series), None)
    if tag is None:
        return None, None, f'no validation loss tag among {VAL_LOSS_TAGS}; tags seen: {sorted(seen)}'
    loss_per_epoch = {step + 1: value for step, value in sorted(series[tag])  # resumed runs: last value wins
                      if step + 1 >= min_epoch}
    if not loss_per_epoch:
        return None, None, f'no logged validation loss at epoch >= {min_epoch}'
    candidates = {str(f): loss_per_epoch[_epoch_of(f)] for f in folders if _epoch_of(f) in loss_per_epoch}
    if not candidates:
        return None, None, (f'no checkpoint folder epoch matches the logged epochs '
                            f'{min(loss_per_epoch)}-{max(loss_per_epoch)} (>= min epoch {min_epoch})')
    best = min(candidates, key=candidates.get)
    print(f'\nValidation loss ({tag}) per predicted checkpoint, epochs >= {min_epoch}:')
    for f in sorted(candidates, key=_epoch_of):
        print(f'  epoch {_epoch_of(f):4d}  {candidates[f]:.4f}{"  <- best" if f == best else ""}')
    overall = min(loss_per_epoch, key=loss_per_epoch.get)
    if loss_per_epoch[overall] < candidates[best]:
        print(f'WARNING: the lowest logged validation loss (epochs >= {min_epoch}) is at epoch {overall} '
              f'({loss_per_epoch[overall]:.4f}), which has no prediction folder; run prediction for it.')
    return best, candidates[best], tag


def select_checkpoint(folders, select='val_loss', models_dir=None, best_metric='dice', min_epoch=40):
    """Returns (best checkpoint folder, description) or (None, reason)."""
    if select == 'val_loss':
        best, score, tag = pick_best_by_val_loss(folders, models_dir, min_epoch)
        if best is None:
            return None, (f'cannot select on validation loss: {tag}. Pass --models MODEL_DIR, or --select test '
                          f'to select on the test set (biased)')
        return best, f'validation loss {tag} = {score:.6f}, epochs >= {min_epoch} (TensorBoard logs under {models_dir})'
    best, score = pick_best_folder(folders, best_metric)
    if best is None:
        return None, f'no {best_metric}.npy found to pick a best checkpoint'
    print('\nWARNING: selecting the checkpoint on the test set; the comparison below is optimistic.')
    return best, f'TEST-SET mean {best_metric} = {score:.6f} (biased: same subjects as the comparison)'


def _gt_cases(eval_dir):
    """GT relative paths (subjects.txt column 3) in evaluation order: the pairing key of two evaluations."""
    with open(Path(eval_dir) / 'subjects.txt') as f:
        return [line.split('\t')[2] for line in f if line.strip()]


def eval_pair(ref_seg, cmp_seg, gt_dir, ref_out, cmp_out):
    """Evaluate two segmentation folders against gt_dir, restricted to the GT cases both have, so the
    metric arrays are paired column by column. Returns the number of shared cases."""
    eval_seg_vs_gt(ref_seg, gt_dir, ref_out)
    eval_seg_vs_gt(cmp_seg, gt_dir, cmp_out)
    ref_cases, cmp_cases = _gt_cases(ref_out), _gt_cases(cmp_out)
    if ref_cases != cmp_cases:
        shared = sorted(set(ref_cases) & set(cmp_cases))
        print(f'\nWARNING: {ref_seg} covers {len(ref_cases)} GT cases and {cmp_seg} covers {len(cmp_cases)}; '
              f'restricting both to the {len(shared)} shared cases.')
        if not shared:
            return 0
        eval_seg_vs_gt(ref_seg, gt_dir, ref_out, gt_subset=shared)
        eval_seg_vs_gt(cmp_seg, gt_dir, cmp_out, gt_subset=shared)
    return len(_gt_cases(ref_out))


def sensitivity_3t(folders, baseline_eval, gt_dir, root, best, min_epoch=40):
    """Your model vs the baseline at every checkpoint with epoch >= min_epoch (pooled Dice and mean distance,
    paired Wilcoxon), to show the 3T conclusion does not depend on the selected checkpoint.
    Writes root/sensitivity_3t.csv."""
    cases = _gt_cases(baseline_eval)
    rows = []
    for folder in sorted((f for f in folders if (_epoch_of(f) or -1) >= min_epoch), key=_epoch_of):
        out = Path(root) / 'checkpoints' / Path(folder).name
        print(f'\n--- Sensitivity: evaluating {Path(folder).name} ---')
        eval_seg_vs_gt(folder, gt_dir, out, gt_subset=cases)
        if _gt_cases(out) != cases:
            print(f'Skipping {Path(folder).name}: does not cover the same GT cases as the baseline.')
            continue
        row = {'checkpoint': Path(folder).name, 'epoch': _epoch_of(folder), 'selected': str(folder) == str(best)}
        for metric in ('dice', 'mean_distance'):
            pooled = _compare_table(baseline_eval, out, metric).iloc[-1]
            row.update({f'{metric}_baseline': pooled['mean_ref'], f'{metric}_model': pooled['mean_cmp'],
                        f'{metric}_p': pooled['p_wilcoxon']})
        rows.append(row)
    if not rows:
        print(f'\nNo checkpoints with epoch >= {min_epoch} for the sensitivity table.')
        return None
    df = pd.DataFrame(rows)
    df['dice_model_better'] = df['dice_model'] > df['dice_baseline']
    df['mean_distance_model_better'] = df['mean_distance_model'] < df['mean_distance_baseline']
    out_path = Path(root) / 'sensitivity_3t.csv'
    df.to_csv(out_path, index=False)
    print(f'\n=== 3T sensitivity: model vs baseline at every checkpoint (epoch >= {min_epoch}) ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    n = len(df)
    print(f'Model better on Dice at {df.dice_model_better.sum()}/{n} checkpoints '
          f'({(df.dice_p < 0.05).sum()} with p < 0.05); better on mean distance at '
          f'{df.mean_distance_model_better.sum()}/{n} ({(df.mean_distance_p < 0.05).sum()} with p < 0.05).')
    print(f'Wrote {out_path}')
    return df


def summarize_7t(eval_dir, out_dir):
    """Descriptive 7T results: one value per image (its present hemisphere), averaged per subject over
    images (contrasts/sessions), then mean, SD, median and IQR across subjects. Writes results_7t_images.csv
    and results_7t_summary.csv into out_dir."""
    with open(Path(eval_dir) / 'subjects.txt') as f:
        cases = [line.rstrip('\n').split('\t') for line in f if line.strip()]
    present = np.load(Path(eval_dir) / 'dice.npy') != 0
    images = pd.DataFrame({'subject': [c[0] for c in cases], 'hemi': [c[1] for c in cases],
                           'gt': [c[2] for c in cases]})
    for metric in MAXIMIZE_METRICS + MINIMIZE_METRICS:
        path = Path(eval_dir) / f'{metric}.npy'
        if path.exists():
            images[metric] = np.where(present, np.load(path), 0).sum(axis=0)
    metrics = [m for m in MAXIMIZE_METRICS + MINIMIZE_METRICS if m in images]
    per_subject = images.groupby('subject')[metrics].mean()
    summary = pd.DataFrame({'n_subjects': per_subject.count(), 'n_images': images[metrics].count(),
                            'mean': per_subject.mean(), 'std': per_subject.std(ddof=1),
                            'median': per_subject.median(), 'q25': per_subject.quantile(0.25),
                            'q75': per_subject.quantile(0.75)}).rename_axis('metric').reset_index()
    images.to_csv(Path(out_dir) / 'results_7t_images.csv', index=False)
    summary.to_csv(Path(out_dir) / 'results_7t_summary.csv', index=False)
    print(f'\n=== 7T result ({len(per_subject)} subjects, {len(images)} images; per-subject means) ===')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Wrote {Path(out_dir) / "results_7t_summary.csv"}')
    return summary


def result_7t(best, seg_7t, gt_7t, root_7t, baseline_7t=None):
    """Evaluate the selected checkpoint's 7T predictions (prediction.py SPLIT='test7t') against the 7T GT,
    and compare with the baseline's 7T segmentations if given."""
    name = Path(best).name
    best_7t = Path(seg_7t) / name
    if not best_7t.is_dir():
        print(f"\n7T: no predictions for {name} under {seg_7t}; run prediction.py with SPLIT='test7t', "
              f"CHECKPOINTS=['{name}'], then re-run this script.")
        return None
    if not Path(gt_7t).is_dir():
        print(f'\n7T: GT folder {gt_7t} not found (set it with --gt-7t).')
        return None
    root_7t = Path(root_7t)
    root_7t.mkdir(parents=True, exist_ok=True)
    print(f'\n--- 7T: evaluating {best_7t} against {gt_7t} ---')
    eval_seg_vs_gt(best_7t, gt_7t, root_7t / name)
    summary = summarize_7t(root_7t / name, root_7t)
    if baseline_7t:
        if not Path(baseline_7t).is_dir():
            print(f'\n7T: baseline folder {baseline_7t} not found; skipping the 7T comparison.')
        elif eval_pair(baseline_7t, best_7t, gt_7t, root_7t / 'baseline', root_7t / name):
            print('\nNOTE: 7T tests below treat every image (hemisphere x contrast/session) as independent, '
                  'although images of the same subject are not. With 4 subjects a subject-level Wilcoxon '
                  'cannot go below p = 0.125, so report the 7T comparison as descriptive.')
            run_comparison(root_7t / 'baseline', root_7t / name, out_dir=root_7t)
    return summary


def compare_best_to_baseline(sweep_path, folders, baseline_seg, gt_dir, best_metric='dice', select='val_loss',
                             models_dir=None, min_epoch=40, sensitivity=True,
                             seg_7t=None, gt_7t=None, baseline_7t=None):
    """Select the final checkpoint (select='val_loss': lowest synthetic validation loss at epoch >= min_epoch,
    no real data used; select='test': best mean best_metric on the test results, biased), then:
    3T comparison with the baseline (<sweep>/comparison/), the 3T sensitivity table across checkpoints, and
    the 7T result of the selected checkpoint (<sweep>/comparison_7t/) when its 7T predictions exist."""
    for name, d in (('baseline', baseline_seg), ('GT', gt_dir)):
        if not Path(d).is_dir():
            print(f'\nSkipping comparison: {name} folder {d} not found (set it with --{name.lower()}).')
            return None

    best, selected_by = select_checkpoint(folders, select, models_dir, best_metric, min_epoch)
    if best is None:
        print(f'\nSkipping comparison: {selected_by}.')
        return None
    print(f'\nSelected checkpoint: {best}\n  selected by {selected_by}')

    sweep_path = Path(sweep_path)
    sweep_root = sweep_path.parent if sweep_path in map(Path, folders) else sweep_path
    root = sweep_root / COMPARISON_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'selection.txt', 'w') as f:
        f.write(f'selected_checkpoint\t{best}\nselected_by\t{selected_by}\n'
                f'baseline\t{baseline_seg}\ngt\t{gt_dir}\n')

    best_eval, baseline_eval = root / Path(best).name, root / 'baseline'
    print(f'\n--- 3T: evaluating baseline {baseline_seg} and {Path(best).name} against {gt_dir} ---')
    if not eval_pair(baseline_seg, best, gt_dir, baseline_eval, best_eval):
        print('\nSkipping statistics: the baseline and the selected checkpoint share no GT cases.')
        return None
    summary = run_comparison(baseline_eval, best_eval, out_dir=root)

    if sensitivity:
        sensitivity_3t(folders, baseline_eval, gt_dir, root, best, min_epoch)
    if seg_7t:
        result_7t(best, seg_7t, gt_7t, sweep_root / f'{COMPARISON_DIRNAME}_7t', baseline_7t)
    return summary


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
    best_metric = _pop_flag(argv, '--best-metric', default='dice')  # only used with --select test
    select = _pop_flag(argv, '--select', default='val_loss')
    if select not in ('val_loss', 'test'):
        sys.exit(f'--select must be val_loss or test, got {select}')
    min_epoch = int(_pop_flag(argv, '--min-epoch', default=40))
    seg_7t_arg = _pop_flag(argv, '--seg-7t')
    gt_7t = _pop_flag(argv, '--gt-7t', default=DEFAULT_GT_7T)
    baseline_7t = _pop_flag(argv, '--baseline-7t')
    sensitivity = not _pop_flag(argv, '--no-sensitivity', n_values=0, default=False)
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
                # 7T predictions of the same experiment: --seg-7t, else <DEFAULT_SEG_7T_ROOT>/<experiment name>
                seg_7t = seg_7t_arg or str(Path(DEFAULT_SEG_7T_ROOT) / Path(arg).resolve().name)
                compare_best_to_baseline(arg, folders, baseline_seg, gt_dir, best_metric, select, models_dir,
                                         min_epoch, sensitivity, seg_7t, gt_7t, baseline_7t)
    if models_dir and not loss_plotted:
        plot_loss_curves(models_dir)
    print(f'\nDone: wrote metrics CSVs for {n_written} folder(s).')
