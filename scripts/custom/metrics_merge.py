"""
Shared helper to consolidate SynthSeg evaluation .npy metrics (dice, hausdorff,
hausdorff_95/99, mean_distance) into CSV files, without re-running prediction or evaluation.
Used by prediction.py and collect_metrics.py.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

DISTANCE_METRICS = ('hausdorff', 'hausdorff_95', 'hausdorff_99', 'mean_distance')


def merge_metrics_npy(path_segm, eval_label_names=('138', '139')):
    """Merge evaluation .npy files under path_segm into metrics.csv (wide) and metrics_tidy.csv
    (long: metric, label, subject, value).
    Arrays are (n_labels, n_subjects): rows are evaluation labels, columns are matched subjects.
    Subject names are taken from segmentation files in path_segm; if they were deleted
    (DELETE_TMP_PREDICTIONS=True) or the count mismatches, falls back to subject_0..N.
    :param path_segm: folder containing the metric .npy files (searched recursively).
    :param eval_label_names: row names matching the evaluation_labels used during evaluation.
    :return: (csv_path, tidy_path), or (None, None) if no .npy files were found.
    """
    npy_files = sorted(Path(path_segm).rglob('*.npy'))
    if not npy_files:
        print(f'No .npy files found under {path_segm}')
        return None, None

    subject_names = [p.name.replace('.nii.gz', '') for p in sorted(Path(path_segm).glob('*.nii.gz'))]
    dfs = []
    tidy_rows = []
    for npy_file in npy_files:
        array = np.load(npy_file, allow_pickle=False)
        rel_path = str(npy_file.relative_to(path_segm))

        if array.ndim == 2:
            n_labels, n_subjects = array.shape
            if len(subject_names) == n_subjects:
                columns = subject_names
            else:
                columns = [f'subject_{i}' for i in range(n_subjects)]
            # Create a row per label (138, 139)
            df = pd.DataFrame(array, columns=columns)
            df.insert(0, 'label', list(eval_label_names)[:n_labels])
            df.insert(0, 'file', rel_path)
            # Long format: one row per (metric, label, subject) value
            for label_name, row in zip(list(eval_label_names)[:n_labels], array):
                for subject, value in zip(columns, row):
                    tidy_rows.append({'metric': Path(rel_path).stem, 'label': label_name,
                                      'subject': subject, 'value': value})
        else:
            # Fallback for 1D or 3D arrays
            df = pd.DataFrame(
                {
                    'file': rel_path,
                    'index': np.arange(array.size),
                    'value': array.ravel(),
                }
            )

        dfs.append(df)

    csv_path = os.path.join(path_segm, 'metrics.csv')
    pd.concat(dfs, ignore_index=True).to_csv(csv_path, index=False)
    print(f'Combined {len(npy_files)} .npy files into {csv_path}')

    tidy_path = None
    if tidy_rows:
        tidy_path = os.path.join(path_segm, 'metrics_tidy.csv')
        pd.DataFrame(tidy_rows).to_csv(tidy_path, index=False)
        print(f'Wrote {len(tidy_rows)} rows (all metrics, long format) to {tidy_path}')

    return csv_path, tidy_path


def print_metrics_summary(folders, eval_label_names=('138', '139')):
    """Print mean +/- std of every metric, pooled across the given checkpoint folders
    (e.g. all epochs of one experiment).
    Absent-label entries (dice == 0 for the hemisphere not present in a file) are excluded
    from every metric, so distance penalties for absent labels don't pollute the averages.
    :param folders: iterable of checkpoint seg folders containing the metric .npy files.
    :param eval_label_names: row names matching the evaluation_labels used during evaluation.
    """
    dice_stack = []
    metric_stacks = {}
    for folder in folders:
        folder = Path(folder)
        dice_file = folder / 'dice.npy'
        if not dice_file.exists():
            continue
        dice = np.load(dice_file)
        if dice.ndim != 2:
            continue
        dice_stack.append(dice)
        for name in DISTANCE_METRICS:
            f = folder / f'{name}.npy'
            if f.exists():
                arr = np.load(f)
                if arr.shape == dice.shape:
                    metric_stacks.setdefault(name, []).append(arr)
    if not dice_stack:
        print('No dice.npy files found to summarize.')
        return

    dice_all = np.concatenate(dice_stack, axis=1)
    present = dice_all != 0.0  # (n_labels, n_subjects) mask of present labels
    arrays = {'dice': dice_all}
    for name, stack in metric_stacks.items():
        arrays[name] = np.concatenate(stack, axis=1)

    n_epochs = len(dice_stack)
    print(f'\nSummary over {n_epochs} epoch(s), {dice_all.shape[1]} pooled subject samples:')
    print(f'{"metric":<15}{"label":<8}{"mean":>12}{"std":>12}{"n":>6}')
    for name in ['dice'] + [m for m in DISTANCE_METRICS if m in arrays]:
        arr = arrays[name]
        for row_idx, label_name in enumerate(eval_label_names[:arr.shape[0]]):
            values = arr[row_idx][present[row_idx]]
            if values.size:
                print(f'{name:<15}{label_name:<8}{np.mean(values):>12.4f}'
                      f'{np.std(values):>12.4f}{values.size:>6}')
        # pooled over both labels
        pooled = np.concatenate([arr[r][present[r]] for r in range(arr.shape[0])])
        if pooled.size:
            print(f'{name:<15}{"all":<8}{np.mean(pooled):>12.4f}'
                  f'{np.std(pooled):>12.4f}{pooled.size:>6}')
