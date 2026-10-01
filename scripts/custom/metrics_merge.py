"""
Shared helper to consolidate SynthSeg evaluation .npy metrics (dice, hausdorff,
hausdorff_95/99, mean_distance) into CSV files, without re-running prediction or evaluation.
Used by prediction.py and collect_metrics.py.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path


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
