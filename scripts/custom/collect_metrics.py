"""
Collect evaluation metrics (.npy: dice, hausdorff, hausdorff_95/99, mean_distance) into
metrics.csv (wide) and metrics_tidy.csv (long) WITHOUT re-running prediction or evaluation.

Usage:
    python collect_metrics.py [path ...]

Each argument can be:
  - a checkpoint seg folder containing .npy files (processed directly), or
  - a parent folder, which is swept recursively for subfolders containing .npy files
    (e.g. /home/althause/data/seg/<experiment> sweeps all its checkpoint folders).
With no arguments, sweeps the default segmentation root /home/althause/data/seg.

If segmentations were deleted (DELETE_TMP_PREDICTIONS=True), subject columns fall back to
subject_0..N (the .npy files alone don't record file names).
"""
import sys
from pathlib import Path

from metrics_merge import merge_metrics_npy

DEFAULT_SEG_ROOT = '/home/althause/data/seg'


def find_seg_folders(path):
    """Return [path] if it directly contains .npy files, else its subfolders that do."""
    path = Path(path)
    if list(path.glob('*.npy')):
        return [path]
    return sorted(d for d in path.rglob('*') if d.is_dir() and list(d.glob('*.npy')))


if __name__ == '__main__':
    args = sys.argv[1:] or [DEFAULT_SEG_ROOT]
    n_written = 0
    for arg in args:
        folders = find_seg_folders(arg)
        if not folders:
            print(f'No .npy files found under {arg}')
        for folder in folders:
            print(f'\n=== {folder} ===')
            csv_path, _ = merge_metrics_npy(folder)
            n_written += csv_path is not None
    print(f'\nDone: wrote metrics CSVs for {n_written} folder(s).')
