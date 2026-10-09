"""
Shared helper to consolidate SynthSeg evaluation .npy metrics (dice, hausdorff,
hausdorff_95/99, mean_distance) into CSV files, without re-running prediction or evaluation.
Used by prediction.py and collect_metrics.py.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

MAXIMIZE_METRICS = ('dice', 'precision', 'recall')
MINIMIZE_METRICS = ('hausdorff', 'hausdorff_95', 'hausdorff_99', 'mean_distance')
# kept for backwards compatibility
DISTANCE_METRICS = MINIMIZE_METRICS


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
    if
    npy_files = sorted(Path(f'{path_segm}/mauri).rglob('*.npy'))
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


def _epoch_of(folder):
    """Epoch number parsed from a checkpoint folder name (first digit group, e.g.
    dice_finetune_005_20 -> 5), else None."""
    import re
    digits = re.sub('[^0-9]', ' ', Path(folder).name).split()
    return int(digits[0]) if digits else None


def summarize_per_epoch(folders, out_dir=None, eval_label_names=('138', '139')):
    """Compute per-epoch mean +/- std of every metric across the given checkpoint folders,
    print them, report the best epoch per metric, save epoch_metrics_summary.csv and
    epoch_metrics.png (mean per epoch, both labels) into out_dir (default: common parent).
    Absent-label entries (dice == 0 for the hemisphere not present in a file) are excluded
    from every metric, so distance penalties for absent labels don't pollute the averages.
    :param folders: iterable of checkpoint seg folders containing the metric .npy files.
    :param out_dir: where to write epoch_metrics_summary.csv / epoch_metrics.png.
    :param eval_label_names: row names matching the evaluation_labels used during evaluation.
    :return: the summary DataFrame, or None if nothing was found.
    """
    rows = []
    for folder in sorted(map(Path, folders), key=lambda f: (_epoch_of(f) is None, _epoch_of(f))):
        dice_file = folder / 'dice.npy'
        if not dice_file.exists():
            continue
        dice = np.load(dice_file)
        if dice.ndim != 2:
            continue
        present = dice != 0.0  # (n_labels, n_subjects) mask of present labels
        arrays = {}
        for name in MAXIMIZE_METRICS + MINIMIZE_METRICS:
            f = folder / f'{name}.npy'
            if f.exists():
                arr = np.load(f)
                if arr.shape == dice.shape:
                    arrays[name] = arr
        for name, arr in arrays.items():
            for row_idx in range(arr.shape[0]):
                values = arr[row_idx][present[row_idx]]
                if values.size:
                    rows.append({'epoch': _epoch_of(folder), 'folder': folder.name,
                                 'metric': name,
                                 'label': eval_label_names[row_idx] if row_idx < len(eval_label_names) else str(row_idx),
                                 'mean': np.mean(values), 'std': np.std(values), 'n': values.size})
    if not rows:
        print('No metric .npy files found to summarize.')
        return None

    df = pd.DataFrame(rows)
    metrics_order = [m for m in MAXIMIZE_METRICS + MINIMIZE_METRICS if m in set(df.metric)]

    print('\nPer-epoch means (present labels only):')
    for name in metrics_order:
        sub = df[df.metric == name]
        print(f'\n{name}:')
        print(sub.pivot_table(index='epoch', columns='label', values='mean', aggfunc='first')
              .to_string(float_format=lambda v: f'{v:.4f}'))

    print('\nBest epoch per metric (by mean of both labels):')
    for name in metrics_order:
        sub = df[df.metric == name].groupby('epoch')['mean'].mean()
        best = sub.idxmax() if name in MAXIMIZE_METRICS else sub.idxmin()
        print(f'  {name:<15}epoch {best} ({sub[best]:.4f})')

    if out_dir is None:
        out_dir = os.path.commonpath([str(f) for f in folders])
    out_dir = Path(out_dir)
    csv_path = out_dir / 'epoch_metrics_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nWrote per-epoch summary to {csv_path}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(metrics_order), figsize=(5 * len(metrics_order), 4), squeeze=False)
        for ax, name in zip(axes[0], metrics_order):
            for label, sub in df[df.metric == name].groupby('label'):
                sub = sub.sort_values('epoch')
                ax.errorbar(sub.epoch, sub['mean'], yerr=sub['std'], marker='o', capsize=3, label=label)
            ax.set_title(name)
            ax.set_xlabel('epoch')
            ax.legend()
        fig.tight_layout()
        png_path = out_dir / 'epoch_metrics.png'
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f'Wrote per-epoch plot to {png_path}')
    except Exception as e:
        print(f'Could not write plot: {e}')

    return df


def _read_tb_scalars(event_files, tags=('epoch_loss', 'loss', 'val_loss')):
    """Read (step, value) scalar series from TF event files, handling both TF1
    (simple_value) and TF2/Keras3 (tensor field) encodings.
    Returns (series, seen) where series is {tag: [(step, value), ...]} for the requested
    tags, and seen is {tag: count} for ALL scalar tags encountered (for diagnostics)."""
    # same import path as SynthSeg/validate.py (proven to work on the server)
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def tensor_to_float(tensor_proto):
        """Decode a scalar tensor proto across TF versions."""
        try:
            from tensorflow.python.framework.tensor_util import make_ndarray
            return float(make_ndarray(tensor_proto))
        except ImportError:
            pass
        import tensorflow as tf
        try:
            return float(tf.make_ndarray(tensor_proto))
        except AttributeError:
            pass
        # manual fallback: scalar protos store the value in typed repeated fields
        for field in ('float_val', 'double_val', 'int_val', 'int64_val'):
            values = getattr(tensor_proto, field)
            if values:
                return float(values[0])
        raise ValueError('could not decode scalar tensor proto')

    series = {}
    seen = {}
    for event_file in sorted(map(str, event_files)):
        try:
            events = summary_iterator(event_file)
            for event in events:
                for v in event.summary.value:
                    seen[v.tag] = seen.get(v.tag, 0) + 1
                    if v.tag not in tags:
                        continue
                    # HasField works regardless of proto oneof naming across TF versions
                    if v.HasField('simple_value'):
                        value = v.simple_value
                    elif v.HasField('tensor'):
                        value = tensor_to_float(v.tensor)
                    else:
                        continue
                    series.setdefault(v.tag, []).append((event.step, value))
        except Exception as e:  # e.g. tf.errors.DataLossError on a truncated last record
            print(f'[WARNING] could not read {event_file}: {type(e).__name__}: {e}')
    return series, seen


def plot_loss_curves(models_dir, out_dir=None):
    """Plot training and validation loss vs epoch from TensorBoard event files under
    models_dir (e.g. model_dir/logs/train and .../logs/validation as written by the Keras
    TensorBoard callback), in the same style as epoch_metrics.png.
    Saves loss_curves.png into out_dir (default: models_dir) and prints the final losses.
    :param models_dir: directory searched recursively for events.out.tfevents.* files.
    :param out_dir: where to write loss_curves.png.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        print(f'[WARNING] --models directory does not exist: {models_dir}')
        return
    event_files = sorted(models_dir.rglob('events.out.tfevents.*'))
    print(f'Found {len(event_files)} TensorBoard event file(s) under {models_dir}:')
    for f in event_files:
        print(f'  {f}')
    if not event_files:
        print('Directory contents (to help locate the logs):')
        for p in sorted(models_dir.iterdir()):
            print(f'  {p.name}{"/" if p.is_dir() else ""}')
        logs_dir = models_dir / 'logs'
        if logs_dir.is_dir():
            print('logs/ contents:')
            for p in sorted(logs_dir.iterdir()):
                print(f'  {p.name}{"/" if p.is_dir() else ""}')
        return

    groups = {}
    for f in event_files:
        groups.setdefault(f.parent.name, []).append(f)  # e.g. 'train', 'validation'

    try:
        curves = {}
        for name, files in groups.items():
            series, seen = _read_tb_scalars(files)
            print(f'{name}: scalar tags seen: '
                  + (', '.join(f'{t} (x{n})' for t, n in sorted(seen.items())) or 'none'))
            tag = 'epoch_loss' if 'epoch_loss' in series else ('loss' if 'loss' in series else None)
            if tag:
                pts = sorted(series[tag])
                curves[name] = (np.array([s for s, _ in pts]), np.array([v for _, v in pts]))
        if not curves:
            print(f'[WARNING] no loss scalars (epoch_loss/loss/val_loss) found in event files '
                  f'under {models_dir} -- see tags listed above.')
            return
    except ImportError:
        import traceback
        print('[WARNING] could not import tensorflow summary tools; skipping loss curves:')
        traceback.print_exc()
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (steps, values) in sorted(curves.items()):
        ax.plot(steps, values, marker='o', markersize=3, label=name)
        print(f'{name}: {len(values)} epochs, final loss {values[-1]:.4f}, '
              f'min {values.min():.4f} (epoch {int(steps[values.argmin()])})')
    ax.set_title('loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_dir) if out_dir else models_dir
    out_path = out_path / 'loss_curves.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Wrote loss curves to {out_path}')
