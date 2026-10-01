"""
Prediction script for custom-trained claustrum segmentation
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Cleans up logs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import tensorflow as tf
keras.backend.set_image_data_format('channels_last')
from SynthSeg.predict import predict
from SynthSeg.evaluate import evaluation
from SynthSeg.validate import validate_training, plot_validation_curves, draw_learning_curve
import numpy as np
from glob import glob
import json
import datetime
import re
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from compare_dice_batch import resample_to_target, evaluate
from claustrum_uncertainty import predict_with_tta


# === OPTIONS ===
DELETE_TMP_PREDICTIONS = True  # Set to True to delete segmentations after evaluation (keeps CSVs)
FORCE_FIELD_STRENGTH = 'both'  # Set to '3T', '7T', 'both', or None for auto-detect
RERUN_TMP_PREDICTIONS = False

# === TTA UNCERTAINTY OPTIONS ===
ENABLE_TTA_UNCERTAINTY = False       # Generate claustrum uncertainty maps via TTA
N_TTA_AUGMENTATIONS = 5            # Number of augmented predictions per image
TTA_UNCERTAINTY_TYPE = 'entropy'    # 'entropy', 'variance', 'confidence', 'mutual_information'
TTA_STRENGTH = 0.25                 # Augmentation strength (0-1). 1.0=training intensity, 0.25=gentle TTA
TTA_QUALITY_MODEL_WEIGHTS = None # '/home/althause/data/weights/quality_model_weights_features.npz'  # Ridge model (.npz) or None to skip

# Store results across all models for summary
all_model_results = []
model_dice_scores = {}  # Track mean dice per model for cleanup

# Save label arrays to files (do this once)
os.makedirs('./data', exist_ok=True)

# Full label set (must match training!)
segmentation_labels = np.array([0, 14, 15, 16, 24,
                                2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
topology_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

np.save('./data/labels_classes_priors/segmentation_labels.npy', segmentation_labels)
np.save('./data/labels_classes_priors/topology_classes.npy', topology_classes)

# Find all model checkpoints and sort by epoch number
# model_dir = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260218_203703' # /home/althause/data/logs/10000_steps_minus_bad_labels.log
# model_dir = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260218_211654' # /home/althause/data/logs/10000_steps_gaussian_noise.log
model_dir = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260923_093722_only_quality_labels_clau10x' # /home/althause/data/logs/7t_validation.log
model_files = sorted(glob(os.path.join(model_dir, '*.h5')))
model_files = model_files[19:21]
# model_files = model_files[::10]
# model_files = [f for f in model_files if 'dice_finetune_031' in f]
# Ground truth directories for evaluation
gt_dirs = {
    '7T': [
        Path('/home/althause/data/claustrum_gt/7T/T1_CONTROL'),
        Path('/home/althause/data/claustrum_gt/7T/T2'),
    ],
    '3T': [
        Path('/home/althause/data/claustrum_gt/3T/T1_CONTROL_edit_08.26'),
        Path('/home/althause/data/claustrum_gt/3T/T1_VCFS_edit_08.26'),
    ],
    'other': [
        Path('/Users/aaronalthauser/Downloads/claustrumData/high-res-manual-labels/case16_RH_LH.label.nii.gz')
    ]
}


def find_ground_truth(subject_id, hemisphere, modality=None):
    patterns = [
        f'*{subject_id}*{hemisphere}*.nii.gz',
        f'*{subject_id}*_{hemisphere}_*.nii.gz',
        f'sub-{subject_id}*{hemisphere}*.nii.gz',
    ]

    for gt_dir in active_gt_dirs:
        if not gt_dir.exists():
            continue
        for pattern in patterns:
            matches = list(gt_dir.glob(pattern))
            if matches:
                if modality:
                    for m in matches:
                        if modality in m.name.lower():
                            return m
                return matches[0]
    return None


def match_ground_truth_files(path_segmentations, ground_truth_dirs):
    """Match prediction files with ground-truth files by subject/hemisphere."""
    matches = []
    for prediction in sorted(Path(path_segmentations).glob('*.nii.gz')):
        name = prediction.name
        match = re.search(r'(\d+).*?(lh|rh)', name, re.IGNORECASE)
        if not match:
            continue
        subject_id, hemisphere = match.groups()
        gt = None
        patterns = [
            f'*{subject_id}*{hemisphere}*.nii.gz',
            f'*{subject_id}*_{hemisphere}_*.nii.gz',
            f'sub-{subject_id}*{hemisphere}*.nii.gz',
        ]
        for directory in ground_truth_dirs:
            if not directory.exists():
                continue
            for pattern in patterns:
                candidates = sorted(directory.glob(pattern))
                if candidates:
                    gt = candidates[0]
                    break
            if gt is not None:
                break
        if gt is not None:
            matches.append((str(prediction), str(gt)))
    return matches

# Extract epoch number for sorting (assumes format like 'dice_finetune_005_20.h5')
def get_epoch(f):
    match = re.search(r'_(\d+)_\d+\.h5$', f)
    return int(match.group(1)) if match else 0


#############
# Prediction
#############

model_files = sorted(model_files, key=get_epoch)

for path_model in model_files:
    # Paths
    exp = model_dir.split('/')[-1]
    model_name = os.path.basename(path_model).replace('.h5', '')
    model = os.path.basename(path_model)
    path_images = '/home/althause/data/TEST'

    # Determine field strength (manual override or auto-detect)
    if FORCE_FIELD_STRENGTH:
        field_strength = FORCE_FIELD_STRENGTH
        # print(f"MANUAL override: Using {field_strength} ground truth")
    else:
        # Auto-detect from TEST folder path or filenames
        field_strength = '3T'  # default
        if '7T' in path_images or '7t' in path_images.lower():
            field_strength = '7T'
        else:
            # Check if any test images have 7T in their filename
            test_files = glob(os.path.join(path_images, '*'))
            if any('7T' in str(f) or '7t' in str(f).lower() for f in test_files):
                field_strength = '7T'
        # print(f"Auto-detected field strength: {field_strength}")

    # Set active GT directories based on field strength
    if field_strength == 'both':
        active_gt_dirs = gt_dirs['3T'] + gt_dirs['7T']
        # print(f"Using GT directories (3T + 7T): {[str(d) for d in active_gt_dirs]}")
    else:
        active_gt_dirs = gt_dirs.get(field_strength, gt_dirs['3T'])
        #print(f"Using GT directories: {[str(d) for d in active_gt_dirs]}")
    path_segm = f'/home/althause/data/seg/{exp}/{model_name}'
    path_posteriors = os.path.join(path_segm, 'posteriors')
    path_resampled = os.path.join(path_segm, 'resampled')
    path_vol = os.path.join(path_segm, 'volumes.csv')

    # Extract model params
    json_params = os.path.join(model_dir, 'training_params.json')
    with open(json_params, "r") as jsonfile:
        trained_model_params = json.load(jsonfile)

    # Model and labels
    path_segmentation_labels = np.array(trained_model_params.get('segmentation_labels'))
    print(f"Segmentation labels: {path_segmentation_labels}")
    path_topology_classes = np.array(trained_model_params.get('generation_classes'))

    # Create output directories
    os.makedirs(path_segm, exist_ok=True)
    os.makedirs(path_posteriors, exist_ok=True)
    os.makedirs(path_resampled, exist_ok=True)

    # Parameters (must match training!)
    n_neutral_labels = trained_model_params.get('n_neutral_labels')
    cropping = trained_model_params.get('cropping')
    target_res = trained_model_params.get('target_res')
    flip = True
    sigma_smoothing = 0.5
    keep_biggest_component = False

    # Architecture (must match training!)
    n_levels = trained_model_params.get('n_levels')
    nb_conv_per_level = trained_model_params.get('nb_conv_per_level')
    conv_size = trained_model_params.get('conv_size')
    unet_feat_count = trained_model_params.get('unet_feat_count')
    activation = trained_model_params.get('activation')
    feat_multiplier = trained_model_params.get('feat_multiplier')
    n_channels = trained_model_params.get('n_channels')

    '''
    print("\n=== Prediction Configuration ===")
    print(f"Input images: {path_images}")
    print(f"Output directory: {path_segm}")
    print(f"Model: {path_model}")
    print(f"Target resolution: {target_res}")
    print(f"n_channels: {n_channels}")
    print("\nStarting prediction...\n")
    print(f'path_images: {path_images}')
    print(f'path_segm: {path_segm}')
    '''

    # gt_matches = match_ground_truth_files(path_segm, active_gt_dirs)
    # gt_all = [gt_path for _, gt_path in gt_matches]
    path_gt = '/home/althause/data/claustrum_gt/3T'

    # Delete stale evaluation arrays so old results cannot be re-merged below
    for stale_npy in sorted(Path(path_segm).rglob('*.npy')):
        stale_npy.unlink()

    # Run prediction
    
    predict(path_images,
            path_segm,
            path_model,
            path_segmentation_labels,
            n_neutral_labels=n_neutral_labels,
            path_posteriors=path_posteriors,
            path_resampled=path_resampled,
            path_volumes=path_vol,
            cropping=cropping,
            target_res=target_res,
            flip=flip,
            topology_classes=path_topology_classes,
            sigma_smoothing=sigma_smoothing,
            keep_biggest_component=keep_biggest_component,
            n_levels=n_levels,
            nb_conv_per_level=nb_conv_per_level,
            conv_size=conv_size,
            unet_feat_count=unet_feat_count,
            feat_multiplier=feat_multiplier,
            activation=activation,
            gt_folder=path_gt,
            evaluation_labels=[138,139],  # claustrum labels
            compute_distances=True,
            recompute=True)

    print("\nPrediction complete!")

    # Merge evaluation .npy files into metrics.csv
    # Arrays are (n_labels, n_subjects): rows are evaluation labels, columns are matched subjects
    eval_label_names = ['138', '139']  # matches evaluation_labels passed to predict()
    npy_files = sorted(Path(path_segm).rglob('*.npy'))
    if npy_files:
        subject_names = [p.name.replace('.nii.gz', '') for p in sorted(Path(path_segm).glob('*.nii.gz'))]
        dfs = []
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
                df.insert(0, 'label', eval_label_names[:n_labels])
                df.insert(0, 'file', rel_path)
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
    else:
        print(f'No .npy files found under {path_segm}')

    # Collect per-epoch Dice scores for plot_validation_curves (expects <dir>/epoch_*/dice.npy)
    path_dice = os.path.join(path_segm, 'dice.npy')
    if os.path.isfile(path_dice):
        epoch_dir = os.path.join(model_dir, 'validation', f'epoch_{get_epoch(path_model):03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        # Mask exact-0 scores as NaN: each single-hemisphere file scores exactly 0 for its absent
        # label, which would otherwise halve the curve. Caveat: a genuine dice of 0 (total failure)
        # is masked too. plot_validation_curves averages with np.nanmean.
        dice_scores = np.load(path_dice).astype('float64')
        dice_scores[dice_scores == 0.0] = np.nan
        np.save(os.path.join(epoch_dir, 'dice.npy'), dice_scores)
    
    '''
    #############
    ## Validation
    #############

    validate_training(
        image_dir=path_images,
        gt_dir=path_gt,
        models_dir=model_dir,
        validation_main_dir=model_dir,
        labels_segmentation=path_segmentation_labels,
        n_neutral_labels=n_neutral_labels,
        cropping=cropping,
        target_res=target_res,
        flip=flip,
        topology_classes=path_topology_classes,
        sigma_smoothing=sigma_smoothing,
        keep_biggest_component=keep_biggest_component,
        n_levels=n_levels,
        nb_conv_per_level=nb_conv_per_level,
        conv_size=conv_size,
        unet_feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier,
        activation=activation,
        recompute=False
    )
    '''
#############
# Validation curves
#############

# Experiments to compare on the validation/learning curves. Entries are experiment directory names
# resolved against the models root (the parent of model_dir), or absolute paths. The current
# experiment is always plotted first, so validation_curves.png is saved into its validation/ dir.
compare_experiments = [exp]

# plot_validation_curves expects one folder per network, with per-epoch subfolders containing dice.npy.
# dice.npy rows are [138, 139] (no background row), so average both rows explicitly via eval_indices.
models_root = os.path.dirname(model_dir)
list_validation_dirs = []
architecture_names = []
list_tb_files = []
list_tb_names = []
for compare_exp in compare_experiments:
    compare_model_dir = compare_exp if os.path.isabs(compare_exp) else os.path.join(models_root, compare_exp)
    compare_name = os.path.basename(compare_model_dir.rstrip('/'))
    compare_val_dir = os.path.join(compare_model_dir, 'validation')
    if os.path.isdir(compare_val_dir):
        list_validation_dirs.append(compare_val_dir)
        architecture_names.append(compare_name)
    else:
        print(f"[WARNING] No validation directory for experiment '{compare_name}' "
              f"({compare_val_dir}); skipping it on the validation curves.")
    exp_tb_files = sorted(glob(os.path.join(compare_model_dir, '**', 'events.out.tfevents.*'), recursive=True))
    if exp_tb_files:
        list_tb_files.append(exp_tb_files)
        list_tb_names.append(compare_name)

print(f"\nPlotting validation curves for: {architecture_names}")
if list_validation_dirs:
    plot_validation_curves(list_validation_dirs, architecture_names=architecture_names,
                           eval_indices=np.array([0, 1]), plot_legend=True)
    print(f"Validation curves saved to: {os.path.join(list_validation_dirs[0], 'validation_curves.png')}")
else:
    print("No validation curves were generated.")

# draw_learning_curve plots 1 - loss from tensorboard event files; it shows the figure but does not
# save it, so save the current figure afterwards. It draws one curve per element of
# path_tensorboard_files, so wrap each experiment's event files (sorted() order is chronological
# for tfevents files) as a single element, with matching names.
if list_tb_files:
    draw_learning_curve(list_tb_files, list_tb_names)
    plt.savefig(os.path.join(model_dir, 'learning_curve.png'), dpi=300)
    print(f"\nLearning curve saved to: {os.path.join(model_dir, 'learning_curve.png')}")
else:
    print(f"\nNo tensorboard event files found under {models_root}; skipping learning curve.")


'''
# Summary across all epochs
if all_model_results:
    df = pd.DataFrame(all_model_results)
    summary_path = os.path.join(model_dir, 'epoch_evaluation_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print("SUMMARY: Performance across epochs")
    print(f"{'='*60}")

    # Group by epoch and compute mean metrics
    if 'dice' in df.columns:
        agg_cols = {'dice': ['mean', 'std']}
        for col in ['precision', 'recall', 'iou']:
            if col in df.columns:
                agg_cols[col] = ['mean', 'std']
        epoch_summary = df.groupby('epoch').agg(agg_cols)
        epoch_summary.columns = ['_'.join(c) for c in epoch_summary.columns]

        # Composite score: equal weight of dice, precision, recall, iou
        metric_means = [c for c in epoch_summary.columns if c.endswith('_mean')]
        epoch_summary['composite'] = epoch_summary[metric_means].mean(axis=1)

        has_prec = 'precision_mean' in epoch_summary.columns
        has_rec  = 'recall_mean'    in epoch_summary.columns
        has_iou  = 'iou_mean'       in epoch_summary.columns

        header = f"{'Epoch':>6}  {'Dice':>7} ±{'':>5}  {'Prec':>7} ±{'':>5}  {'Recall':>7} ±{'':>5}  {'IoU':>7} ±{'':>5}  {'Composite':>9}"
        print(f"\n{header}")
        print("-" * len(header))
        for epoch, row in epoch_summary.iterrows():
            p = f"{row['precision_mean']:7.4f} {row['precision_std']:5.4f}" if has_prec else f"{'N/A':>7}  {'':>5}"
            r = f"{row['recall_mean']:7.4f} {row['recall_std']:5.4f}"       if has_rec  else f"{'N/A':>7}  {'':>5}"
            i = f"{row['iou_mean']:7.4f} {row['iou_std']:5.4f}"             if has_iou  else f"{'N/A':>7}  {'':>5}"
            print(f"{epoch:>6}  {row['dice_mean']:7.4f} {row['dice_std']:5.4f}  {p}  {r}  {i}  {row['composite']:9.4f}")

        best_by_dice      = epoch_summary['dice_mean'].idxmax()
        best_by_composite = epoch_summary['composite'].idxmax()
        print(f"\nBest epoch by Dice:      {best_by_dice} (Dice={epoch_summary.loc[best_by_dice,'dice_mean']:.4f})")
        print(f"Best epoch by Composite: {best_by_composite} (Composite={epoch_summary.loc[best_by_composite,'composite']:.4f})")

    print(f"\nFull results saved to: {summary_path}")
    # Re-run prediction for best model and keep outputs
    if RERUN_TMP_PREDICTIONS:
        if model_dice_scores:
            best_model_path = max(model_dice_scores, key=model_dice_scores.get)
            best_model = os.path.basename(best_model_path)
            print(f"\n{'='*60}")
            print(f"Re-running prediction for BEST model: {best_model}")
            print(f"{'='*60}\n")

            # Final prediction paths (these will be kept)
            path_segm_final = f'/home/althause/data/seg/{exp}/BEST_{best_model.replace(".h5", "")}'
            path_posteriors_final = os.path.join(path_segm_final, 'posteriors')
            path_resampled_final = os.path.join(path_segm_final, 'resampled')
            path_vol_final = os.path.join(path_segm_final, 'volumes.csv')

            os.makedirs(path_segm_final, exist_ok=True)
            os.makedirs(path_posteriors_final, exist_ok=True)
            os.makedirs(path_resampled_final, exist_ok=True)

            predict(path_images,
                    path_segm_final,
                    best_model_path,
                    path_segmentation_labels,
                    n_neutral_labels=n_neutral_labels,
                    path_posteriors=path_posteriors_final,
                    path_resampled=path_resampled_final,
                    path_volumes=path_vol_final,
                    cropping=cropping,
                    target_res=target_res,
                    flip=flip,
                    topology_classes=path_topology_classes,
                    sigma_smoothing=sigma_smoothing,
                    keep_biggest_component=keep_biggest_component,
                    n_levels=n_levels,
                    nb_conv_per_level=nb_conv_per_level,
                    conv_size=conv_size,
                    unet_feat_count=unet_feat_count,
                    feat_multiplier=feat_multiplier,
                    activation=activation)

            print(f"\nBest model predictions saved to: {path_segm_final}")
'''
