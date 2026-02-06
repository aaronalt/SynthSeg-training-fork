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
import numpy as np
from glob import glob
import json
import datetime
import re
import shutil
import pandas as pd
from pathlib import Path
import compare_dice_batch as evaluate

# === OPTIONS ===
DELETE_TMP_PREDICTIONS = False  # Set to True to delete segmentations after evaluation (keeps CSVs)
FIELD_STRENGTH = '3T'

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
model_dir = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260205_095904'
model_files = sorted(glob(os.path.join(model_dir, '*.h5')))
# model_files = model_files[:10]
model_files = [f for f in model_files if 'dice_finetune_031' in f]
# Ground truth directories for evaluation
gt_dirs = {
    '7T': [
        Path('/home/althause/data/claustrum_gt/7T/T1_CONTROL'),
        Path('/home/althause/data/claustrum_gt/7T/T2'),
    ],
    '3T': [
        Path('/home/althause/data/claustrum_gt/3T/T1_CONTROL'),
        Path('/home/althause/data/claustrum_gt/3T/T1_VCFS'),
    ],
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

# Extract epoch number for sorting (assumes format like 'dice_finetune_005_20.h5')
def get_epoch(f):
    match = re.search(r'_(\d+)_\d+\.h5$', f)
    return int(match.group(1)) if match else 0

model_files = sorted(model_files, key=get_epoch)

print(f"Found {len(model_files)} model checkpoints to evaluate")
print(f"Models: {[os.path.basename(f) for f in model_files]}")

for path_model in model_files:
    # Paths
    exp = model_dir.split('/')[-1]
    model_name = os.path.basename(path_model).replace('.h5', '')
    model = os.path.basename(path_model)
    path_images = '/home/althause/data/training_split/test'
    path_segm = f'/home/althause/data/seg/{exp}/{model_name}'
    path_posteriors = os.path.join(path_segm, 'posteriors')
    path_resampled = os.path.join(path_segm, 'resampled')
    path_vol = os.path.join(path_segm, 'volumes.csv')
    if FIELD_STRENGTH == '7T':
        path_images = '/home/althause/data/training_split/test'
        active_gt_dirs = gt_dirs['7T']
    elif FIELD_STRENGTH == '3T':
        path_images = '/home/althause/data/training_split_3T/test'
        active_gt_dirs = gt_dirs['3T']
    else:  # 'both'
        path_images = '/home/althause/data/training_split/test_all'  # combined
        active_gt_dirs = gt_dirs['7T'] + gt_dirs['3T']
        # Extract model params
    json_params = os.path.join(model_dir, 'training_params.json')
    with open(json_params, "r") as jsonfile:
        trained_model_params = json.load(jsonfile)

    # Model and labels
    path_segmentation_labels = np.array(trained_model_params.get('segmentation_labels'))
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

    print("\n=== Prediction Configuration ===")
    print(f"Input images: {path_images}")
    print(f"Output directory: {path_segm}")
    print(f"Model: {path_model}")
    print(f"Target resolution: {target_res}")
    print(f"n_channels: {n_channels}")
    print("\nStarting prediction...\n")
    print(f'path_images: {path_images}')
    print(f'path_segm: {path_segm}')

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
            activation=activation)

    print("\nPrediction complete!")

    # === DICE EVALUATION ===
    path_subj = Path(path_segm)
    for sub in sorted(path_subj.glob('*nii.gz')):
        name = sub.stem
        match = re.search(r'(\d+).*?(lh|rh)', name, re.IGNORECASE)

        if not match:
            continue

        subject_id = match.group(1)
        hemisphere = match.group(2).lower()

        # Detect modality from filename
        modality = None
        if 't2w-cor' in name.lower() or 't2w_cor' in name.lower():
            modality = 't2w-cor'
        elif 't2w-tra' in name.lower() or 't2w_tra' in name.lower():
            modality = 't2w-tra'
        elif 't1w' in name.lower():
            modality = 't1w'

        gt = find_ground_truth(subject_id, hemisphere, modality)
        group = FIELD_STRENGTH

        if gt:
            print(f"Found {group} GT match for {subject_id} {hemisphere}: {gt.name}")

            results = evaluate.evaluate(
                sub,
                gt,
                subject_id=subject_id,
                hemi=hemisphere,
                save_resampled=False,
                save_output=True,
                output_dir=path_subj,
                group=group,
                model=path_model,
                prediction_path=path_segm
            )

            # Collect results for summary
            if results:
                results['model'] = model
                results['epoch'] = get_epoch(path_model)
                all_model_results.append(results)

    # Calculate mean dice for this model
    model_results = [r for r in all_model_results if r.get('model') == model]
    if model_results and 'dice' in model_results[0]:
        mean_dice = np.mean([r['dice'] for r in model_results])
        model_dice_scores[path_model] = mean_dice
        print(f"Mean Dice for {model}: {mean_dice:.4f}")

    # Clean up temporary outputs (delete nifti/posteriors/resampled, keep CSVs)
    if DELETE_TMP_PREDICTIONS:
        # Delete nifti files
        for f in glob(os.path.join(path_segm, '*.nii.gz')):
            os.remove(f)
        # Delete posteriors and resampled directories
        if os.path.exists(path_posteriors):
            shutil.rmtree(path_posteriors)
        if os.path.exists(path_resampled):
            shutil.rmtree(path_resampled)
        print(f"Cleaned up segmentations (kept CSVs): {path_segm}")

    # Clear GPU memory between models
    keras.backend.clear_session()
    tf.keras.backend.clear_session()
    print(f"\n{'='*50}")
    print(f"Completed evaluation for {model}")
    print(f"{'='*50}\n")

# Summary across all epochs
if all_model_results:
    df = pd.DataFrame(all_model_results)
    summary_path = os.path.join(model_dir, 'epoch_evaluation_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print("SUMMARY: Performance across epochs")
    print(f"{'='*60}")

    # Group by epoch and compute mean dice
    if 'dice' in df.columns:
        epoch_summary = df.groupby('epoch')['dice'].agg(['mean', 'std', 'count'])
        print(epoch_summary)
        best_epoch = epoch_summary['mean'].idxmax()
        print(f"\nBest epoch: {best_epoch} (Dice: {epoch_summary['mean'].max():.4f})")

    print(f"\nFull results saved to: {summary_path}")
    RERUN_TMP_PREDICTIONS = False
    # Re-run prediction for best model and keep outputs
    if DELETE_TMP_PREDICTIONS or RERUN_TMP_PREDICTIONS:
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
