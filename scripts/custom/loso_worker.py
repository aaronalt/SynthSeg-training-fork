"""
LOSO worker: trains on N-1 subjects, predicts on held-out subject, evaluates.
Called by loso_label_quality.py — not intended to be run directly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python loso_worker.py \
        --held_out sub-5166 \
        --labels_dir /path/to/all_labels \
        --gt_dir /path/to/ground_truth \
        --output_dir /path/to/loso_output \
        --checkpoint /path/to/pretrained_weights.h5 \
        --fold_dir /path/to/loso_output/fold_sub-5166
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import re
import json
import shutil
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from pathlib import Path

from training_feature_extraction import training
from SynthSeg.predict import predict
import compare_dice_batch as evaluate


# === Label configuration (must match your training setup) ===
GENERATION_LABELS = np.array([0, 14, 15, 16, 24,
                               2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                               41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
SEGMENTATION_LABELS = GENERATION_LABELS.copy()
N_NEUTRAL_LABELS = 5
GENERATION_CLASSES = np.array([0, 1, 2, 3, 4,
                                5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Claustrum label weights
LABEL_WEIGHTS = np.ones(len(SEGMENTATION_LABELS))
for lbl in [138, 139]:
    LABEL_WEIGHTS[np.where(SEGMENTATION_LABELS == lbl)[0][0]] = 10.0


def find_gt(gt_dirs, subject_id, hemisphere):
    patterns = [
        f'*{subject_id}*{hemisphere}*.nii.gz',
        f'sub-{subject_id}*{hemisphere}*.nii.gz',
        f'*{subject_id}*.nii.gz',
    ]
    for gt_dir in gt_dirs:
        p = Path(gt_dir)
        if not p.exists():
            continue
        for pat in patterns:
            matches = list(p.glob(pat))
            if matches:
                return matches[0]
    return None


def run_fold(held_out, labels_dir, gt_dirs, output_dir, checkpoint,
             epochs, steps_per_epoch, intensity_priors_dir):

    fold_dir = os.path.join(output_dir, f'fold_{held_out}')
    train_labels_dir = os.path.join(fold_dir, 'train_labels')
    model_dir = os.path.join(fold_dir, 'model')
    seg_dir = os.path.join(fold_dir, 'segmentations')
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    results_path = os.path.join(fold_dir, 'results.json')
    if os.path.exists(results_path):
        print(f"[{held_out}] Already done, skipping.")
        with open(results_path) as f:
            return json.load(f)

    # === Symlink all labels except held-out subject ===
    all_labels = sorted(glob(os.path.join(labels_dir, '*.nii.gz')))
    n_train = 0
    for lf in all_labels:
        name = os.path.basename(lf)
        # Skip if this file belongs to held-out subject
        if held_out.replace('sub-', '') in name or held_out in name:
            continue
        dest = os.path.join(train_labels_dir, name)
        if not os.path.exists(dest):
            os.symlink(os.path.abspath(lf), dest)
        n_train += 1

    print(f"\n[{held_out}] Training on {n_train} subjects (held out: {held_out})")

    # === Load intensity priors ===
    means = np.load(os.path.join(intensity_priors_dir, 'prior_means.npy'))
    stds  = np.load(os.path.join(intensity_priors_dir, 'prior_stds.npy'))

    # === Quick training (few epochs, start from pretrained checkpoint) ===
    training(
        train_labels_dir,
        model_dir,
        generation_labels=GENERATION_LABELS,
        segmentation_labels=SEGMENTATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        generation_classes=GENERATION_CLASSES,
        batchsize=1,
        n_channels=1,
        target_res=0.50,
        output_shape=192,
        prior_distributions='normal',
        prior_means=means,
        prior_stds=stds,
        flipping=True,
        scaling_bounds=0.2,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=2.0,
        randomise_res=True,
        max_res_iso=1.5,
        max_res_aniso=2.0,
        bias_field_std=0.3,
        noise_std=100,
        n_levels=5,
        nb_conv_per_level=2,
        conv_size=3,
        unet_feat_count=24,
        feat_multiplier=2,
        activation='elu',
        lr=1e-4,
        wl2_epochs=0,
        dice_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint=checkpoint,
        skip_pretrain=False,
        finetune=False,
        label_weights=LABEL_WEIGHTS,
    )

    # === Find the final checkpoint ===
    fold_models = sorted(glob(os.path.join(model_dir, 'dice_finetune_*.h5')))
    if not fold_models:
        print(f"[{held_out}] ERROR: No model checkpoints found after training.")
        return None
    final_model = fold_models[-1]
    print(f"[{held_out}] Using model: {os.path.basename(final_model)}")

    # === Find held-out subject's label file(s) ===
    held_out_id = held_out.replace('sub-', '')
    held_out_labels = [f for f in all_labels
                       if held_out_id in os.path.basename(f) or held_out in os.path.basename(f)]
    if not held_out_labels:
        print(f"[{held_out}] ERROR: No label files found for held-out subject.")
        return None

    # === Predict on held-out subject's label maps ===
    # Use the label maps as input images (they will be used to generate synthetic images)
    # We need actual MRI images for prediction — find them from gt_dirs or a paired images dir
    held_out_images_dir = os.path.join(fold_dir, 'held_out_images')
    os.makedirs(held_out_images_dir, exist_ok=True)

    # Symlink held-out label files to use as prediction inputs
    # (prediction uses label maps directly through the synthesis pipeline)
    for lf in held_out_labels:
        dest = os.path.join(held_out_images_dir, os.path.basename(lf))
        if not os.path.exists(dest):
            os.symlink(os.path.abspath(lf), dest)

    predict(
        held_out_images_dir,
        seg_dir,
        final_model,
        SEGMENTATION_LABELS,
        n_neutral_labels=N_NEUTRAL_LABELS,
        target_res=0.50,
        flip=True,
        sigma_smoothing=0.5,
        keep_biggest_component=False,
        n_levels=5,
        nb_conv_per_level=2,
        conv_size=3,
        unet_feat_count=24,
        feat_multiplier=2,
        activation='elu',
    )

    # === Evaluate against GT ===
    fold_results = []
    seg_files = sorted(glob(os.path.join(seg_dir, '*.nii.gz')))

    for seg_path in seg_files:
        name = Path(seg_path).stem.replace('.nii', '')
        match = re.search(r'(\d+).*?(lh|rh)', name, re.IGNORECASE)
        if not match:
            continue
        subject_id = match.group(1)
        hemisphere = match.group(2).lower()

        gt_path = find_gt(gt_dirs, subject_id, hemisphere)
        if gt_path is None:
            print(f"  [{held_out}] No GT found for {subject_id} {hemisphere}")
            continue

        res = evaluate.evaluate(
            seg_path, gt_path,
            subject_id=subject_id,
            hemi=hemisphere,
            save_output=False,
        )
        if res:
            res['held_out_subject'] = held_out
            fold_results.append(res)
            print(f"  [{held_out}] {subject_id} {hemisphere}: "
                  f"Dice={res['dice']:.4f}  Prec={res['precision']:.4f}  "
                  f"Recall={res['recall']:.4f}  IoU={res['iou']:.4f}")

    # Save fold results
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)

    # Clean up model weights and symlinks to save disk space
    shutil.rmtree(train_labels_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.rmtree(seg_dir, ignore_errors=True)

    return fold_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--held_out',            required=True)
    parser.add_argument('--labels_dir',          required=True)
    parser.add_argument('--gt_dirs',             required=True, nargs='+')
    parser.add_argument('--output_dir',          required=True)
    parser.add_argument('--checkpoint',          required=True)
    parser.add_argument('--intensity_priors_dir',required=True)
    parser.add_argument('--epochs',              type=int, default=10)
    parser.add_argument('--steps_per_epoch',     type=int, default=1000)
    args = parser.parse_args()

    run_fold(
        held_out=args.held_out,
        labels_dir=args.labels_dir,
        gt_dirs=args.gt_dirs,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        intensity_priors_dir=args.intensity_priors_dir,
    )
