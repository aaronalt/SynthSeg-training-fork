"""
Train the 2-branch quality prediction CNN.

Reads TTA outputs (entropy maps + segmentations) and actual Dice scores
from evaluation CSVs, then trains the model to predict Dice from
(claustrum seg, uncertainty map) pairs — no ground truth needed at inference.

Usage:
    python train_quality_model.py \
        --tta_dir /path/to/tta_uncertainty \
        --dice_csv /path/to/epoch_evaluation_summary.csv \
        --save_path /path/to/quality_model_weights.h5
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import re
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom as nd_zoom
from glob import glob

from claustrum_uncertainty import build_quality_model, CLAUSTRUM_LABELS


def load_and_match_data(tta_dir, dice_csv, roi_size=(64, 64, 64)):
    """
    Load TTA outputs and match with actual Dice scores.

    Args:
        tta_dir: Directory containing tta_segmentations/ and uncertainty_maps/
        dice_csv: Path to CSV with columns: subject_id, hemisphere, dice
        roi_size: Fixed ROI size for the quality model

    Returns:
        seg_rois, unc_rois, dice_scores, subject_ids
    """
    # Load Dice scores
    df = pd.read_csv(dice_csv)
    print(f"Loaded {len(df)} evaluation results from {dice_csv}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dice range: {df['dice'].min():.4f} - {df['dice'].max():.4f}")

    # Build lookup: (subject_id, hemisphere) -> dice
    dice_lookup = {}
    for _, row in df.iterrows():
        key = (str(row['subject_id']), row['hemisphere'].lower())
        dice_lookup[key] = row['dice']

    # Find TTA outputs
    seg_dir = os.path.join(tta_dir, 'tta_segmentations')
    unc_dir = os.path.join(tta_dir, 'uncertainty_maps')

    seg_files = sorted(glob(os.path.join(seg_dir, '*_tta_seg.nii.gz')))
    print(f"\nFound {len(seg_files)} TTA segmentations in {seg_dir}")

    seg_rois = []
    unc_rois = []
    dice_scores = []
    subject_ids = []
    matched = 0
    unmatched = 0

    for seg_path in seg_files:
        # Extract subject_id and hemisphere from filename
        # Filename format: sub-XXXX.lh.crop_tta_seg.nii.gz
        basename = os.path.basename(seg_path).replace('_tta_seg.nii.gz', '')
        match = re.search(r'(\d+).*?(lh|rh)', basename, re.IGNORECASE)
        if not match:
            print(f"  Skipping (no ID/hemi match): {basename}")
            continue

        subject_id = match.group(1)
        hemi = match.group(2).lower()

        # Look up Dice
        key = (subject_id, hemi)
        if key not in dice_lookup:
            print(f"  No Dice score for {subject_id} {hemi}, skipping")
            unmatched += 1
            continue

        dice = dice_lookup[key]

        # Find corresponding entropy map
        entropy_path = os.path.join(unc_dir, f'{basename}_entropy.nii.gz')
        if not os.path.isfile(entropy_path):
            # Try other uncertainty types
            for utype in ['entropy', 'variance', 'confidence', 'mutual_information']:
                alt = os.path.join(unc_dir, f'{basename}_{utype}.nii.gz')
                if os.path.isfile(alt):
                    entropy_path = alt
                    break
            else:
                print(f"  No uncertainty map for {basename}, skipping")
                unmatched += 1
                continue

        # Load volumes (already cropped around claustrum)
        seg_data = nib.load(seg_path).get_fdata().astype(np.int32)
        unc_data = nib.load(entropy_path).get_fdata().astype(np.float32)

        # Create binary claustrum mask
        cl_binary = np.isin(seg_data, CLAUSTRUM_LABELS).astype(np.float32)

        # Resize to fixed input size (images are already claustrum-cropped)
        def resize_to_roi(vol, target_size):
            zf = [t / max(s, 1) for t, s in zip(target_size, vol.shape[:3])]
            return nd_zoom(vol.astype(np.float32), zf, order=1)

        seg_roi = resize_to_roi(cl_binary, roi_size)
        unc_roi = resize_to_roi(unc_data, roi_size)

        seg_rois.append(seg_roi)
        unc_rois.append(unc_roi)
        dice_scores.append(dice)
        subject_ids.append(f"{subject_id}_{hemi}")
        matched += 1

    print(f"\nMatched: {matched}, Unmatched: {unmatched}")
    print(f"Dice distribution: mean={np.mean(dice_scores):.4f}, "
          f"std={np.std(dice_scores):.4f}")

    return (np.array(seg_rois), np.array(unc_rois),
            np.array(dice_scores), subject_ids)


def train(tta_dir, dice_csv, save_path, roi_size=(64, 64, 64),
          epochs=100, batch_size=4, val_split=0.2):
    """Train the quality model and save weights."""

    # Load data
    seg_rois, unc_rois, dice_scores, subject_ids = load_and_match_data(
        tta_dir, dice_csv, roi_size)

    if len(dice_scores) < 10:
        print(f"\nWARNING: Only {len(dice_scores)} samples. "
              f"Need at least ~20 for meaningful training.")
        if len(dice_scores) < 5:
            print("Too few samples, aborting.")
            return None

    # Add channel dimension
    X_seg = seg_rois[..., np.newaxis]
    X_unc = unc_rois[..., np.newaxis]
    y = dice_scores.astype(np.float32)

    print(f"\nTraining data shapes:")
    print(f"  Seg ROIs: {X_seg.shape}")
    print(f"  Unc ROIs: {X_unc.shape}")
    print(f"  Dice scores: {y.shape}")

    # Build model
    model = build_quality_model(roi_size)
    model.summary()

    # Callbacks
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, verbose=1),
    ]

    # Train
    print(f"\nTraining with {len(y)} samples "
          f"({int(len(y) * (1 - val_split))} train, "
          f"{int(len(y) * val_split)} val)")

    history = model.fit(
        [X_seg, X_unc], y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Save weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_weights(save_path)
    print(f"\nModel saved to {save_path}")

    # Print final metrics
    val_loss = min(history.history.get('val_loss', [float('inf')]))
    print(f"Best validation MSE: {val_loss:.6f} (RMSE: {np.sqrt(val_loss):.4f})")

    # Quick evaluation: predict on all data and show correlation
    predictions = model.predict([X_seg, X_unc], verbose=0).flatten()
    correlation = np.corrcoef(y, predictions)[0, 1]
    print(f"\nFull dataset correlation (Pearson r): {correlation:.4f}")
    print(f"\nPer-subject predictions vs actual:")
    print(f"{'Subject':<20} {'Actual':>8} {'Predicted':>10}")
    print("-" * 40)
    for sid, actual, pred in zip(subject_ids, y, predictions):
        print(f"{sid:<20} {actual:>8.4f} {pred:>10.4f}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train quality prediction CNN')
    parser.add_argument('--tta_dir', required=True,
                        help='Path to tta_uncertainty directory')
    parser.add_argument('--dice_csv', required=True,
                        help='Path to evaluation CSV with actual Dice scores')
    parser.add_argument('--save_path', default='./quality_model_weights.h5',
                        help='Where to save trained weights')
    parser.add_argument('--roi_size', type=int, default=64,
                        help='ROI cube size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    args = parser.parse_args()

    train(
        tta_dir=args.tta_dir,
        dice_csv=args.dice_csv,
        save_path=args.save_path,
        roi_size=(args.roi_size,) * 3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
