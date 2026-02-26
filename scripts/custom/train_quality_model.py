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
from glob import glob

from claustrum_uncertainty import CLAUSTRUM_LABELS, extract_quality_features


def load_and_match_data(tta_dir, dice_csv, roi_size=(64, 64, 64)):
    """
    Load TTA outputs and match with actual Dice scores.

    Supports two modes:
    - Single TTA dir: tta_dir contains tta_segmentations/ and uncertainty_maps/
    - Parent seg dir: tta_dir is a parent containing multiple checkpoint dirs,
      each with a tta_uncertainty/ subdirectory. Matches Dice scores per
      checkpoint using the prediction_path column in the CSV.

    Args:
        tta_dir: Single TTA dir, or parent seg dir with multiple checkpoints
        dice_csv: Path to CSV with columns: subject_id, hemisphere, dice,
                  and optionally prediction_path (for multi-checkpoint matching)
        roi_size: Unused (kept for API compatibility)

    Returns:
        features, dice_scores, subject_ids
    """
    # Load Dice scores
    df = pd.read_csv(dice_csv)
    print(f"Loaded {len(df)} evaluation results from {dice_csv}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dice range: {df['dice'].min():.4f} - {df['dice'].max():.4f}")

    # Discover TTA directories
    # Case 1: tta_dir itself has tta_segmentations/
    # Case 2: tta_dir is parent, scan for */tta_uncertainty/tta_segmentations/
    tta_dirs = []
    if os.path.isdir(os.path.join(tta_dir, 'tta_segmentations')):
        tta_dirs.append(tta_dir)
    else:
        # Scan subdirectories for tta_uncertainty folders
        for subdir in sorted(os.listdir(tta_dir)):
            candidate = os.path.join(tta_dir, subdir, 'tta_uncertainty')
            if os.path.isdir(os.path.join(candidate, 'tta_segmentations')):
                tta_dirs.append(candidate)

    print(f"\nFound {len(tta_dirs)} TTA output directories")

    # Build Dice lookup
    # If prediction_path column exists, use (prediction_path, subject_id, hemi) as key
    # to match the correct Dice for each checkpoint
    has_pred_path = 'prediction_path' in df.columns
    dice_lookup = {}
    for _, row in df.iterrows():
        sid = str(row['subject_id'])
        hemi = row['hemisphere'].lower()
        dice = row['dice']
        if has_pred_path:
            pred_path = str(row['prediction_path'])
            dice_lookup[(pred_path, sid, hemi)] = dice
        # Always add a simple key as fallback
        simple_key = (sid, hemi)
        if simple_key not in dice_lookup:
            dice_lookup[simple_key] = dice

    all_features = []
    dice_scores = []
    subject_ids = []
    matched = 0
    unmatched = 0

    for tta_d in tta_dirs:
        seg_dir = os.path.join(tta_d, 'tta_segmentations')
        unc_dir = os.path.join(tta_d, 'uncertainty_maps')

        # Derive the prediction_path for this checkpoint
        # tta_d is like .../dice_finetune_055_100/tta_uncertainty
        # prediction_path is .../dice_finetune_055_100
        checkpoint_path = os.path.dirname(tta_d)
        checkpoint_name = os.path.basename(checkpoint_path)

        seg_files = sorted(glob(os.path.join(seg_dir, '*_tta_seg.nii.gz')))
        print(f"\n  [{checkpoint_name}] {len(seg_files)} TTA segmentations")

        for seg_path in seg_files:
            basename = os.path.basename(seg_path).replace('_tta_seg.nii.gz', '')
            match = re.search(r'(\d+).*?(lh|rh)', basename, re.IGNORECASE)
            if not match:
                continue

            subject_id = match.group(1)
            hemi = match.group(2).lower()

            # Look up Dice — try checkpoint-specific key first
            dice = None
            if has_pred_path:
                dice = dice_lookup.get((checkpoint_path, subject_id, hemi))
            if dice is None:
                dice = dice_lookup.get((subject_id, hemi))
            if dice is None:
                print(f"    No Dice for {subject_id} {hemi}, skipping")
                unmatched += 1
                continue

            # Find entropy map
            entropy_path = os.path.join(unc_dir, f'{basename}_entropy.nii.gz')
            if not os.path.isfile(entropy_path):
                for utype in ['entropy', 'variance', 'confidence', 'mutual_information']:
                    alt = os.path.join(unc_dir, f'{basename}_{utype}.nii.gz')
                    if os.path.isfile(alt):
                        entropy_path = alt
                        break
                else:
                    unmatched += 1
                    continue

            # Load and extract features (hemisphere-aware)
            seg_data = nib.load(seg_path).get_fdata().astype(np.int32)
            unc_data = nib.load(entropy_path).get_fdata().astype(np.float32)
            features = extract_quality_features(seg_data, unc_data, hemisphere=hemi)

            all_features.append(features)
            dice_scores.append(dice)
            subject_ids.append(f"{checkpoint_name}/{subject_id}_{hemi}")
            matched += 1

    print(f"\nTotal matched: {matched}, Unmatched: {unmatched}")
    if matched > 0:
        print(f"Dice distribution: mean={np.mean(dice_scores):.4f}, "
              f"std={np.std(dice_scores):.4f}, "
              f"range=[{np.min(dice_scores):.4f}, {np.max(dice_scores):.4f}]")

    return (np.array(all_features), np.array(dice_scores), subject_ids)



def train(tta_dir, dice_csv, save_path, roi_size=(64, 64, 64),
          epochs=100, batch_size=4, val_split=0.2):
    """Train the quality model and save weights."""

    # Load data (features already extracted per-hemisphere)
    X, dice_scores, subject_ids = load_and_match_data(
        tta_dir, dice_csv, roi_size)

    # Filter out Dice=0 (likely GT matching failures, not real scores)
    valid = dice_scores > 0.01
    n_removed = (~valid).sum()
    if n_removed > 0:
        print(f"\nRemoved {n_removed} samples with Dice~0 (likely GT matching failures)")
        X = X[valid]
        dice_scores = dice_scores[valid]
        subject_ids = [s for s, v in zip(subject_ids, valid) if v]

    if len(dice_scores) < 10:
        print(f"\nWARNING: Only {len(dice_scores)} samples.")
        if len(dice_scores) < 5:
            print("Too few samples, aborting.")
            return None

    y = dice_scores.astype(np.float32)
    feature_names = [
        'cl_volume', 'cl_fraction', 'compactness',
        'unc_mean_cl', 'unc_std_cl', 'unc_max_cl', 'unc_median_cl', 'unc_high_frac',
        'unc_boundary_mean', 'unc_boundary_max',
        'unc_global_mean', 'unc_global_std',
    ]

    print(f"\nFeature matrix: {X.shape} ({len(feature_names)} features)")
    print(f"Features: {feature_names}")
    print(f"Dice scores: n={len(y)}, mean={y.mean():.4f}, std={y.std():.4f}")

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # === Approach 1: Ridge regression (baseline) ===
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    # Leave-one-out cross-validation for honest evaluation
    loo_preds = cross_val_predict(ridge, X_norm, y, cv=LeaveOneOut())
    loo_corr = np.corrcoef(y, loo_preds)[0, 1]
    loo_rmse = np.sqrt(np.mean((y - loo_preds) ** 2))

    # Fit on all data for final model
    ridge.fit(X_norm, y)
    print(f"\n=== Ridge Regression (LOO cross-validated) ===")
    print(f"Pearson r: {loo_corr:.4f}")
    print(f"RMSE: {loo_rmse:.4f}")
    print(f"Best alpha: {ridge.alpha_:.2f}")

    # Feature importance
    coefs = ridge.coef_
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature importance (Ridge coefficients):")
    for name, coef in importance:
        print(f"  {name:<20s} {coef:+.4f}")

    # === Approach 2: Small neural net (if enough data) ===
    nn_corr = None
    if len(y) >= 20:
        import keras
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model as KerasModel
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        inp = Input(shape=(X_norm.shape[1],))
        h = Dense(32, activation='relu')(inp)
        h = Dropout(0.3)(h)
        h = Dense(16, activation='relu')(h)
        h = Dropout(0.2)(h)
        out = Dense(1, activation='linear')(h)
        nn_model = KerasModel(inputs=inp, outputs=out)
        nn_model.compile(optimizer='adam', loss='mse')

        # LOO for neural net too (slower but honest)
        nn_loo_preds = np.zeros_like(y)
        for i in range(len(y)):
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            nn_model.fit(X_norm[mask], y[mask],
                         epochs=200, batch_size=max(1, len(y) // 4),
                         verbose=0, validation_split=0.15,
                         callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])
            nn_loo_preds[i] = nn_model.predict(X_norm[i:i+1], verbose=0)[0, 0]

        nn_corr = np.corrcoef(y, nn_loo_preds)[0, 1]
        nn_rmse = np.sqrt(np.mean((y - nn_loo_preds) ** 2))

        print(f"\n=== Small Neural Net (LOO cross-validated) ===")
        print(f"Pearson r: {nn_corr:.4f}")
        print(f"RMSE: {nn_rmse:.4f}")

        # Train final NN on all data
        nn_model.fit(X_norm, y, epochs=200,
                     batch_size=max(1, len(y) // 4), verbose=0,
                     callbacks=[EarlyStopping(patience=30, restore_best_weights=True)])

    # Pick best approach
    best = 'nn' if (nn_corr is not None and nn_corr > loo_corr) else 'ridge'
    print(f"\n=== Best approach: {best} ===")

    # Save model + normalization params
    save_data = {
        'X_mean': X_mean, 'X_std': X_std,
        'feature_names': feature_names,
        'best_approach': best,
        'ridge_coef': ridge.coef_,
        'ridge_intercept': ridge.intercept_,
        'ridge_alpha': ridge.alpha_,
    }
    np.savez(save_path.replace('.h5', '_features.npz'), **save_data)
    print(f"Feature model saved to {save_path.replace('.h5', '_features.npz')}")

    if best == 'nn':
        nn_model.save_weights(save_path)
        print(f"NN weights saved to {save_path}")

    # Per-subject results (using LOO predictions for honest eval)
    preds = loo_preds if best == 'ridge' else nn_loo_preds
    print(f"\nPer-subject predictions vs actual (LOO):")
    print(f"{'Subject':<20} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 48)
    for sid, actual, pred in sorted(zip(subject_ids, y, preds), key=lambda x: x[1]):
        print(f"{sid:<20} {actual:>8.4f} {pred:>10.4f} {pred-actual:>+8.4f}")

    return ridge


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
