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
            # Use basename of prediction_path for matching (avoids local vs remote path mismatch)
            pred_basename = os.path.basename(str(row['prediction_path']).rstrip('/'))
            dice_lookup[(pred_basename, sid, hemi)] = dice
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

            # Look up Dice — use checkpoint-specific key when available
            dice = None
            if has_pred_path:
                dice = dice_lookup.get((checkpoint_name, subject_id, hemi))
                # Don't fall back to simple key — that would assign wrong checkpoint's Dice
            else:
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

    # === Extract subject IDs for LOSO grouping ===
    # subject_ids look like "checkpoint_name/5740_lh" — extract numeric subject
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV

    subject_groups = []
    for sid in subject_ids:
        m = re.search(r'(\d+)_(?:lh|rh)', sid)
        subject_groups.append(m.group(1) if m else sid)
    unique_subjects = sorted(set(subject_groups))
    print(f"\nLOSO subjects ({len(unique_subjects)}): {unique_subjects}")

    def loso_cv(fit_predict_fn, X_data, y_data, groups):
        """Leave-One-Subject-Out cross-validation."""
        preds = np.zeros_like(y_data)
        unique = sorted(set(groups))
        for held_out in unique:
            test_mask = np.array([g == held_out for g in groups])
            train_mask = ~test_mask
            preds[test_mask] = fit_predict_fn(
                X_data[train_mask], y_data[train_mask],
                X_data[test_mask])
        return preds

    def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=95):
        """Bootstrap confidence interval for a metric."""
        rng = np.random.RandomState(42)
        n = len(y_true)
        scores = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            scores[b] = metric_fn(y_true[idx], y_pred[idx])
        lo = np.percentile(scores, (100 - ci) / 2)
        hi = np.percentile(scores, 100 - (100 - ci) / 2)
        return lo, hi

    def pearson_r(a, b):
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    def spearman_r(a, b):
        if len(a) < 3:
            return 0.0
        return spearmanr(a, b).correlation

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    # === Approach 1: Ridge regression ===
    def ridge_fit_predict(X_tr, y_tr, X_te):
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, len(y_tr)))
        m.fit(X_tr, y_tr)
        return m.predict(X_te)

    loso_preds_ridge = loso_cv(ridge_fit_predict, X_norm, y, subject_groups)
    ridge_pearson = pearson_r(y, loso_preds_ridge)
    ridge_spearman = spearman_r(y, loso_preds_ridge)
    ridge_rmse = rmse(y, loso_preds_ridge)

    # Bootstrap CIs
    ridge_pearson_ci = bootstrap_ci(y, loso_preds_ridge, pearson_r)
    ridge_spearman_ci = bootstrap_ci(y, loso_preds_ridge, spearman_r)
    ridge_rmse_ci = bootstrap_ci(y, loso_preds_ridge, lambda a, b: rmse(a, b))

    # Fit final model on all data
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_norm, y)

    print(f"\n=== Ridge Regression (LOSO cross-validated, {len(unique_subjects)} folds) ===")
    print(f"Pearson r:  {ridge_pearson:.4f}  95% CI [{ridge_pearson_ci[0]:.4f}, {ridge_pearson_ci[1]:.4f}]")
    print(f"Spearman r: {ridge_spearman:.4f}  95% CI [{ridge_spearman_ci[0]:.4f}, {ridge_spearman_ci[1]:.4f}]")
    print(f"RMSE:       {ridge_rmse:.4f}  95% CI [{ridge_rmse_ci[0]:.4f}, {ridge_rmse_ci[1]:.4f}]")
    print(f"Best alpha: {ridge.alpha_:.2f}")

    # Feature importance
    coefs = ridge.coef_
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nFeature importance (Ridge coefficients):")
    for name, coef in importance:
        print(f"  {name:<20s} {coef:+.4f}")

    # === Approach 2: Small neural net (if enough data) ===
    nn_pearson = None
    loso_preds_nn = None
    if len(y) >= 20:
        import keras
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model as KerasModel
        from keras.callbacks import EarlyStopping

        def nn_fit_predict(X_tr, y_tr, X_te):
            inp = Input(shape=(X_tr.shape[1],))
            h = Dense(32, activation='relu')(inp)
            h = Dropout(0.3)(h)
            h = Dense(16, activation='relu')(h)
            h = Dropout(0.2)(h)
            out = Dense(1, activation='linear')(h)
            m = KerasModel(inputs=inp, outputs=out)
            m.compile(optimizer='adam', loss='mse')
            m.fit(X_tr, y_tr, epochs=200,
                  batch_size=max(1, len(y_tr) // 4), verbose=0,
                  validation_split=0.15,
                  callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])
            return m.predict(X_te, verbose=0).flatten()

        loso_preds_nn = loso_cv(nn_fit_predict, X_norm, y, subject_groups)
        nn_pearson = pearson_r(y, loso_preds_nn)
        nn_spearman = spearman_r(y, loso_preds_nn)
        nn_rmse = rmse(y, loso_preds_nn)

        nn_pearson_ci = bootstrap_ci(y, loso_preds_nn, pearson_r)
        nn_spearman_ci = bootstrap_ci(y, loso_preds_nn, spearman_r)
        nn_rmse_ci = bootstrap_ci(y, loso_preds_nn, lambda a, b: rmse(a, b))

        print(f"\n=== Small Neural Net (LOSO cross-validated, {len(unique_subjects)} folds) ===")
        print(f"Pearson r:  {nn_pearson:.4f}  95% CI [{nn_pearson_ci[0]:.4f}, {nn_pearson_ci[1]:.4f}]")
        print(f"Spearman r: {nn_spearman:.4f}  95% CI [{nn_spearman_ci[0]:.4f}, {nn_spearman_ci[1]:.4f}]")
        print(f"RMSE:       {nn_rmse:.4f}  95% CI [{nn_rmse_ci[0]:.4f}, {nn_rmse_ci[1]:.4f}]")

        # Train final NN on all data
        inp = Input(shape=(X_norm.shape[1],))
        h = Dense(32, activation='relu')(inp)
        h = Dropout(0.3)(h)
        h = Dense(16, activation='relu')(h)
        h = Dropout(0.2)(h)
        out = Dense(1, activation='linear')(h)
        nn_model = KerasModel(inputs=inp, outputs=out)
        nn_model.compile(optimizer='adam', loss='mse')
        nn_model.fit(X_norm, y, epochs=200,
                     batch_size=max(1, len(y) // 4), verbose=0,
                     callbacks=[EarlyStopping(patience=30, restore_best_weights=True)])

    # Pick best approach
    best = 'nn' if (nn_pearson is not None and nn_pearson > ridge_pearson) else 'ridge'
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

    # Per-subject results (LOSO predictions)
    preds = loso_preds_ridge if best == 'ridge' else loso_preds_nn
    print(f"\nPer-subject predictions vs actual (LOSO):")
    print(f"{'Subject':<40} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 68)
    for sid, actual, pred in sorted(zip(subject_ids, y, preds), key=lambda x: x[1]):
        print(f"{sid:<40} {actual:>8.4f} {pred:>10.4f} {pred-actual:>+8.4f}")

    # Save LOSO results as CSV for plots (Bland-Altman, scatter)
    results_df = pd.DataFrame({
        'subject': subject_ids,
        'subject_group': subject_groups,
        'actual_dice': y,
        'predicted_dice': preds,
        'error': preds - y,
    })
    results_csv = save_path.replace('.h5', '_loso_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nLOSO results saved to {results_csv}")

    # Print summary for paper
    best_pearson = ridge_pearson if best == 'ridge' else nn_pearson
    best_spearman = ridge_spearman if best == 'ridge' else nn_spearman
    best_rmse = ridge_rmse if best == 'ridge' else nn_rmse
    best_p_ci = ridge_pearson_ci if best == 'ridge' else nn_pearson_ci
    best_s_ci = ridge_spearman_ci if best == 'ridge' else nn_spearman_ci
    best_r_ci = ridge_rmse_ci if best == 'ridge' else nn_rmse_ci
    print(f"\n{'='*60}")
    print(f"RESULTS FOR PAPER ({best}, LOSO CV, {len(unique_subjects)} subjects)")
    print(f"{'='*60}")
    print(f"Pearson r  = {best_pearson:.3f} (95% CI: {best_p_ci[0]:.3f}-{best_p_ci[1]:.3f})")
    print(f"Spearman r = {best_spearman:.3f} (95% CI: {best_s_ci[0]:.3f}-{best_s_ci[1]:.3f})")
    print(f"RMSE       = {best_rmse:.4f} (95% CI: {best_r_ci[0]:.4f}-{best_r_ci[1]:.4f})")
    print(f"N samples  = {len(y)}, N subjects = {len(unique_subjects)}")
    print(f"{'='*60}")

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
