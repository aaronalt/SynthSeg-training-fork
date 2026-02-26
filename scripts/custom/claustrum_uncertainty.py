"""
Claustrum Uncertainty Estimation via Test-Time Augmentation (TTA)
and Segmentation Quality Prediction using 2-Subnetwork CNN.

Reference: Sikha et al. (2025) "Uncertainty-aware segmentation quality
prediction via deep learning Bayesian Modeling"

Adapted for 3D brain MRI claustrum segmentation (labels 138/139)
using the actual SynthSeg augmentation layers (BiasFieldCorruption,
GaussianNoiseCorruption, IntensityAugmentation) for TTA — the same
generators used during training.

Usage:
    # As module (called from prediction.py):
    from claustrum_uncertainty import predict_with_tta
    results = predict_with_tta(path_images, path_model, ...)
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from scipy.ndimage import zoom
from pathlib import Path

from SynthSeg.predict import preprocess, build_model, postprocess, get_flip_indices
from ext.lab2im import utils, edit_volumes

keras.backend.set_image_data_format('channels_last')

CLAUSTRUM_LABELS = [138, 139]  # LH, RH


# ===================================================================
# SECTION 1: TTA AUGMENTATION (numpy, mirrors SynthSeg intensity chain)
# ===================================================================

def apply_tta_augmentation(image, bias_field_std=0.3, bias_scale=0.025,
                           noise_std=100, gamma_std=0.5, tta_strength=0.25):
    """
    Apply intensity augmentations matching the SynthSeg training chain,
    scaled by tta_strength for test-time use.

    During training, full-strength augmentation teaches robustness.
    At test time, we want *gentle* perturbations that probe the model's
    decision boundaries without overwhelming the signal.

    tta_strength=1.0 matches training intensity (too aggressive for TTA).
    tta_strength=0.25 (default) applies 25% of training augmentation —
    enough to reveal boundary uncertainty without destroying predictions.

    Chain:
        1. Bias field corruption — smooth multiplicative field
        2. Gaussian noise — voxel-wise additive noise
        3. Gamma augmentation — random contrast shift
        4. Re-normalise to [0, 1]

    Input images are already [0, 1] from predict's preprocess(), so
    noise_std is scaled by 1/300 to match the effective level after
    training's own clip-to-300 + normalise step.

    Args:
        image: (1, H, W, D, C) numpy array, values in [0, 1]
        bias_field_std, bias_scale, noise_std, gamma_std: augmentation
            parameters (should match training config)
        tta_strength: scale factor (0-1) applied to all augmentation
            parameters. 1.0 = full training intensity, 0.25 = gentle TTA.

    Returns:
        Augmented image, same shape, re-normalised to [0, 1]
    """
    from scipy.ndimage import zoom as nd_zoom

    # Scale augmentation parameters by tta_strength
    bias_field_std = bias_field_std * tta_strength
    noise_std = noise_std * tta_strength
    gamma_std = gamma_std * tta_strength

    aug = image.copy().astype(np.float32)
    spatial_shape = aug.shape[1:-1]  # (H, W, D)
    n_dims = len(spatial_shape)

    # 1. Smooth multiplicative bias field
    if bias_field_std > 0:
        # Sample small random field, upsample to full resolution
        small_shape = [max(1, int(np.ceil(s * bias_scale)))
                       for s in spatial_shape]
        small_field = np.random.normal(0, bias_field_std, small_shape)
        zoom_factors = [s / max(sf, 1)
                        for s, sf in zip(spatial_shape, small_shape)]
        bias_field = nd_zoom(small_field, zoom_factors, order=3)
        bias_field = np.exp(bias_field)  # multiplicative
        # Apply to each channel
        for c in range(aug.shape[-1]):
            aug[0, ..., c] *= bias_field

    # 2. Additive Gaussian noise (scaled for [0,1] range)
    if noise_std > 0 and np.random.rand() < 0.95:  # prob=0.95 as in training
        effective_std = np.random.uniform(0, noise_std / 300.0)
        noise = np.random.normal(0, effective_std, aug.shape)
        aug += noise.astype(np.float32)

    # 3. Gamma augmentation (random contrast)
    if gamma_std > 0:
        gamma = np.exp(np.random.normal(0, gamma_std))
        # Apply per-channel (separate_channels=True in training)
        for c in range(aug.shape[-1]):
            ch = aug[0, ..., c]
            mn, mx = ch.min(), ch.max()
            if mx - mn > 1e-8:
                ch = (ch - mn) / (mx - mn)
                ch = np.power(ch, gamma)
                aug[0, ..., c] = ch * (mx - mn) + mn

    # 4. Re-normalise to [0, 1]
    mn, mx = aug.min(), aug.max()
    if mx - mn > 1e-8:
        aug = (aug - mn) / (mx - mn)
    else:
        aug = np.clip(aug, 0, 1)

    return aug.astype(np.float32)


# ===================================================================
# SECTION 2: TTA POSTERIOR GENERATION & UNCERTAINTY COMPUTATION
# ===================================================================

def generate_tta_posteriors(image_preprocessed, seg_net, n_augmentations=10,
                            bias_field_std=0.3, bias_scale=0.025,
                            noise_std=100, gamma_std=0.5, tta_strength=0.25):
    """
    Run N augmented forward passes through the segmentation model.

    Args:
        image_preprocessed: (1, H, W, D, 1) preprocessed image
        seg_net: Trained SynthSeg Keras model (may include flip averaging)
        n_augmentations: Total predictions including the un-augmented one
        bias_field_std, bias_scale, noise_std, gamma_std: augmentation
            parameters (should match training config)

    Returns:
        np.ndarray of shape (N, H, W, D, n_labels) — squeezed posteriors
    """
    all_posteriors = []

    # 1. Original (un-augmented) prediction
    post = seg_net.predict(image_preprocessed, verbose=0)
    all_posteriors.append(np.squeeze(post))

    # 2. Augmented predictions — each call samples fresh random params
    for _ in range(n_augmentations - 1):
        aug = apply_tta_augmentation(
            image_preprocessed,
            bias_field_std=bias_field_std,
            bias_scale=bias_scale,
            noise_std=noise_std,
            gamma_std=gamma_std,
            tta_strength=tta_strength)
        post = seg_net.predict(aug, verbose=0)
        all_posteriors.append(np.squeeze(post))

    return np.stack(all_posteriors, axis=0)  # (N, H, W, D, n_labels)


def compute_claustrum_uncertainty(all_posteriors, labels_segmentation,
                                  claustrum_labels=None):
    """
    Compute uncertainty maps focused on claustrum labels.

    Args:
        all_posteriors: (N, H, W, D, n_labels)
        labels_segmentation: 1-D array of label values in channel order
        claustrum_labels: [138, 139] by default

    Returns:
        dict with keys: confidence, variance, entropy, mutual_information,
             mean_posteriors_all, mean_posteriors_claustrum
    """
    if claustrum_labels is None:
        claustrum_labels = CLAUSTRUM_LABELS

    labels_segmentation = np.asarray(labels_segmentation)

    # Indices of claustrum channels in the posteriors
    cl_indices = []
    for cl in claustrum_labels:
        idx = np.where(labels_segmentation == cl)[0]
        if len(idx) > 0:
            cl_indices.append(idx[0])
    if not cl_indices:
        raise ValueError(
            f"Claustrum labels {claustrum_labels} not in {labels_segmentation}")

    n_aug = all_posteriors.shape[0]
    cl_post = all_posteriors[..., cl_indices]         # (N, H, W, D, 2)
    mean_cl = np.mean(cl_post, axis=0)                # (H, W, D, 2)
    mean_all = np.mean(all_posteriors, axis=0)         # (H, W, D, n_labels)

    # -- Confidence: summed claustrum probability --
    cl_total = np.sum(mean_cl, axis=-1)               # (H, W, D)

    # -- Variance across TTA runs (mean over L/R channels) --
    variance = np.mean(np.var(cl_post, axis=0), axis=-1)

    # -- Entropy of mean prediction (claustrum vs. background) --
    bg_prob = 1.0 - cl_total
    probs = np.stack(
        [bg_prob] + [mean_cl[..., i] for i in range(mean_cl.shape[-1])],
        axis=-1)
    probs = np.clip(probs, 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)

    # -- Mutual Information = H[E[p]] - E[H[p]] --
    indiv_ent = np.zeros_like(entropy)
    for t in range(n_aug):
        cl_t = cl_post[t]
        bg_t = 1.0 - np.sum(cl_t, axis=-1)
        p_t = np.stack(
            [bg_t] + [cl_t[..., j] for j in range(cl_t.shape[-1])], axis=-1)
        p_t = np.clip(p_t, 1e-8, 1.0)
        indiv_ent += -np.sum(p_t * np.log(p_t), axis=-1)
    indiv_ent /= n_aug
    mutual_info = np.clip(entropy - indiv_ent, 0, None)

    return dict(
        confidence=cl_total,
        variance=variance,
        entropy=entropy,
        mutual_information=mutual_info,
        mean_posteriors_all=mean_all,
        mean_posteriors_claustrum=mean_cl,
    )


def compute_tta_dice(all_posteriors, labels_segmentation,
                     claustrum_labels=None):
    """
    Compute TTA self-consistency Dice: mean pairwise Dice between all
    N argmax segmentations derived from TTA posteriors, restricted to
    claustrum labels. This is a per-subject quality metric that requires
    no ground truth.

    High TTA Dice = model gives consistent claustrum predictions under
    augmentation = likely reliable segmentation.

    Computed separately for each claustrum label (LH/RH) and combined.

    Args:
        all_posteriors: (N, H, W, D, n_labels) from generate_tta_posteriors
        labels_segmentation: 1-D array of label values in channel order
        claustrum_labels: [138, 139] by default

    Returns:
        dict with keys:
            tta_dice_combined: float — mean pairwise Dice over all claustrum
            tta_dice_per_label: dict {label: float} — per-hemisphere
            pairwise_dices: (N, N) array — full pairwise matrix (combined)
    """
    if claustrum_labels is None:
        claustrum_labels = CLAUSTRUM_LABELS

    labels_segmentation = np.asarray(labels_segmentation)
    n_aug = all_posteriors.shape[0]

    # Argmax each TTA run to get hard segmentations
    seg_indices = np.argmax(all_posteriors, axis=-1)  # (N, H, W, D)
    # Map indices back to label values
    tta_segs = labels_segmentation[seg_indices]        # (N, H, W, D)

    def _pairwise_dice(masks):
        """Mean pairwise Dice for a list of N binary masks."""
        n = len(masks)
        dices = []
        for i in range(n):
            for j in range(i + 1, n):
                intersection = np.sum(masks[i] & masks[j])
                union = np.sum(masks[i]) + np.sum(masks[j])
                if union == 0:
                    continue  # both empty — skip, not meaningful
                else:
                    dices.append(2.0 * intersection / union)
        return float(np.mean(dices)) if dices else float('nan')

    # Per-label Dice
    per_label = {}
    all_masks_combined = []
    for cl in claustrum_labels:
        masks = [(tta_segs[t] == cl) for t in range(n_aug)]
        per_label[int(cl)] = _pairwise_dice(masks)
        if not all_masks_combined:
            all_masks_combined = [m.copy() for m in masks]
        else:
            for t in range(n_aug):
                all_masks_combined[t] |= masks[t]

    # Combined (any claustrum label)
    combined_dice = _pairwise_dice(all_masks_combined)

    # Full pairwise matrix (combined) for reference
    pairwise = np.full((n_aug, n_aug), np.nan, dtype=np.float32)
    np.fill_diagonal(pairwise, 1.0)
    for i in range(n_aug):
        for j in range(i + 1, n_aug):
            inter = np.sum(all_masks_combined[i] & all_masks_combined[j])
            total = np.sum(all_masks_combined[i]) + np.sum(all_masks_combined[j])
            d = 2.0 * inter / total if total > 0 else np.nan
            pairwise[i, j] = pairwise[j, i] = d

    return dict(
        tta_dice_combined=float(combined_dice),
        tta_dice_per_label=per_label,
        pairwise_dices=pairwise,
    )


# ===================================================================
# SECTION 3: QUALITY PREDICTION — 2-SUBNETWORK CNN (3D)
# ===================================================================

def extract_quality_features(seg_data, unc_data):
    """
    Extract summary statistics from a segmentation + uncertainty map pair.
    Used by the ridge regression quality model to predict Dice.

    Args:
        seg_data: 3D int array (segmentation labels)
        unc_data: 3D float array (entropy/uncertainty map, same shape)

    Returns:
        1-D numpy array of 12 features
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    claustrum_mask = np.isin(seg_data, CLAUSTRUM_LABELS)
    cl_volume = claustrum_mask.sum()
    total_volume = seg_data.size
    cl_fraction = cl_volume / total_volume if total_volume > 0 else 0

    if cl_volume > 0:
        unc_in_cl = unc_data[claustrum_mask]
        unc_mean_cl = unc_in_cl.mean()
        unc_std_cl = unc_in_cl.std()
        unc_max_cl = unc_in_cl.max()
        unc_median_cl = float(np.median(unc_in_cl))
        unc_high_frac = float((unc_in_cl > np.percentile(unc_data, 90)).mean())
    else:
        unc_mean_cl = unc_std_cl = unc_max_cl = unc_median_cl = 0.0
        unc_high_frac = 1.0

    dilated = binary_dilation(claustrum_mask, iterations=2)
    boundary = dilated & ~claustrum_mask
    if boundary.sum() > 0:
        unc_boundary_mean = unc_data[boundary].mean()
        unc_boundary_max = unc_data[boundary].max()
    else:
        unc_boundary_mean = unc_boundary_max = 0.0

    unc_global_mean = unc_data.mean()
    unc_global_std = unc_data.std()

    eroded = binary_erosion(claustrum_mask, iterations=1)
    surface_voxels = claustrum_mask.sum() - eroded.sum()
    compactness = surface_voxels / max(cl_volume, 1)

    return np.array([
        cl_volume, cl_fraction, compactness,
        unc_mean_cl, unc_std_cl, unc_max_cl, unc_median_cl, unc_high_frac,
        unc_boundary_mean, unc_boundary_max,
        unc_global_mean, unc_global_std,
    ], dtype=np.float32)


def load_quality_model(weights_path):
    """
    Load a trained ridge regression quality model from .npz file.

    Args:
        weights_path: Path to .npz file saved by train_quality_model.py

    Returns:
        dict with keys: coef, intercept, X_mean, X_std
        or None if file not found
    """
    if not weights_path or not os.path.isfile(weights_path):
        return None

    data = np.load(weights_path, allow_pickle=True)
    return dict(
        coef=data['ridge_coef'],
        intercept=float(data['ridge_intercept']),
        X_mean=data['X_mean'],
        X_std=data['X_std'],
    )


def predict_quality(seg_data, unc_data, quality_params):
    """
    Predict Dice score for a single subject using the ridge model.

    Args:
        seg_data: 3D int array (segmentation)
        unc_data: 3D float array (entropy map)
        quality_params: dict from load_quality_model()

    Returns:
        float: predicted Dice score
    """
    features = extract_quality_features(seg_data, unc_data)
    X_norm = (features - quality_params['X_mean']) / (quality_params['X_std'] + 1e-8)
    pred = float(X_norm @ quality_params['coef'] + quality_params['intercept'])
    return np.clip(pred, 0.0, 1.0)


# ===================================================================
# SECTION 4: HIGH-LEVEL TTA PREDICTION PIPELINE
# ===================================================================

def predict_with_tta(path_images,
                     path_model,
                     labels_segmentation,
                     output_dir,
                     n_neutral_labels=None,
                     n_augmentations=10,
                     n_levels=5,
                     nb_conv_per_level=2,
                     conv_size=3,
                     unet_feat_count=24,
                     feat_multiplier=2,
                     activation='elu',
                     target_res=1.0,
                     cropping=None,
                     min_pad=None,
                     sigma_smoothing=0.5,
                     topology_classes=None,
                     keep_biggest_component=False,
                     noise_std=100,
                     bias_field_std=0.3,
                     bias_scale=0.025,
                     gamma_std=0.5,
                     tta_strength=0.25,
                     uncertainty_type='entropy',
                     quality_model_weights=None):
    """
    Full TTA pipeline: for each image, generate N augmented predictions
    using the actual SynthSeg augmentation layers, compute claustrum-
    focused uncertainty maps, and optionally predict segmentation quality
    (Dice) using a trained 2-subnetwork model.

    Args:
        path_images: Directory or single image path
        path_model: Path to trained SynthSeg model (.h5)
        labels_segmentation: Array of segmentation label values
        output_dir: Where to save uncertainty maps and TTA segmentations
        n_neutral_labels: Number of non-sided labels (for flip augmentation)
        n_augmentations: Number of TTA iterations (default 10)
        Architecture / preprocessing params: same as SynthSeg predict()
        noise_std, bias_field_std, bias_scale, gamma_std: augmentation
            params (should match training — loaded from training_params.json)
        uncertainty_type: 'entropy', 'variance', 'confidence',
                          'mutual_information'
        quality_model_weights: Path to trained quality model (optional)
        quality_roi_size: ROI size for quality model

    Returns:
        List of result dicts per image, each containing:
            path, segmentation, uncertainty_map, predicted_dice (if model)
    """
    labels_segmentation, _ = utils.get_list_labels(
        label_list=labels_segmentation)

    flip = n_neutral_labels is not None
    if flip:
        labels_segmentation, flip_indices, unique_idx = get_flip_indices(
            labels_segmentation, n_neutral_labels)
    else:
        labels_segmentation, unique_idx = np.unique(
            labels_segmentation, return_index=True)
        flip_indices = None

    if topology_classes is not None:
        topology_classes = utils.load_array_if_path(
            topology_classes, load_as_numpy=True)[unique_idx]

    # Resolve image paths
    path_images_abs = os.path.abspath(path_images)
    if os.path.isdir(path_images_abs):
        image_paths = utils.list_images_in_folder(path_images_abs)
    else:
        image_paths = [path_images_abs]

    # Build segmentation network
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(image_paths[0])
    model_input_shape = [None] * n_dims + [n_channels]
    seg_net = build_model(
        path_model=path_model, input_shape=model_input_shape,
        labels_segmentation=labels_segmentation,
        n_levels=n_levels, nb_conv_per_level=nb_conv_per_level,
        conv_size=conv_size, unet_feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier, activation=activation,
        sigma_smoothing=sigma_smoothing, flip_indices=flip_indices,
        gradients=False)

    # Load quality prediction model if provided
    quality_params = load_quality_model(quality_model_weights)
    if quality_params is not None:
        print(f"Loaded quality prediction model from {quality_model_weights}")

    # Output directories
    tta_seg_dir = os.path.join(output_dir, 'tta_segmentations')
    unc_dir = os.path.join(output_dir, 'uncertainty_maps')
    os.makedirs(tta_seg_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)

    results = []
    print(f"\nTTA uncertainty estimation: {len(image_paths)} images, "
          f"{n_augmentations} augmentations each")

    for idx, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path).replace('.nii.gz', '')
        print(f"  [{idx+1}/{len(image_paths)}] {basename}")

        # Preprocess
        if (cropping is not None) and (min_pad is not None):
            _crop = utils.reformat_to_list(cropping, length=n_dims, dtype='int')
            _min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
            _min_pad = np.minimum(_crop, _min_pad)
        else:
            _crop = cropping
            _min_pad = min_pad

        image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(
            img_path, n_levels, target_res, crop=_crop, min_pad=_min_pad)

        # TTA forward passes with SynthSeg-matched augmentations
        all_posteriors = generate_tta_posteriors(
            image, seg_net, n_augmentations,
            bias_field_std=bias_field_std,
            bias_scale=bias_scale,
            noise_std=noise_std,
            gamma_std=gamma_std,
            tta_strength=tta_strength)

        # Compute claustrum uncertainty
        unc_results = compute_claustrum_uncertainty(
            all_posteriors, labels_segmentation)

        # Compute TTA self-consistency Dice (no ground truth needed)
        tta_dice_results = compute_tta_dice(
            all_posteriors, labels_segmentation)

        # Postprocess mean posteriors -> segmentation
        mean_post = unc_results['mean_posteriors_all']
        seg, posteriors, volumes = postprocess(
            post_patch=mean_post[np.newaxis],
            shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
            n_dims=n_dims, labels_segmentation=labels_segmentation,
            keep_biggest_component=keep_biggest_component,
            aff=aff, im_res=im_res, topology_classes=topology_classes)

        # Helper: map a volume from cropped/padded space to original space
        def _to_original_space(vol):
            vol = edit_volumes.crop_volume_with_idx(
                vol, pad_idx, n_dims=3, return_copy=False)
            if crop_idx is not None:
                full = np.zeros(shape, dtype=np.float32)
                full[crop_idx[0]:crop_idx[3],
                     crop_idx[1]:crop_idx[4],
                     crop_idx[2]:crop_idx[5]] = vol
                vol = full
            return edit_volumes.align_volume_to_ref(
                vol, np.eye(4), aff, n_dims=n_dims, return_copy=False)

        unc_map = _to_original_space(unc_results[uncertainty_type])
        conf_map = _to_original_space(unc_results['confidence'])

        # Save outputs
        seg_path = os.path.join(
            tta_seg_dir, f'{basename}_tta_seg.nii.gz')
        unc_path = os.path.join(
            unc_dir, f'{basename}_{uncertainty_type}.nii.gz')
        conf_path = os.path.join(
            unc_dir, f'{basename}_confidence.nii.gz')
        utils.save_volume(seg.astype('int32'), aff, h, seg_path)
        utils.save_volume(unc_map.astype('float32'), aff, h, unc_path)
        utils.save_volume(conf_map.astype('float32'), aff, h, conf_path)

        tta_combined = tta_dice_results['tta_dice_combined']
        tta_per_label = tta_dice_results['tta_dice_per_label']
        print(f"    TTA Dice: {tta_combined:.4f}  "
              f"(LH={tta_per_label.get(138, 0):.4f}, "
              f"RH={tta_per_label.get(139, 0):.4f})")

        result = dict(
            path=img_path, basename=basename,
            seg_path=seg_path, uncertainty_path=unc_path, confidence_path=conf_path,
            uncertainty_type=uncertainty_type,
            tta_dice=tta_combined,
            tta_dice_lh=tta_per_label.get(138, 0.0),
            tta_dice_rh=tta_per_label.get(139, 0.0),
            volumes=volumes, predicted_dice=None,
        )

        # Quality prediction (ridge regression on extracted features)
        if quality_params is not None:
            result['predicted_dice'] = predict_quality(
                seg, unc_map, quality_params)
            print(f"    Predicted Dice: {result['predicted_dice']:.4f}")

        results.append(result)

    # Save TTA Dice summary CSV
    if results:
        import pandas as pd
        summary_rows = []
        for r in results:
            summary_rows.append({
                'subject': r['basename'],
                'tta_dice_combined': r['tta_dice'],
                'tta_dice_lh': r['tta_dice_lh'],
                'tta_dice_rh': r['tta_dice_rh'],
                'predicted_dice': r.get('predicted_dice'),
            })
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, 'tta_dice_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nTTA Dice summary saved to {csv_path}")
        print(df.to_string(index=False))

    print(f"\nTTA complete. Outputs saved to {output_dir}")
    return results


# ===================================================================
# SECTION 5: STANDALONE EXECUTION
# ===================================================================

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Claustrum TTA uncertainty estimation')
    parser.add_argument('--images', required=True,
                        help='Input image directory or single file')
    parser.add_argument('--model', required=True,
                        help='Path to trained SynthSeg model (.h5)')
    parser.add_argument('--model_dir', required=True,
                        help='Model directory with training_params.json')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--n_augmentations', type=int, default=10,
                        help='Number of TTA iterations (default: 10)')
    parser.add_argument('--uncertainty', default='entropy',
                        choices=['entropy', 'variance', 'confidence',
                                 'mutual_information'],
                        help='Uncertainty type (default: entropy)')
    parser.add_argument('--quality_weights', default=None,
                        help='Path to quality prediction model weights')
    args = parser.parse_args()

    # Load training params
    json_path = os.path.join(args.model_dir, 'training_params.json')
    with open(json_path, 'r') as f:
        params = json.load(f)

    seg_labels = np.array(params['segmentation_labels'])
    topo_classes = np.array(params['generation_classes'])

    results = predict_with_tta(
        path_images=args.images,
        path_model=args.model,
        labels_segmentation=seg_labels,
        output_dir=args.output,
        n_neutral_labels=params.get('n_neutral_labels'),
        n_augmentations=args.n_augmentations,
        n_levels=params.get('n_levels', 5),
        nb_conv_per_level=params.get('nb_conv_per_level', 2),
        conv_size=params.get('conv_size', 3),
        unet_feat_count=params.get('unet_feat_count', 24),
        feat_multiplier=params.get('feat_multiplier', 2),
        activation=params.get('activation', 'elu'),
        target_res=params.get('target_res', 1.0),
        cropping=params.get('cropping'),
        sigma_smoothing=0.5,
        topology_classes=topo_classes,
        noise_std=params.get('noise_std', 100),
        bias_field_std=params.get('bias_field_std', 0.3),
        bias_scale=params.get('bias_scale', 0.025),
        gamma_std=0.5,
        uncertainty_type=args.uncertainty,
        quality_model_weights=args.quality_weights,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TTA UNCERTAINTY SUMMARY")
    print("=" * 60)
    for r in results:
        line = f"  {r['basename']}: uncertainty saved"
        if r['predicted_dice'] is not None:
            line += f", predicted Dice={r['predicted_dice']:.4f}"
        print(line)
