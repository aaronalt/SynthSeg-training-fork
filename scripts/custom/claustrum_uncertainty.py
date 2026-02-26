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
import keras.layers as KL
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from scipy.ndimage import zoom
from pathlib import Path

from SynthSeg.predict import preprocess, build_model, postprocess, get_flip_indices
from ext.lab2im import utils, edit_volumes, layers

keras.backend.set_image_data_format('channels_last')

CLAUSTRUM_LABELS = [138, 139]  # LH, RH


# ===================================================================
# SECTION 1: TTA AUGMENTATION MODEL (using actual SynthSeg layers)
# ===================================================================

def build_tta_augmentation_model(input_shape, bias_field_std=0.3,
                                 bias_scale=0.025, noise_std=100,
                                 gamma_std=0.5):
    """
    Build a Keras model that applies the same SynthSeg intensity
    augmentation layers used during training.

    Chain (mirrors labels_to_image_model.py lines 186-195):
        1. BiasFieldCorruption  — smooth multiplicative bias field
        2. GaussianNoiseCorruption — voxel-wise Gaussian noise
        3. IntensityAugmentation — clip, min-max normalise, gamma

    Because the input is already normalised to [0, 1] (by predict's
    preprocess), GaussianNoiseCorruption's noise_std is scaled by
    1/300 to match the effective noise level after training's own
    normalisation step (training clips to 300 then normalises).

    Each call to model.predict() produces a *different* augmentation
    because every layer samples fresh random values internally.

    Args:
        input_shape: e.g. [None, None, None, 1]  (dynamic spatial dims)
        bias_field_std: max std dev for the bias field (same as training)
        bias_scale: ratio for the small-field sampling grid
        noise_std: raw noise std from training (will be /300 internally)
        gamma_std: std dev of log-gamma exponent

    Returns:
        Keras Model  (input → augmented output, same shape)
    """
    image_input = KL.Input(shape=input_shape, name='tta_input')

    # 1. Bias field corruption (multiplicative — scale-invariant)
    x = layers.BiasFieldCorruption(
        bias_field_std=bias_field_std,
        bias_scale=bias_scale,
        same_bias_for_all_channels=False,
        prob=1.0)(image_input)

    # 2. Gaussian noise corruption (scale noise_std for [0,1] images)
    x = layers.GaussianNoiseCorruption(
        max_noise_std=noise_std / 300.0,
        prob=0.95)(x)

    # 3. Intensity augmentation: clip + normalise back to [0,1] + gamma
    #    noise_std=0 here (noise already applied above), clip=0 → no clip,
    #    normalise=True re-maps to [0,1] after bias+noise shifts range,
    #    gamma_std controls contrast augmentation strength.
    x = layers.IntensityAugmentation(
        noise_std=0,
        clip=0,
        normalise=True,
        norm_perc=0,
        gamma_std=gamma_std,
        separate_channels=True,
        prob_noise=0,
        prob_gamma=1.0)(x)

    return Model(inputs=image_input, outputs=x, name='tta_augmentation')


# ===================================================================
# SECTION 2: TTA POSTERIOR GENERATION & UNCERTAINTY COMPUTATION
# ===================================================================

def generate_tta_posteriors(image_preprocessed, seg_net, aug_model,
                            n_augmentations=10):
    """
    Run N augmented forward passes through the segmentation model.

    Args:
        image_preprocessed: (1, H, W, D, 1) preprocessed image
        seg_net: Trained SynthSeg Keras model (may include flip averaging)
        aug_model: TTA augmentation Keras model (built by
                   build_tta_augmentation_model)
        n_augmentations: Total predictions including the un-augmented one

    Returns:
        np.ndarray of shape (N, H, W, D, n_labels) — squeezed posteriors
    """
    all_posteriors = []

    # 1. Original (un-augmented) prediction
    post = seg_net.predict(image_preprocessed, verbose=0)
    all_posteriors.append(np.squeeze(post))

    # 2. Augmented predictions — each aug_model.predict() call samples
    #    fresh random augmentation parameters via the SynthSeg layers
    for _ in range(n_augmentations - 1):
        aug = aug_model.predict(image_preprocessed, verbose=0)
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


# ===================================================================
# SECTION 3: QUALITY PREDICTION — 2-SUBNETWORK CNN (3D)
# ===================================================================

def build_quality_model(roi_size=(64, 64, 64)):
    """
    Build a 2-branch 3D CNN for Dice quality prediction from
    (claustrum segmentation map, uncertainty map) inputs.

    Architecture mirrors the 2D model from Sikha et al. (2025),
    adapted for 3D with same padding to preserve spatial dims.

    Branches: Conv3D(64)->Pool -> Conv3D(64)->Pool -> Conv3D(32)->Pool
              -> Conv3D(32)->Pool -> Conv3D(16)
    Merge -> Flatten -> Dense(128) -> Dense(128) -> Dense(1, linear)

    Args:
        roi_size: spatial dimensions of input ROI (H, W, D)

    Returns:
        Compiled Keras Model
    """
    from keras.layers import (Input, Conv3D, MaxPool3D,
                              Flatten, Dense, concatenate)

    input_shape = (*roi_size, 1)
    filters = [64, 64, 32, 32, 16]

    seg_input = Input(shape=input_shape, name='segmentation_input')
    unc_input = Input(shape=input_shape, name='uncertainty_input')

    def branch(x):
        for i, nf in enumerate(filters):
            x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
            if i < len(filters) - 1:  # MaxPool on all but last layer
                x = MaxPool3D(pool_size=(2, 2, 2))(x)
        return x

    seg_branch = branch(seg_input)
    unc_branch = branch(unc_input)

    merged = concatenate([seg_branch, unc_branch])
    flat = Flatten()(merged)
    dense = Dense(128, activation='relu')(flat)
    dense = Dense(128, activation='relu')(dense)
    output = Dense(1, activation='linear', name='dice_prediction')(dense)

    model = Model(inputs=[seg_input, unc_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def extract_claustrum_roi(volume, seg_map, roi_size=(64, 64, 64),
                          claustrum_labels=None, padding=10):
    """
    Crop and resize a region around the predicted claustrum.

    Args:
        volume: 3D array (H, W, D) — the map to crop
        seg_map: Segmentation map of same spatial shape
        roi_size: Fixed output size
        claustrum_labels: Labels identifying claustrum
        padding: Voxels of context around bounding box

    Returns:
        Cropped and resized volume of shape roi_size
    """
    if claustrum_labels is None:
        claustrum_labels = CLAUSTRUM_LABELS

    mask = np.isin(seg_map, claustrum_labels)

    if mask.any():
        coords = np.where(mask)
        mins = [max(0, int(c.min()) - padding) for c in coords]
        maxs = [min(s, int(c.max()) + padding + 1)
                for c, s in zip(coords, volume.shape[:3])]
        cropped = volume[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    else:
        # Fallback: center crop
        center = [s // 2 for s in volume.shape[:3]]
        slices = tuple(
            slice(max(0, c - r // 2), min(s, c + r // 2))
            for c, r, s in zip(center, roi_size, volume.shape[:3]))
        cropped = volume[slices]

    # Resize to fixed ROI
    if cropped.shape[:3] != roi_size:
        zf = [r / max(c, 1) for r, c in zip(roi_size, cropped.shape[:3])]
        cropped = zoom(cropped.astype(np.float32), zf, order=1)

    return cropped.astype(np.float32)


def train_quality_model(seg_maps, uncertainty_maps, dice_scores,
                        roi_size=(64, 64, 64), epochs=50, batch_size=4,
                        val_split=0.2, save_path=None):
    """
    Train the 2-subnetwork quality prediction model.

    Args:
        seg_maps: list of 3D binary claustrum segmentation arrays
        uncertainty_maps: list of 3D uncertainty arrays (same shape)
        dice_scores: list/array of ground-truth Dice scores
        roi_size: ROI size (must match build_quality_model)
        epochs, batch_size, val_split: training hyperparameters
        save_path: path to save trained weights (.h5)

    Returns:
        Trained Keras Model
    """
    from keras.callbacks import EarlyStopping

    model = build_quality_model(roi_size)

    X_seg = np.array(seg_maps)[..., np.newaxis]
    X_unc = np.array(uncertainty_maps)[..., np.newaxis]
    y = np.array(dice_scores, dtype=np.float32)

    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)

    model.fit([X_seg, X_unc], y,
              batch_size=batch_size, epochs=epochs,
              validation_split=val_split, callbacks=[early_stop], verbose=1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_weights(save_path)
        print(f"Quality model saved to {save_path}")

    return model


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
                     uncertainty_type='entropy',
                     quality_model_weights=None,
                     quality_roi_size=(64, 64, 64)):
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

    # Build TTA augmentation model (uses actual SynthSeg layers)
    aug_model = build_tta_augmentation_model(
        input_shape=model_input_shape,
        bias_field_std=bias_field_std,
        bias_scale=bias_scale,
        noise_std=noise_std,
        gamma_std=gamma_std)
    print(f"Built TTA augmentation model "
          f"(bias_field_std={bias_field_std}, noise_std={noise_std}, "
          f"gamma_std={gamma_std})")

    # Load quality prediction model if provided
    quality_model = None
    if quality_model_weights and os.path.isfile(quality_model_weights):
        quality_model = build_quality_model(quality_roi_size)
        quality_model.load_weights(quality_model_weights)
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

        # TTA forward passes using SynthSeg augmentation layers
        all_posteriors = generate_tta_posteriors(
            image, seg_net, aug_model, n_augmentations)

        # Compute claustrum uncertainty
        unc_results = compute_claustrum_uncertainty(
            all_posteriors, labels_segmentation)

        # Postprocess mean posteriors -> segmentation
        mean_post = unc_results['mean_posteriors_all']
        seg, posteriors, volumes = postprocess(
            post_patch=mean_post[np.newaxis],
            shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
            n_dims=n_dims, labels_segmentation=labels_segmentation,
            keep_biggest_component=keep_biggest_component,
            aff=aff, im_res=im_res, topology_classes=topology_classes)

        # Map uncertainty to original image space
        unc_map = unc_results[uncertainty_type]
        unc_map = edit_volumes.crop_volume_with_idx(
            unc_map, pad_idx, n_dims=3, return_copy=False)
        if crop_idx is not None:
            full_unc = np.zeros(shape, dtype=np.float32)
            full_unc[crop_idx[0]:crop_idx[3],
                     crop_idx[1]:crop_idx[4],
                     crop_idx[2]:crop_idx[5]] = unc_map
            unc_map = full_unc
        unc_map = edit_volumes.align_volume_to_ref(
            unc_map, np.eye(4), aff, n_dims=n_dims, return_copy=False)

        # Save outputs
        seg_path = os.path.join(
            tta_seg_dir, f'{basename}_tta_seg.nii.gz')
        unc_path = os.path.join(
            unc_dir, f'{basename}_{uncertainty_type}.nii.gz')
        utils.save_volume(seg.astype('int32'), aff, h, seg_path)
        utils.save_volume(unc_map.astype('float32'), aff, h, unc_path)

        result = dict(
            path=img_path, basename=basename,
            seg_path=seg_path, uncertainty_path=unc_path,
            uncertainty_type=uncertainty_type,
            volumes=volumes, predicted_dice=None,
        )

        # Quality prediction (if model available)
        if quality_model is not None:
            cl_seg_binary = np.isin(seg, CLAUSTRUM_LABELS).astype(np.float32)
            seg_roi = extract_claustrum_roi(
                cl_seg_binary, seg, quality_roi_size)
            unc_roi = extract_claustrum_roi(
                unc_map, seg, quality_roi_size)
            pred_dice = quality_model.predict(
                [seg_roi[np.newaxis, ..., np.newaxis],
                 unc_roi[np.newaxis, ..., np.newaxis]], verbose=0)
            result['predicted_dice'] = float(pred_dice[0, 0])
            print(f"    Predicted Dice: {result['predicted_dice']:.4f}")

        results.append(result)

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
