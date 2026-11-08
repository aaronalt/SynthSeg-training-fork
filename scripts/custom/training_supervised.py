"""
Supervised fine-tuning baseline for claustrum segmentation
#######
Version: supervised.dice.baseline
- Real images + ground-truth labels (no synthetic GMM generation)
- Online augmentation from SynthSeg/training_supervised.py (spatial, flip, bias field, intensity)
- Initialised from the Mauri UNet weights and fine-tuned end to end
- Weighted soft Dice loss (claustrum labels up-weighted)
- Validation Dice on real, un-augmented validation scans every epoch
- LR: reduce on plateau of validation claustrum Dice, down to 1e-6
#######
Built on SynthSeg (Billot et al., 2023, Medical Image Analysis). See
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import re
import json
import random
import datetime
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras import models, callbacks
from keras.optimizers import Adam

from SynthSeg.training_supervised import build_augmentation_model, build_model_inputs
from ext.lab2im import utils
from ext.neuron import models as nrn_models

tf.config.experimental.enable_tensor_float_32_execution(True)

# =====================================================================================
# CONFIGURATION
# =====================================================================================
SEED = 103
version = 'supervised.dice.baseline'
EXCLUDE_SUBJECTS = []

# Data. Raw images: one flat folder per split. Label maps: anywhere under LABEL_ROOT (searched recursively).
# Images and labels are paired by subject + hemisphere (+ session, if the image name has one) and contrast, e.g.
#   image  TRAIN_2/sub-6351.rh.crop.nii.gz           label  .../sub-6351_ses-T6_t1w_rh_training_label.nii.gz
#   image  TRAIN_2/sub-6351.t2w.cor.rh.crop.nii.gz   label  .../<name containing the t2w_cor label tag>
TRAIN = Path('/home/althause/data/TRAIN_3')
VAL = Path('/home/althause/data/VAL_3')
TEST = Path('/home/althause/data/TEST_2')
LABEL_ROOT = Path('/home/althause/data/claustrum_gt/3T_like_relabelled')

# Per contrast: text identifying it in the image filename, and in the label filename.
# Images containing none of the image tags are T1w. If T2 images share the T1w label maps, set their label tag to '_t1w_'.
CONTRAST_TAGS = {
    't1w':     {'image': None,        'label': '_t1w_'},
    't2w_cor': {'image': '.t2w.cor.', 'label': '_t2w_cor_'},   # TODO: check label tag
    't2w_tra': {'image': '.t2w.tra.', 'label': '_t2w_tra_'},   # TODO: check label tag
}
CONTRASTS = ['t1w']   # contrasts to train on, e.g. ['t1w', 't2w_cor', 't2w_tra']

# Pre-trained model to fine-tune
path_checkpoint = '/home/althause/data/weights/mauri_unet_weights.h5'

# Architecture (must match the Mauri model)
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
feat_multiplier = 2
activation = 'elu'

# Training
batchsize = 1
lr = 1e-4                 # initial learning rate
min_lr = 1e-6             # floor for the plateau schedule
plateau_factor = 0.5      # LR multiplier when validation Dice stalls
plateau_patience = 10     # epochs without improvement before reducing LR
dice_epochs = 200
steps_per_epoch = 1000

# Labels: 5 neutral, then 15 left, then 15 right (order required by RandomFlip)
segmentation_labels = np.array([0, 14, 15, 16, 24,
                                2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
CLAUSTRUM_LABELS = [138, 139]

label_weights = np.ones(len(segmentation_labels))
for lbl in CLAUSTRUM_LABELS:
    label_weights[np.where(segmentation_labels == lbl)[0][0]] = 10.0

# Shape. output_shape must be divisible by 2 ** n_levels (= 32).
output_shape = 192
val_crop_shape = [192, 192, 192]

# Augmentation (resolution randomisation intentionally off in this baseline)
flipping = True
scaling_bounds = 0.2
rotation_bounds = 15
shearing_bounds = 0.015
translation_bounds = 15
nonlin_std = 2.0
nonlin_scale = 0.04
bias_field_std = 0.3
bias_scale = 0.025


# =====================================================================================
# HELPERS
# =====================================================================================
def parse_name(name):
    """Subject, hemisphere and (optional) session from a filename."""
    sub = re.search(r'sub-([A-Za-z0-9]+)', name)
    hemi = re.search(r'(?:^|[._-])(lh|rh)(?=[._-])', name)
    ses = re.search(r'ses-([A-Za-z0-9]+)', name)
    if not (sub and hemi):
        return None
    return sub.group(1), hemi.group(1), ses.group(1) if ses else None


def image_contrast(name):
    for contrast, tags in CONTRAST_TAGS.items():
        if tags['image'] and tags['image'] in name:
            return contrast
    return next(c for c, tags in CONTRAST_TAGS.items() if tags['image'] is None)


def index_labels(root):
    """Map each (subject, hemisphere) to all label maps found for it under root."""
    index = {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            parsed = parse_name(f)
            if f.endswith(('.nii', '.nii.gz')) and parsed:
                index.setdefault(parsed[:2], []).append(os.path.join(dirpath, f))
    return index


def find_pairs(image_folder, label_index, contrasts):
    """Pair every image of the selected contrasts with its single matching label map."""
    images, labels = [], []
    for im in sorted(Path(image_folder).glob('*.nii*')):
        contrast = image_contrast(im.name)
        if contrast not in contrasts or any(s in im.name for s in EXCLUDE_SUBJECTS):
            continue
        parsed = parse_name(im.name)
        assert parsed, f'Cannot read subject/hemisphere from {im.name}'
        sub, hemi, ses = parsed
        matches = [l for l in label_index.get((sub, hemi), [])
                   if CONTRAST_TAGS[contrast]['label'] in os.path.basename(l)
                   and (ses is None or f'ses-{ses}' in os.path.basename(l))]
        assert len(matches) == 1, \
            f'{im.name} ({contrast}): {len(matches)} label maps found under {LABEL_ROOT}: {matches}'
        images.append(str(im))
        labels.append(matches[0])
    assert images, f'No images of contrasts {contrasts} found in {image_folder}'
    return images, labels


def check_dataset(images, labels, label_list, require_same_shape):
    """The supervised model has a fixed input shape, so all training volumes must share shape and resolution.
    Also checks that every label value is in the label list."""
    ref_shape, ref_res = None, None
    for im, lab in zip(images, labels):
        im_shape, _, _, n_ch, _, im_res = utils.get_volume_info(im, aff_ref=np.eye(4))
        lab_shape, _, _, _, _, lab_res = utils.get_volume_info(lab, aff_ref=np.eye(4))
        assert n_ch == 1, f'{im} has {n_ch} channels, expected 1'
        assert list(im_shape) == list(lab_shape), f'Shape mismatch: {im} {im_shape} vs {lab} {lab_shape}'
        assert np.allclose(im_res, lab_res, atol=1e-3), f'Resolution mismatch: {im} vs {lab}'
        if require_same_shape:
            if ref_shape is None:
                ref_shape, ref_res = list(im_shape), np.array(im_res)
            assert list(im_shape) == ref_shape, \
                f'{im} has shape {im_shape}, expected {ref_shape}. Crop/pad all training volumes to one shape.'
            assert np.allclose(im_res, ref_res, atol=1e-3), f'{im} resolution {im_res} differs from {ref_res}'
        unknown = np.setdiff1d(np.unique(utils.load_volume(lab, dtype='int32')), label_list)
        assert unknown.size == 0, f'{lab} contains labels not in segmentation_labels: {unknown}. Map them to 0 first.'


def min_max_normalise(x):
    """Matches IntensityAugmentation(normalise=True) used during training."""
    return (x - x.min()) / (x.max() - x.min() + 1e-7)


def crop_or_pad(vol, shape):
    """Centre crop or zero-pad each spatial axis to the requested shape."""
    out = vol
    for ax, s in enumerate(shape):
        n = out.shape[ax]
        if n > s:
            start = (n - s) // 2
            out = np.take(out, np.arange(start, start + s), axis=ax)
        elif n < s:
            pad = [(0, 0)] * out.ndim
            pad[ax] = ((s - n) // 2, s - n - (s - n) // 2)
            out = np.pad(out, pad)
    return out


def load_pretrained(unet_model, path):
    """Load weights by layer name and fail loudly if any UNet layer is not found in the checkpoint.
    Keras by_name loading raises on shape mismatches (e.g. wrong label count) but silently skips missing names."""
    with h5py.File(path, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        saved = [n.decode() if isinstance(n, bytes) else n for n in g.attrs['layer_names']]
    unet_layers = [l.name for l in unet_model.layers if l.weights and l.name.startswith('unet')]
    missing = [n for n in unet_layers if n not in saved]
    assert not missing, f'These UNet layers are missing from the checkpoint: {missing}'
    unet_model.load_weights(path, by_name=True)
    print(f'Loaded {len(unet_layers)} UNet layers from {path}')


def build_weighted_dice_model(unet_model, label_list, weights):
    """Soft Dice (Milletari et al., 2016, squared denominator as in SynthSeg), averaged across labels with weights."""
    labels_gt = unet_model.get_layer('labels_out').output
    pred = unet_model.outputs[0]
    lab_np = np.array(label_list, dtype='int32')
    w_np = (np.array(weights) / np.sum(weights)).astype('float32')

    def dice_loss(x):
        gt, p = x
        gt_onehot = tf.cast(tf.equal(gt, tf.constant(lab_np)), 'float32')   # [B, X, Y, Z, n_labels]
        axes = [1, 2, 3]
        top = 2 * tf.reduce_sum(gt_onehot * p, axis=axes)
        bottom = tf.reduce_sum(tf.square(gt_onehot) + tf.square(p), axis=axes)
        dice = top / tf.maximum(bottom, 1e-7)
        return 1 - tf.reduce_sum(dice * tf.constant(w_np), axis=-1, keepdims=True)

    loss = KL.Lambda(dice_loss, name='weighted_dice_loss')([labels_gt, pred])
    return models.Model(unet_model.inputs, loss)


class ValidationDice(callbacks.Callback):
    """Each epoch: copy the current UNet weights into an un-augmented UNet, segment the validation scans,
    and log hard Dice for the claustrum labels. Saves best and last UNet weights (loadable by SynthSeg predict)."""

    def __init__(self, train_unet, images, labels, label_list, eval_labels, crop_shape, unet_kwargs, model_dir):
        super().__init__()
        self.train_unet = train_unet
        self.label_list = np.array(label_list)
        self.eval_labels = eval_labels
        self.model_dir = model_dir
        self.best = -np.inf
        self.val_unet = nrn_models.unet(input_shape=list(crop_shape) + [1], name='unet', **unet_kwargs)
        self.data = []
        for im_path, lab_path in zip(images, labels):
            im = min_max_normalise(utils.load_volume(im_path, aff_ref=np.eye(4), dtype='float32'))
            lab = utils.load_volume(lab_path, aff_ref=np.eye(4), dtype='int32')
            lab_c = crop_or_pad(lab, crop_shape)
            for l in eval_labels:
                if np.sum(lab == l) != np.sum(lab_c == l):
                    print(f'WARNING: centre crop removes part of label {l} in {lab_path}; enlarge val_crop_shape.')
            self.data.append((crop_or_pad(im, crop_shape), lab_c))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        for layer in self.val_unet.layers:
            if layer.weights:
                layer.set_weights(self.train_unet.get_layer(layer.name).get_weights())
        scores = {l: [] for l in self.eval_labels}
        for im, lab in self.data:
            post = self.val_unet.predict(im[np.newaxis, ..., np.newaxis], verbose=0)[0]
            seg = self.label_list[np.argmax(post, axis=-1)]
            for l in self.eval_labels:
                a, b = seg == l, lab == l
                denom = a.sum() + b.sum()
                if denom > 0:
                    scores[l].append(2 * np.logical_and(a, b).sum() / denom)
        for l in self.eval_labels:
            logs[f'val_dice_{l}'] = float(np.mean(scores[l]))
        mean_dice = float(np.mean([logs[f'val_dice_{l}'] for l in self.eval_labels]))
        logs['val_claustrum_dice'] = mean_dice
        self.val_unet.save(os.path.join(self.model_dir, 'unet_last.h5'))
        if mean_dice > self.best:
            self.best = mean_dice
            self.val_unet.save(os.path.join(self.model_dir, 'unet_best.h5'))
            print(f'\nEpoch {epoch + 1}: new best validation claustrum Dice {mean_dice:.4f}')


# =====================================================================================
# SETUP
# =====================================================================================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{version}"
path_model_dir = os.path.join('./models/test', experiment_name)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

label_index = index_labels(LABEL_ROOT)
train_images, train_labels = find_pairs(TRAIN, label_index, CONTRASTS)
val_images, val_labels = find_pairs(VAL, label_index, CONTRASTS)
test_images = sorted(str(p) for p in TEST.glob('*.nii*'))

check_dataset(train_images, train_labels, segmentation_labels, require_same_shape=True)
check_dataset(val_images, val_labels, segmentation_labels, require_same_shape=False)

with open(os.path.join(path_model_dir, 'config.json'), 'w') as f:
    json.dump({'experiment': experiment_name, 'version': version, 'seed': SEED,
               'checkpoint': path_checkpoint, 'label_root': str(LABEL_ROOT), 'contrasts': CONTRASTS,
               'contrast_tags': {c: CONTRAST_TAGS[c] for c in CONTRASTS},
               'lr': lr, 'min_lr': min_lr,
               'plateau_factor': plateau_factor, 'plateau_patience': plateau_patience,
               'epochs': dice_epochs, 'steps_per_epoch': steps_per_epoch,
               'label_weights': label_weights.tolist(), 'output_shape': output_shape,
               'augmentation': {'flipping': flipping, 'scaling_bounds': scaling_bounds,
                                'rotation_bounds': rotation_bounds, 'shearing_bounds': shearing_bounds,
                                'translation_bounds': translation_bounds, 'nonlin_std': nonlin_std,
                                'nonlin_scale': nonlin_scale, 'bias_field_std': bias_field_std,
                                'bias_scale': bias_scale},
               'train_images': train_images, 'train_labels': train_labels,
               'val_images': val_images, 'val_labels': val_labels, 'test_images': test_images},
              f, indent=2)

# =====================================================================================
# MODEL
# =====================================================================================
im_shape, _, _, n_channels, _, atlas_res = utils.get_volume_info(train_images[0], aff_ref=np.eye(4))
augmentation_model = build_augmentation_model(im_shape, n_channels, segmentation_labels, n_neutral_labels,
                                              atlas_res, target_res=None,
                                              output_shape=output_shape,
                                              output_div_by_n=2 ** n_levels,
                                              flipping=flipping,
                                              aff=np.eye(4),
                                              scaling_bounds=scaling_bounds,
                                              rotation_bounds=rotation_bounds,
                                              shearing_bounds=shearing_bounds,
                                              translation_bounds=translation_bounds,
                                              nonlin_std=nonlin_std,
                                              nonlin_scale=nonlin_scale,
                                              randomise_res=False,
                                              data_res=None,
                                              bias_field_std=bias_field_std,
                                              bias_scale=bias_scale)
unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

unet_kwargs = dict(nb_labels=len(segmentation_labels), nb_levels=n_levels, nb_conv_per_level=nb_conv_per_level,
                   conv_size=conv_size, nb_features=unet_feat_count, feat_mult=feat_multiplier,
                   activation=activation, batch_norm=-1)
unet_model = nrn_models.unet(input_model=augmentation_model, input_shape=unet_input_shape,
                             name='unet', **unet_kwargs)
load_pretrained(unet_model, path_checkpoint)

train_model = build_weighted_dice_model(unet_model, segmentation_labels, label_weights)
train_model.compile(optimizer=Adam(learning_rate=lr), loss=lambda y_true, y_pred: y_pred)

generator = build_model_inputs(train_images, train_labels, batchsize, subjects_prob=None)
input_generator = utils.build_training_generator(generator, batchsize)

# Callback order matters: ValidationDice writes val_claustrum_dice into logs before the others read it.
cbs = [ValidationDice(unet_model, val_images, val_labels, segmentation_labels, CLAUSTRUM_LABELS,
                      val_crop_shape, unet_kwargs, path_model_dir),
       callbacks.ReduceLROnPlateau(monitor='val_claustrum_dice', mode='max', factor=plateau_factor,
                                   patience=plateau_patience, min_lr=min_lr, verbose=1),
       callbacks.CSVLogger(os.path.join(log_dir, 'training_log.csv')),
       callbacks.TensorBoard(log_dir=log_dir)]

# =====================================================================================
# TRAIN
# =====================================================================================
train_model.fit(input_generator, epochs=dice_epochs, steps_per_epoch=steps_per_epoch, callbacks=cbs)