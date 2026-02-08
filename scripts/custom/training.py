"""
SynthSeg training script for claustrum segmentation
Optimized for NVIDIA RTX 2000 Ada with CUDA 12.x and TensorFlow 2.15
"""


import os
import tensorflow as tf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
'''
# Configure GPU for TensorFlow 2.15
print("=== GPU Configuration ===")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
'''
import datetime
import re
import json
import numpy as np
from pathlib import Path
from create_train_test_split_stratified import create_train_test_split, extract_test_from_validation
from training_feature_extraction import training
from SynthSeg.estimate_priors import build_intensity_stats

# Paths
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# === DATA SETUP ===
import shutil
from ext.lab2im import utils as lab2im_utils

# Per-experiment split directories to avoid conflicts between concurrent runs
split_dir_3T = f'/home/althause/data/training_split_3T/{experiment_name}'
split_dir_main = f'/home/althause/data/training_split/{experiment_name}'
filtered_3T_base = Path(f'/home/althause/data/3T/training_labels_native_filtered/{experiment_name}')

# Step 1: Split 3T data — 10% for training, 90% for validation
train_path_3T, val_path_3T, _, val_probs_3T = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/3T/training_labels_native',
            'subdirs': ['t1w'],
            'field_strength': '3T',
        },
    ],
    output_dir=split_dir_3T,
    method='symlink',
    test_ratio=0.9,
    seed=42,
    prob_mode='uniform',
    verbose=True,
)

# Get 3T validation subject IDs to exclude from main training split
with open(os.path.join(split_dir_3T, 'split_info.json')) as f:
    split_info_3T = json.load(f)
val_subjects_3T = set(split_info_3T['test_subjects'])
print(f"3T validation subjects (excluded from training): {val_subjects_3T}")
print(f"3T validation set: {val_path_3T}")

# Step 2: Create filtered 3T directory excluding validation subjects
filtered_3T_dir = filtered_3T_base / 't1w'
if filtered_3T_base.exists():
    shutil.rmtree(filtered_3T_base)
filtered_3T_dir.mkdir(parents=True)

source_3T_dir = Path('/home/althause/data/3T/training_labels_native/t1w')
n_included = 0
n_excluded = 0
for f in sorted(source_3T_dir.glob('*.nii.gz')):
    sub_match = re.match(r'(sub-\d+)', f.name)
    if sub_match and sub_match.group(1) in val_subjects_3T:
        n_excluded += 1
    else:
        os.symlink(f.resolve(), filtered_3T_dir / f.name)
        n_included += 1
print(f"3T filtered: {n_included} train files, {n_excluded} val files excluded")

# Step 3: Combine all 7T + 10% 3T train into one training directory
combined_train_dir = os.path.join(split_dir_main, 'train_combined')
os.makedirs(combined_train_dir, exist_ok=True)

# Add all 7T files (no holdout — only testing on 3T)
for subdir in ['t1w', 't2w-cor', 't2w-tra']:
    subdir_path = Path(f'/home/althause/data/7T/training_labels_native/{subdir}')
    if not subdir_path.exists():
        continue
    for f in sorted(subdir_path.glob('*.nii.gz')):
        dest = os.path.join(combined_train_dir, f'7T_{subdir}_{f.name}')
        if not os.path.exists(dest):
            os.symlink(f.resolve(), dest)

for f in sorted(filtered_3T_dir.glob('*.nii.gz')):
    dest = os.path.join(combined_train_dir, f'3T_{f.name}')
    if not os.path.exists(dest):
        os.symlink(f.resolve(), str(dest))

n_combined = len(os.listdir(combined_train_dir))
print(f"Combined training: {n_combined} files (7T train + 10% 3T train subjects)")

all_train_paths = combined_train_dir
all_val_paths = val_path_3T  # Validate on 90% held-out 3T subjects
val_subjects_prob = val_probs_3T

# Pre-trained model
path_checkpoint = '/home/althause/data/weights/mauri_unet_weights.h5'  # Original pretrained weights
# path_checkpoint = '/home/althause/SynthSeg-training-fork/models/test/experiment_20260203_184041/dice_finetune_020_20.h5'  # Resume from last checkpoint
batchsize = 1

# Architecture parameters
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2

# Training parameters
lr = 1e-4
wl2_epochs = 0  # Skip warmup - using pretrained weights
dice_epochs = 100  # Train longer to learn small structures
steps_per_epoch = 5000
validation_steps = 200  # More steps for stable validation with small 3T set

# Generation and segmentation labels
# Use ALL labels for segmentation - model learns full anatomy, extract claustrum at inference
path_generation_labels = np.array([0, 14, 15, 16, 24,
                                   2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
# Keep all labels distinct - no collapsing to background
path_segmentation_labels = path_generation_labels.copy()

# === LABEL WEIGHTS FOR DICE LOSS ===
# Higher weight = more importance during training
# Claustrum labels (138=LH, 139=RH) weighted more heavily
label_weights = np.ones(len(path_segmentation_labels))
claustrum_weight = 10.0

# Find claustrum indices
lh_claustrum_idx = np.where(path_segmentation_labels == 138)[0][0]
rh_claustrum_idx = np.where(path_segmentation_labels == 139)[0][0]
label_weights[lh_claustrum_idx] = claustrum_weight
label_weights[rh_claustrum_idx] = claustrum_weight
print(f"Claustrum weight: {claustrum_weight}x at indices {lh_claustrum_idx} (LH=138), {rh_claustrum_idx} (RH=139)")

# Shape and resolution
target_res = 0.50
output_shape = 192  # Resampled at runtime to 192³ @ 0.5mm (96mm FOV)
n_channels = 1

# GMM sampling - maps each generation label to an intensity class
# Max index must be < number of prior classes (19 classes = indices 0-18)
prior_distributions = 'normal'
path_generation_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Spatial deformation parameters
flipping = True
scaling_bounds = 0.2
rotation_bounds = 15
shearing_bounds = 0.012
translation_bounds = False
nonlin_std = 4.0
bias_field_std = 0.7

# Acquisition resolution parameters
randomise_res = True
max_res_iso = 1.5
max_res_aniso = 2.0
data_res = None
thickness = None

path_mri = '/home/althause/data/intensity_estimation/images'
path_training_labels = '/home/althause/data/intensity_estimation/labels'
'''
means, stds = build_intensity_stats(path_mri, path_training_labels, path_mri,
                                    estimation_labels=path_generation_labels,
                                    estimation_classes=path_generation_classes
                                    )
'''
means = np.load(os.path.join(path_mri, 'prior_means.npy'))
stds = np.load(os.path.join(path_mri, 'prior_stds.npy'))

# Start training - pretrain
skip_pretrain = False
training(all_train_paths,
         path_model_dir,
         generation_labels=path_generation_labels,
         segmentation_labels=path_segmentation_labels,
         n_neutral_labels=n_neutral_labels,
         batchsize=batchsize,
         n_channels=n_channels,
         target_res=target_res,
         output_shape=output_shape,
         prior_distributions=prior_distributions,
         generation_classes=path_generation_classes,
         flipping=flipping,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         randomise_res=randomise_res,
         bias_field_std=bias_field_std,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         wl2_epochs=wl2_epochs,
         dice_epochs=dice_epochs,
         steps_per_epoch=steps_per_epoch,
         validation_steps=validation_steps,
         checkpoint=path_checkpoint,
         val_path=all_val_paths,
         skip_pretrain=skip_pretrain,
         finetune=False,
         subjects_prob=None,  # Equal probability for combined 7T+3T training set
         val_subjects_prob=val_subjects_prob,
         data_res=data_res,
         thickness=thickness,
         max_res_iso=max_res_iso,
         max_res_aniso=max_res_aniso,
         prior_means=means,
         prior_stds=stds,
         label_weights=label_weights
         )
