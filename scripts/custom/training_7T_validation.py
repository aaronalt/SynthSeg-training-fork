"""
SynthSeg training script for claustrum segmentation
NEW APPROACH: 80% 7T train, 20% 7T validation + selected high-quality 3T validation
"""

import os
import tensorflow as tf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import datetime
import re
import json
import numpy as np
from pathlib import Path
from create_train_test_split_stratified import create_train_test_split
from training_feature_extraction import training
from SynthSeg.estimate_priors import build_intensity_stats
import shutil

# === CONFIGURATION ===
# High-quality 3T subjects to add to validation (manual curation)
HIGH_QUALITY_3T_VALIDATION = ['sub-7796', 'sub-8510']

# Subjects to exclude from training (poor labels)
EXCLUDE_SUBJECTS = []  # Add any bad subjects here

# Experiment setup
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Split directories
split_dir_7T = f'/home/althause/data/training_split_7T/{experiment_name}'
split_dir_3T = f'/home/althause/data/training_split_3T/{experiment_name}'
os.makedirs(split_dir_7T, exist_ok=True)
os.makedirs(split_dir_3T, exist_ok=True)

print("\n" + "="*70)
print("NEW TRAINING APPROACH: 7T-focused validation")
print("="*70)

# === STEP 1: Split 7T data (80% train, 20% validation) ===
print("\n--- Step 1: Splitting 7T data (80% train, 20% validation) ---")
train_path_7T, val_path_7T, _, val_probs_7T = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/7T/training_labels_native',
            'subdirs': ['t1w', 't2w-cor', 't2w-tra'],
            'field_strength': '7T',
        },
    ],
    output_dir=split_dir_7T,
    method='symlink',
    test_ratio=0.2,  # 20% validation
    seed=42,
    prob_mode='uniform',
    verbose=True,
)

# Get 7T validation subject IDs
with open(os.path.join(split_dir_7T, 'split_info.json')) as f:
    split_info_7T = json.load(f)
val_subjects_7T = set(split_info_7T['test_subjects'])
train_subjects_7T = set(split_info_7T['train_subjects'])

print(f"7T training subjects ({len(train_subjects_7T)}): {sorted(train_subjects_7T)}")
print(f"7T validation subjects ({len(val_subjects_7T)}): {sorted(val_subjects_7T)}")
print(f"7T training path: {train_path_7T}")
print(f"7T validation path: {val_path_7T}")

# === STEP 2: Add high-quality 3T subjects to validation ===
print("\n--- Step 2: Adding high-quality 3T subjects to validation ---")
print(f"High-quality 3T subjects for validation: {HIGH_QUALITY_3T_VALIDATION}")

# Create combined validation directory
combined_val_dir = os.path.join(split_dir_7T, 'val_combined')
os.makedirs(combined_val_dir, exist_ok=True)

# Copy 7T validation files
val_7T_path = Path(val_path_7T)
n_7T_val = 0
for f in sorted(val_7T_path.glob('*.nii.gz')):
    dest = os.path.join(combined_val_dir, f.name)
    if not os.path.exists(dest):
        os.symlink(f.resolve(), dest)
        n_7T_val += 1

# Add high-quality 3T validation files
source_3T_dir = Path('/home/althause/data/3T/training_labels_native/t1w')
n_3T_val = 0
for f in sorted(source_3T_dir.glob('*.nii.gz')):
    sub_match = re.match(r'(sub-\d+)', f.name)
    if sub_match and sub_match.group(1) in HIGH_QUALITY_3T_VALIDATION:
        dest = os.path.join(combined_val_dir, f'3T_{f.name}')
        if not os.path.exists(dest):
            os.symlink(f.resolve(), dest)
            n_3T_val += 1

print(f"Combined validation set: {n_7T_val} 7T files + {n_3T_val} 3T files = {n_7T_val + n_3T_val} total")

# Create validation probability distribution (uniform across all val files)
# BrainGenerator expects a numpy array with one probability per file (not per subject)
val_subjects_combined = list(val_subjects_7T) + HIGH_QUALITY_3T_VALIDATION
n_total_val_files = n_7T_val + n_3T_val
val_probs_combined = np.ones(n_total_val_files, dtype='float32') / n_total_val_files  # Uniform

# === STEP 3: Training data = 80% 7T only (no 3T in training) ===
print("\n--- Step 3: Training set = 80% 7T ---")
all_train_paths = train_path_7T
n_train = len(list(Path(train_path_7T).glob('*.nii.gz')))
print(f"Training set: {n_train} files (7T only)")

all_val_paths = combined_val_dir
val_subjects_prob = val_probs_combined

# Save split info
split_summary = {
    'experiment': experiment_name,
    'train_7T_subjects': sorted(train_subjects_7T),
    'val_7T_subjects': sorted(val_subjects_7T),
    'val_3T_subjects': HIGH_QUALITY_3T_VALIDATION,
    'n_train_files': n_train,
    'n_val_7T_files': n_7T_val,
    'n_val_3T_files': n_3T_val,
}
with open(os.path.join(path_model_dir, 'split_summary.json'), 'w') as f:
    json.dump(split_summary, f, indent=2)

print("\n" + "="*70)
print("TRAINING CONFIGURATION SUMMARY")
print("="*70)
print(f"Training: {n_train} files (80% 7T)")
print(f"Validation: {n_7T_val} 7T + {n_3T_val} 3T = {n_7T_val + n_3T_val} files")
print(f"Validation subjects: {sorted(val_subjects_combined)}")
print("="*70 + "\n")

# === MODEL PARAMETERS ===
# Pre-trained model
path_checkpoint = '/home/althause/data/weights/mauri_unet_weights.h5'
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
dice_epochs = 100
steps_per_epoch = 10000
validation_steps = 60  # ~20 validation subjects × 3 modalities = 60 files (1 pass)

# Generation and segmentation labels
path_generation_labels = np.array([0, 14, 15, 16, 24,
                                   2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
path_segmentation_labels = path_generation_labels.copy()

# Label weights for Dice loss (Claustrum=10, others=1)
label_weights = np.ones(len(path_segmentation_labels))
for lbl in [138, 139]:
    label_weights[np.where(path_segmentation_labels == lbl)[0][0]] = 10.0

# Shape and resolution
target_res = 0.50
output_shape = 192
n_channels = 1

# GMM sampling
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
nonlin_std = 2.0
bias_field_std = 0.3
noise_std = 100

# Acquisition resolution parameters
randomise_res = True
max_res_iso = 1.5
max_res_aniso = 2.0
data_res = None
thickness = None

# Load intensity priors
path_mri = '/home/althause/data/intensity_estimation/images'
means = np.load(os.path.join(path_mri, 'prior_means.npy'))
stds = np.load(os.path.join(path_mri, 'prior_stds.npy'))

# === START TRAINING ===
print("\nStarting training...\n")
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
         noise_std=noise_std,
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
         subjects_prob=None,  # Equal probability for 7T training set
         val_subjects_prob=val_subjects_prob,
         data_res=data_res,
         thickness=thickness,
         max_res_iso=max_res_iso,
         max_res_aniso=max_res_aniso,
         prior_means=means,
         prior_stds=stds,
         label_weights=label_weights
         )
