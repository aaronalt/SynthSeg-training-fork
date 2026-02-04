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
import numpy as np
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

# 3T training/validation files (4 non-VCFS subjects - used for BOTH training and validation)
trainval_3T_dir = '/home/althause/data/3T/training_labels_1mm/t1w'  # 1mm isotropic, 128³
# 3T test files are in holdout dir (8 VCFS subjects - NOT used in training)

# Subject-aware stratified split for 7T data
train_path_7T, _, subjects_prob_7T, _ = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/7T/training_labels_1mm',
            'subdirs': ['t1w', 't2w-cor', 't2w-tra'],
            'field_strength': '7T',
        },
    ],
    output_dir='/home/althause/data/training_split',
    method='symlink',
    test_ratio=0.3,
    seed=42,
    prob_mode='sqrt',
    verbose=True,
)

# Get 3T files
trainval_files_3T = lab2im_utils.list_images_in_folder(trainval_3T_dir)
print(f"3T data: {len(trainval_files_3T)} files for training AND validation")

# Create combined training directory (7T + 3T)
combined_train_dir = '/home/althause/data/training_split/train_combined'
if os.path.exists(combined_train_dir):
    shutil.rmtree(combined_train_dir)
os.makedirs(combined_train_dir)

# Add 7T training files
for f in lab2im_utils.list_images_in_folder(train_path_7T):
    dest = os.path.join(combined_train_dir, os.path.basename(f))
    if not os.path.exists(dest):
        os.symlink(f, dest)

# Add 3T files to training (with prefix to avoid name collision)
for f in trainval_files_3T:
    dest = os.path.join(combined_train_dir, '3T_' + os.path.basename(f))
    if not os.path.exists(dest):
        os.symlink(f, dest)

n_train_files = len(os.listdir(combined_train_dir))
print(f"Combined training: {n_train_files} files (7T + 3T)")

# Create 3T validation directory (same files as in training - different synthetic images)
val_3T_dir = '/home/althause/data/training_split/val_3T'
if os.path.exists(val_3T_dir):
    shutil.rmtree(val_3T_dir)
os.makedirs(val_3T_dir)
for f in trainval_files_3T:
    dest = os.path.join(val_3T_dir, os.path.basename(f))
    if not os.path.exists(dest):
        os.symlink(f, dest)

all_train_paths = combined_train_dir
all_val_paths = val_3T_dir
val_subjects_prob = None

print(f"\n=== Data Split Summary ===")
print(f"Training: {combined_train_dir} ({n_train_files} files - 7T + 3T)")
print(f"Validation: {val_3T_dir} ({len(trainval_files_3T)} 3T files)")
print(f"Note: Same 3T files used in both - SynthSeg generates different synthetic images each time")

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
dice_epochs = 200  # Train longer to learn small structures
steps_per_epoch = 1000
validation_steps = 400  # More steps for stable validation with small 3T set

# Generation and segmentation labels
# Use ALL labels for segmentation - model learns full anatomy, extract claustrum at inference
path_generation_labels = np.array([0, 14, 15, 16, 24,
                                   2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
# Keep all labels distinct - no collapsing to background
path_segmentation_labels = path_generation_labels.copy()

# Shape and resolution
target_res = 1.0
output_shape = 128  # All training labels resampled to 128³ @ 1mm
n_channels = 1

# GMM sampling - maps each generation label to an intensity class
# Max index must be < number of prior classes (19 classes = indices 0-18)
prior_distributions = 'normal'
path_generation_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Spatial deformation parameters
flipping = True
scaling_bounds = 0.15
rotation_bounds = 10
shearing_bounds = 0.012
translation_bounds = False
nonlin_std = 2.0
bias_field_std = 0.7

# Acquisition resolution parameters
randomise_res = False
data_res = np.array([0.86, 1.1, 0.86])  # slice spacing i.e. resolution to mimic
thickness = np.array([0.86, 1.1, 0.86])  # slice thickness

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
         subjects_prob=None,  # Equal probability for combined training set
         val_subjects_prob=val_subjects_prob,
         data_res=data_res,
         thickness=thickness,
         prior_means=means,
         prior_stds=stds
         )
