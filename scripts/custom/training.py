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
from standardize_labels import standardize_training_labels
from training_feature_extraction import training
from SynthSeg.estimate_priors import build_intensity_stats

# Paths
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Subject-aware stratified split - train on 7T only
train_path, val_path, subjects_prob, val_subjects_prob = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/7T/training_labels',
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

# Separate split for 3T validation/test data
train_path_3T, _, _, _ = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/3T/training_labels',
            'subdirs': ['t1w'],
            'field_strength': '3T',
        },
    ],
    output_dir='/home/althause/data/training_split_3T',
    method='symlink',
    test_ratio=0.0,  # All goes to "train" which we'll use as val/test pool
    seed=42,
    prob_mode='sqrt',
    verbose=True,
)

# Note: 3T test set is created separately below (test_3T_dir)
# 7T test data can still be extracted if needed:
# extract_test_from_validation(val_dir=val_path, output_test_dir='/home/althause/data/test_set_7T', test_ratio=0.5, seed=42)

# Standardize labels
all_train_paths = str(train_path)
standardize_training_labels(train_path, train_path)

# Split 3T data into validation (70%) and test (30%)
import shutil
from ext.lab2im import utils as lab2im_utils

val_files_3T = lab2im_utils.list_images_in_folder(train_path_3T)
np.random.seed(42)
np.random.shuffle(val_files_3T)
split_idx = int(len(val_files_3T) * 0.7)
val_files_3T_final = val_files_3T[:split_idx]
test_files_3T = val_files_3T[split_idx:]

# Create 3T validation directory
val_3T_dir = '/home/althause/data/training_split/val_3T'
if os.path.exists(val_3T_dir):
    shutil.rmtree(val_3T_dir)
os.makedirs(val_3T_dir)
for f in val_files_3T_final:
    os.symlink(f, os.path.join(val_3T_dir, os.path.basename(f)))

# Create 3T test directory
test_3T_dir = '/home/althause/data/training_split/test_3T'
if os.path.exists(test_3T_dir):
    shutil.rmtree(test_3T_dir)
os.makedirs(test_3T_dir)
for f in test_files_3T:
    os.symlink(f, os.path.join(test_3T_dir, os.path.basename(f)))

print(f"3T split: {len(val_files_3T_final)} validation, {len(test_files_3T)} test")
all_val_paths = val_3T_dir
val_subjects_prob = None  # Reset since we're using filtered list
standardize_training_labels(val_3T_dir, val_3T_dir)

# Pre-trained model
path_checkpoint = '/home/althause/data/weights/mauri_unet_weights.h5'
# #'/home/aaron/SynthSeg-training/SynthSeg-training-fork/models/test/experiment_20260121_122955/dice_pretrain_070_100.h5'
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
wl2_epochs = 2
dice_epochs = 20
steps_per_epoch = 1000

# Generation and segmentation labels
path_generation_labels = np.array([0, 14, 15, 16, 24,
                                   2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
path_segmentation_labels = np.array([0, 0, 0, 0, 0,
                                     2, 3, 0, 0, 0, 0, 0, 0, 12, 0, 17, 18, 0, 0, 138,
                                     41, 42, 0, 0, 0, 0, 0, 0, 51, 0, 53, 54, 0, 0, 139])

# Shape and resolution
target_res = 1.0
output_shape = 160  # Adjust based on your data
n_channels = 1

# GMM sampling
prior_distributions = 'normal'
path_generation_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Spatial deformation parameters
flipping = False
scaling_bounds = 0.1
rotation_bounds = 5
shearing_bounds = 0.005
translation_bounds = False
nonlin_std = 1.0
bias_field_std = 0.2

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
         checkpoint=path_checkpoint,
         val_path=all_val_paths,
         skip_pretrain=skip_pretrain,
         finetune=False,
         subjects_prob=subjects_prob,
         val_subjects_prob=val_subjects_prob,
         data_res=data_res,
         thickness=thickness,
         prior_means=means,
         prior_stds=stds
         )
