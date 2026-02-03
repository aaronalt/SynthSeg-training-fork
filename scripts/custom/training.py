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

# Subject-aware stratified split across 7T and 3T
train_path, val_path, subjects_prob, val_subjects_prob = create_train_test_split(
    base_dirs_with_subdirs=[
        {
            'base_dir': '/home/althause/data/7T/training_labels',
            'subdirs': ['t1w', 't2w-cor', 't2w-tra'],
            'field_strength': '7T',
        },
        {
            'base_dir': '/home/althause/data/3T/training_labels',
            'subdirs': ['t1w'],
            'field_strength': '3T',
        },
    ],
    output_dir='/home/althause/data/training_split',
    method='symlink',
    test_ratio=0.3,
    seed=42,
    prob_mode='sqrt',
    verbose=True,
)

# Extract held-out test set from validation (subject-aware)
extract_test_from_validation(
    val_dir=val_path,
    output_test_dir='/home/althause/data/test_set',
    test_ratio=0.5,
    seed=42,
)

# Standardize labels
all_train_paths = str(train_path)
all_val_paths = str(val_path)
standardize_training_labels(train_path, train_path)
standardize_training_labels(val_path, val_path)

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
lr = 1e-6
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
