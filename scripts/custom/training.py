"""
SynthSeg training script for claustrum segmentation
Optimized for NVIDIA RTX 2000 Ada with CUDA 12.x and TensorFlow 2.15
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import datetime
import tensorflow as tf
import numpy as np
from SynthSeg.training import training
from create_train_test_split import create_train_test_split

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

# Paths
training_label_maps = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/training_labels'
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Split into train/test
train_path, test_path = create_train_test_split(
    base_dir=training_label_maps,
    subdirs=['t1w', 't2w-cor', 't2w-tra'],
    output_dir='/tmp/training_split',
    method='copy',
    test_ratio=0.2,
    seed=42,
    verbose=False
)

path_training_label_maps = str(train_path)

# Pre-trained model
path_checkpoint = '/home/aaron/nas2/DATA_inProgress/Aaron/claustrum_model_weights/outputs/mauri_unet_weights.h5' 

batchsize = 1

# Architecture parameters
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2

# Training parameters
lr = 1e-5
wl2_epochs = 0
dice_epochs = 50
steps_per_epoch = 500

# Generation and segmentation labels
path_generation_labels = np.array([0, 14, 16, 24, 2, 3, 4, 5, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 138, 139])
n_neutral_labels = 4
path_segmentation_labels = np.array([0, 0, 0, 0, 2, 3, 4, 5, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 138, 139])

# Shape and resolution
target_res = 0.35
output_shape = (256, 256, 32)  # Adjust based on your data
n_channels = 4

# GMM sampling
prior_distributions = 'uniform'
path_generation_classes = np.array([0, 3, 10, 3, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 14])

# Spatial deformation parameters
flipping = False
scaling_bounds = 0.1 # 0.2
rotation_bounds = 5 # 15
shearing_bounds = 0.008 # 0.012
translation_bounds = False
nonlin_std = 2.0 # 4.0
bias_field_std = 0.3 # 0.7

# Acquisition resolution parameters
randomise_res = True

print("\n=== Training Configuration ===")
print(f"Training labels: {path_training_label_maps}")
print(f"Output directory: {path_model_dir}")
print(f"Batch size: {batchsize}")
print(f"Output shape: {output_shape}")
print(f"Epochs: {dice_epochs}")
print(f"Steps per epoch: {steps_per_epoch}")
print("\nStarting training...\n")

# Start training
training(path_training_label_maps,
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
         checkpoint=path_checkpoint)
