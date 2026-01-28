"""
SynthSeg training script for claustrum segmentation
Optimized for NVIDIA RTX 2000 Ada with CUDA 12.x and TensorFlow 2.15
"""


import os
import tensorflow as tf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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

import datetime
import numpy as np
# from SynthSeg.training import training
from create_train_test_split import create_train_test_split, extract_test_from_validation
from standardize_labels import standardize_training_labels
from training_feature_extraction import training

# Paths
training_label_maps = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/training_labels'
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Split into train/test
t1w_train_path, t1w_val_path = create_train_test_split(
    base_dir=training_label_maps,
    subdirs=['t1w'],
    output_dir='/tmp/training_split_t1w',
    method='copy',
    test_ratio=0.3,
    seed=42,
    verbose=False
)

t2w_train_path, t2w_val_path = create_train_test_split(
    base_dir=training_label_maps,
    subdirs=['t2w-cor', 't2w-tra'],
    output_dir='/tmp/training_split_t2w',
    method='copy',
    test_ratio=0.3,
    seed=42,
    verbose=False
)

t3_train_path, t3_val_path = create_train_test_split(
    base_dir='/home/aaron/nas2/DATA_inProgress/Aaron/3T/Nifti/derivatives/training_labels',
    subdirs=['t1w'],
    output_dir='/tmp/training_split_3T',
    method='copy',
    test_ratio=0.3,
    seed=42,
    verbose=False
)

extract_test_from_validation(
    val_dir=t1w_val_path,
    output_test_dir='/tmp/test_t1w',
    test_ratio=0.5,
    seed=42
)

extract_test_from_validation(
    val_dir=t2w_val_path,
    output_test_dir='/tmp/test_t2w',
    test_ratio=0.5,
    seed=42
)

extract_test_from_validation(
    val_dir=t3_val_path,
    output_test_dir='/tmp/test_3T',
    test_ratio=0.5,
    seed=42
)

standardize_training_labels(t1w_train_path, t1w_train_path)
standardize_training_labels(t1w_val_path, t1w_val_path)
standardize_training_labels(t2w_train_path, t2w_train_path)
standardize_training_labels(t2w_val_path, t2w_val_path)
standardize_training_labels(t3_train_path, t3_train_path)
standardize_training_labels(t3_val_path, t3_val_path)
train_files_3t = list(t3_train_path.glob('*.nii.gz'))
train_files_7t_t1 = list(t1w_train_path.glob('*.nii.gz'))
train_files_7t_t2 = list(t2w_train_path.glob('*.nii.gz'))
val_files_3t = list(t3_val_path.glob('*.nii.gz'))
val_files_7t_t1 = list(t1w_val_path.glob('*.nii.gz'))
val_files_7t_t2 = list(t2w_val_path.glob('*.nii.gz'))

# Symlink all training/validation files into single directories
all_train_paths = '/tmp/training_all_train'
all_val_paths = '/tmp/training_all_val'
os.makedirs(all_train_paths, exist_ok=True)
os.makedirs(all_val_paths, exist_ok=True)
for prefix, files in [('7t_t1w_', train_files_7t_t1), ('7t_t2w_', train_files_7t_t2), ('3t_', train_files_3t)]:
    for f in files:
        dest = os.path.join(all_train_paths, prefix + os.path.basename(str(f)))
        if not os.path.exists(dest):
            os.symlink(str(f), dest)
for prefix, files in [('7t_t1w_', val_files_7t_t1), ('7t_t2w_', val_files_7t_t2), ('3t_', val_files_3t)]:
    for f in files:
        dest = os.path.join(all_val_paths, prefix + os.path.basename(str(f)))
        if not os.path.exists(dest):
            os.symlink(str(f), dest)
print(all_train_paths)

# Create a probability array
all_train_files = sorted(os.listdir(all_train_paths))
probs = []
for fname in all_train_files:
    if fname.startswith('3t_'):
        probs.append(0.5 / len(train_files_3t))
    elif fname.startswith('7t_t1w_'):
        probs.append(0.3 / len(train_files_7t_t1))
    elif fname.startswith('7t_t2w_'):
        probs.append(0.2 / len(train_files_7t_t2))
# Normalize to ensure they sum to 1.0
subjects_prob = np.array(probs) / np.sum(probs)

# Pre-trained model
path_checkpoint = '/home/aaron/nas2/DATA_inProgress/Aaron/CLAU/claustrum_model_weights/outputs/mauri_unet_weights.h5'
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
lr = 1e-5
wl2_epochs = 0
dice_epochs = 50
steps_per_epoch = 1000

# Generation and segmentation labels
path_generation_labels = np.array([0, 14, 15, 16, 24,
                                   2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                   41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
n_neutral_labels = 5
path_segmentation_labels = np.array([0, 14, 15, 16, 24,
                                     2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                     41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])

# Shape and resolution
target_res = 0.35
output_shape = 128  # Adjust based on your data
n_channels = 1

# GMM sampling
prior_distributions = 'uniform'
path_generation_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Spatial deformation parameters
flipping = False
scaling_bounds = 0.2
rotation_bounds = 15
shearing_bounds = 0.012
translation_bounds = False
nonlin_std = 4.0
bias_field_std = 0.7

# Acquisition resolution parameters
randomise_res = True

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
         subjects_prob=subjects_prob)
