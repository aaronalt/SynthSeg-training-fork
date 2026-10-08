"""
SynthSeg training script for claustrum segmentation
#######
Version: hd95.warmup.reduceLROnPlateau
Changelog: -added reduceLRonPlateau, hd95 loss more gradual warmup 10-40 epochs .0001-.0025 -compare with: hd95.warmup
Check: 
#######
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
from create_train_test_split_stratified import create_train_test_split, extract_test_from_validation
from training_feature_extraction import training
from SynthSeg.estimate_priors import build_intensity_stats
# === DATA SETUP ===
import shutil
from ext.lab2im import utils as lab2im_utils

# === CONFIGURATION ===

# Subjects to exclude from training (poor labels)
EXCLUDE_SUBJECTS = []  # Add any bad subjects here
VERBOSE = False
version = 'hd95.warmup.reduceLROnPlateau'

# Experiment setup
experiment_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{version}"
path_model_dir = os.path.join('./models/test', experiment_name)
os.makedirs(path_model_dir, exist_ok=True)
log_dir = os.path.join(path_model_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Train, val, test sets
TRAIN = Path('/home/althause/data/TRAIN')
VAL = Path('/home/althause/data/VAL')
TEST = Path('/home/althause/data/TEST')
train_subjects = sorted(TRAIN.iterdir())
val_subjects = sorted(VAL.iterdir())
test_subjects = sorted(TEST.iterdir())
# Save split info
split_summary = {
    'experiment': experiment_name,
    'version': version,
    'train_subjects': sorted(train_subjects),
    'val_subjects': sorted(val_subjects),
    'test_subjects': sorted(test_subjects),
}


if VERBOSE:
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Training: {count(train_subjects)} files")
    print(train_subjects)
    print(f"Validation subjects: {count(val_subjects)}")
    print(val_subjects)
    print(f"Test subjects: {count(test_subjects)}")
    print(test_subjects)
    print("="*70 + "\n")

# === MODEL PARAMETERS ===
# Pre-trained model
path_checkpoint =  '/home/althause/data/weights/mauri_unet_weights.h5'
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
wl2_epochs = 0  # 0 to Skip warmup - using pretrained weights
dice_epochs = 200
steps_per_epoch = 1000
validation_steps = 80  

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

# Weight of the HD95 boundary loss added to the soft Dice loss (0 = Dice only)
hd95_weight = 0.0001

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
shearing_bounds = 0.015
translation_bounds = 15
nonlin_std = 5.0
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
path_training_labels = '/home/althause/data/intensity_estimation/labels'
'''
means, stds = build_intensity_stats(path_mri, path_training_labels, path_mri,
                                    estimation_labels=path_generation_labels,
                                    estimation_classes=path_generation_classes
                                    )
'''
means = np.load(os.path.join(path_mri, 'prior_means.npy'))
stds = np.load(os.path.join(path_mri, 'prior_stds.npy'))
path_mri = '/home/althause/data/intensity_estimation/images'
means = np.load(os.path.join(path_mri, 'prior_means.npy'))
stds = np.load(os.path.join(path_mri, 'prior_stds.npy'))

# === START TRAINING ===
skip_pretrain = False
training(TRAIN,
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
         val_path=VAL,
         skip_pretrain=skip_pretrain,
         finetune=False,
         subjects_prob=None,
         val_subjects_prob=None,
         data_res=data_res,
         thickness=thickness,
         max_res_iso=max_res_iso,
         max_res_aniso=max_res_aniso,
         prior_means=means,
         prior_stds=stds,
         label_weights=label_weights,
         hd95_weight=hd95_weight
         )
