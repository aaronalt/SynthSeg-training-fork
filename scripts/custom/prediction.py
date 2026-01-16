"""
Prediction script for custom-trained claustrum segmentation
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Cleans up logs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
keras.backend.set_image_data_format('channels_last')
from SynthSeg.predict import predict
import numpy as np
from glob import glob

# Save label arrays to files (do this once)
os.makedirs('./data', exist_ok=True)

segmentation_labels = np.array([0, 0, 0, 0, 2, 3, 4, 5, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 138, 139]) 
topology_classes = np.array([0, 3, 10, 3, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 14])

np.save('./data/labels_classes_priors/segmentation_labels.npy', segmentation_labels)
np.save('./data/labels_classes_priors/topology_classes.npy', topology_classes)

# Paths
path_images = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/claustrum_manual_labels/prediction'
path_segm = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/segmentations'
path_posteriors = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/segmentations/posteriors'
path_resampled = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/segmentations/resampled'
path_vol = '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/segmentations/volumes.csv'
gt_folder = None  # '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/claustrum_manual_labels/validation'

# Model and labels
path_model = './models/test/experiment_20260115_124240/dice_050.h5'
path_segmentation_labels = './data/labels_classes_priors/segmentation_labels.npy'
path_topology_classes = './data/labels_classes_priors/topology_classes.npy'

# Create output directories
os.makedirs(path_segm, exist_ok=True)
os.makedirs(path_posteriors, exist_ok=True)
os.makedirs(path_resampled, exist_ok=True)

# Parameters (must match training!)
n_neutral_labels = 4
cropping = None
target_res = 0.35
flip = False
sigma_smoothing = 0.5
keep_biggest_component = True

# Architecture (must match training!)
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2
n_channels = 4

compute_distances = True

print("\n=== Prediction Configuration ===")
print(f"Input images: {path_images}")
print(f"Output directory: {path_segm}")
print(f"Model: {path_model}")
print(f"Target resolution: {target_res}")
print(f"n_channels: {n_channels}")
print("\nStarting prediction...\n")

# Run prediction
predict(path_images,
        path_segm,
        path_model,
        path_segmentation_labels, 
        n_neutral_labels=n_neutral_labels,
        path_posteriors=path_posteriors,
        path_resampled=path_resampled,
        path_volumes=path_vol,
        cropping=cropping,
        target_res=target_res,
        flip=flip,
        topology_classes=path_topology_classes, 
        sigma_smoothing=sigma_smoothing,
        keep_biggest_component=keep_biggest_component,
        n_levels=n_levels,
        nb_conv_per_level=nb_conv_per_level,
        conv_size=conv_size,
        unet_feat_count=unet_feat_count,
        feat_multiplier=feat_multiplier,
        activation=activation,
        gt_folder=gt_folder,
        compute_distances=compute_distances)

print("\nPrediction complete!")
