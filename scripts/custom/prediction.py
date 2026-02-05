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
import json

# === OPTIONS ===
DELETE_TMP_PREDICTIONS = False  # Set to True to delete predictions after evaluation (saves disk space)

# Store results across all models for summary
all_model_results = []
model_dice_scores = {}  # Track mean dice per model for cleanup


# Save label arrays to files (do this once)
os.makedirs('./data', exist_ok=True)

# Full label set (must match training!)
segmentation_labels = np.array([0, 14, 15, 16, 24,
                                2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 138,
                                41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 139])
topology_classes = np.array([0, 1, 2, 3, 4,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

np.save('./data/labels_classes_priors/segmentation_labels.npy', segmentation_labels)
np.save('./data/labels_classes_priors/topology_classes.npy', topology_classes)

# Find all model checkpoints and sort by epoch number
model_dir = '/home/althause/SynthSeg-training-fork/models/test/<experiment>'
model_files = sorted(glob(os.path.join(model_dir, '*.h5')))

# Extract epoch number for sorting (assumes format like 'dice_finetune_005_20.h5')
def get_epoch(f):
    match = re.search(r'_(\d+)_\d+\.h5$', f)
    return int(match.group(1)) if match else 0

model_files = sorted(model_files, key=get_epoch)

print(f"Found {len(model_files)} model checkpoints to evaluate")
print(f"Models: {[os.path.basename(f) for f in model_files]}")

for path_model in model_files:
    # Paths
    path_images = '/home/althause/data/training_split/test'
    path_segm = f'/home/althause/data/seg/pred_{path_model}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    path_posteriors = os.path.join(path_segm, 'posteriors')
    path_resampled = os.path.join(path_segm, 'resampled')
    path_vol = os.path.join(path_segm, 'volumes.csv')
    gt_folder = None  # '/home/aaron/nas2/DATA_inProgress/Aaron/7T/Nifti/derivatives/claustrum_manual_labels/validation'

    # Extract model params
    json_params = os.join(model_dir, 'training_params.json')
    with open(json_params, "r") as jsonfile:
        trained_model_params = json.load(jsonfile)

    # Model and labels
    path_segmentation_labels = trained_model_params['segmentation_labels']
    path_topology_classes = trained_model_params['generation_classes']

    # Create output directories
    os.makedirs(path_segm, exist_ok=True)
    os.makedirs(path_posteriors, exist_ok=True)
    os.makedirs(path_resampled, exist_ok=True)

    # Parameters (must match training!)
    n_neutral_labels = trained_model_params['n_neutral_labels']
    cropping = trained_model_params['cropping']
    target_res =  trained_model_params['target_res']
    flip = True
    sigma_smoothing = 0.5
    keep_biggest_component = False

    # Architecture (must match training!)
    n_levels = trained_model_params['n_levels']
    nb_conv_per_level = trained_model_params['nb_conv_per_level']
    conv_size = trained_model_params['conv_size']
    unet_feat_count = trained_model_params['unet_feat_count']
    activation = trained_model_params['activation']
    feat_multiplier = trained_model_params['feat_multiplier']
    n_channels = trained_model_params['n_channels']

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
