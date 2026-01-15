"""
Extract inference model from training checkpoint
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from SynthSeg.model_inputs import build_model_inputs

# Training parameters (must match your training script)
generation_labels = np.array([0, 14, 15, 16, 24, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 138, 139])
n_neutral_labels = 5
output_shape = (256, 256, 32)
n_channels = 4

# Architecture parameters
n_levels = 5
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
activation = 'elu'
feat_multiplier = 2

print("Building inference model...")

# Build just the U-Net for inference
from ext.neuron import models as nrn_models

# Input layer for images (not labels!)
input_shape = list(output_shape) + [n_channels]
image_input = keras.layers.Input(shape=input_shape, name='image_input')

# Build U-Net
last_tensor = image_input
n_labels = len(np.unique(generation_labels))

# U-Net encoder
conv_kwargs = {'padding': 'same', 'activation': activation}
convL_layer = getattr(keras.layers, 'Conv%dD' % 3)

# Encoder
temp_conv_layers = []
for level in range(n_levels):
    n_features = unet_feat_count * (feat_multiplier ** level)
    for _ in range(nb_conv_per_level):
        last_tensor = convL_layer(n_features, conv_size, **conv_kwargs)(last_tensor)
    temp_conv_layers.append(last_tensor)
    if level < n_levels - 1:
        last_tensor = keras.layers.MaxPooling3D(pool_size=2)(last_tensor)

# Decoder
for level in range(n_levels - 2, -1, -1):
    n_features = unet_feat_count * (feat_multiplier ** level)
    last_tensor = keras.layers.UpSampling3D(size=2)(last_tensor)
    last_tensor = keras.layers.concatenate([last_tensor, temp_conv_layers[level]])
    for _ in range(nb_conv_per_level):
        last_tensor = convL_layer(n_features, conv_size, **conv_kwargs)(last_tensor)

# Final segmentation layer
last_tensor = convL_layer(n_labels, 1, padding='same', activation='softmax', name='unet_prediction')(last_tensor)

# Create model
inference_model = keras.Model(inputs=image_input, outputs=last_tensor)

print("Loading weights from training model...")

# Load the full training model
training_model = keras.models.load_model('./outputs/training_label_maps_claustrum/dice_010.h5', compile=False)

# Extract only U-Net weights (skip augmentation layers)
print("\nMatching and transferring weights...")
transferred = 0
for layer in inference_model.layers:
    try:
        weights = training_model.get_layer(layer.name).get_weights()
        layer.set_weights(weights)
        transferred += 1
        print(f"✓ Transferred: {layer.name}")
    except:
        print(f"✗ Skipped: {layer.name} (not in training model)")

print(f"\nTransferred weights for {transferred}/{len(inference_model.layers)} layers")

# Save inference model
output_path = './outputs/training_label_maps_claustrum/dice_010_inference.h5'
inference_model.save(output_path)
print(f"\nSaved inference model to: {output_path}")

# Test it
print("\nTesting inference model...")
test_input = np.random.randn(1, 256, 256, 32, n_channels).astype(np.float32)
test_output = inference_model.predict(test_input, verbose=0)
print(f"Test successful! Output shape: {test_output.shape}")
