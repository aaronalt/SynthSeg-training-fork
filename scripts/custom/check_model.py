#!/usr/bin/env python3
"""
Load and inspect a SynthSeg model with proper custom layer handling.
Based on the SynthSeg framework architecture.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras

# Add SynthSeg path
synthseg_path = '/Users/aaronalthauser/MRI_local/SynthSeg-training-fork'
if os.path.exists(synthseg_path):
    sys.path.insert(0, synthseg_path)
    print(f"✓ Added SynthSeg path: {synthseg_path}")
else:
    print(f"✗ SynthSeg path not found: {synthseg_path}")
    sys.exit(1)

try:
    # Import lab2im layers (this registers custom layers)
    from ext.lab2im import layers as lab2im_layers
    from ext.lab2im import utils as lab2im_utils

    print("✓ Successfully imported lab2im modules")
except ImportError as e:
    print(f"✗ Could not import lab2im: {e}")
    print("\nTrying to import from SynthSeg directly...")
    try:
        from SynthSeg import layers as synthseg_layers

        print("✓ Successfully imported SynthSeg layers")
    except ImportError as e2:
        print(f"✗ Could not import SynthSeg layers: {e2}")
        sys.exit(1)

# Model path
model_path = '/Volumes/VCSF_GROUP/DATA_inProgress/Aaron/models/dice_050.h5'

if not os.path.exists(model_path):
    print(f"✗ Model file not found: {model_path}")
    sys.exit(1)

print(f"\n{'=' * 70}")
print(f"Loading model: {model_path}")
print(f"{'=' * 70}\n")

try:
    # Try loading with compile=False (recommended for SynthSeg models)
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Model loaded successfully!\n")

    # Print summary
    print(f"{'=' * 70}")
    print("MODEL ARCHITECTURE")
    print(f"{'=' * 70}")
    model.summary()

    # Print detailed information
    print(f"\n{'=' * 70}")
    print("MODEL DETAILS")
    print(f"{'=' * 70}")
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    # Count trainable vs non-trainable
    trainable = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")

    # List all layers
    print(f"\n{'=' * 70}")
    print(f"ALL LAYERS ({len(model.layers)} total)")
    print(f"{'=' * 70}")
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape
        print(f"{i + 1:3d}. {layer.name:40s} {layer_type:30s} {str(output_shape)}")

    # Identify custom layers
    print(f"\n{'=' * 70}")
    print("CUSTOM LAYERS")
    print(f"{'=' * 70}")
    custom_layer_types = [
        'RandomSpatialDeformation', 'RandomCrop', 'RandomFlip',
        'GaussianBlur', 'DiceLoss', 'ConvertLabels', 'PadAroundCentre',
        'RandomBiasField', 'IntensityAugmentation', 'RandomGamma'
    ]

    found_custom = False
    for layer in model.layers:
        if layer.__class__.__name__ in custom_layer_types:
            print(f"  - {layer.name} ({layer.__class__.__name__})")
            found_custom = True

    if not found_custom:
        print("  No custom layers detected (all standard Keras layers)")

    print(f"\n{'=' * 70}")
    print("SUCCESS - Model is loadable and appears valid!")
    print(f"{'=' * 70}\n")

except Exception as e:
    print(f"\n{'=' * 70}")
    print(f"✗ ERROR LOADING MODEL")
    print(f"{'=' * 70}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"\nFull traceback:")
    import traceback

    traceback.print_exc()
    print(f"\n{'=' * 70}")
    print("TROUBLESHOOTING SUGGESTIONS:")
    print(f"{'=' * 70}")
    print("1. Verify SynthSeg installation:")
    print(f"   ls -la {synthseg_path}/ext/lab2im/")
    print("\n2. Check if model file is corrupted:")
    print(f"   h5ls {model_path}")
    print("\n3. Try loading in Python interactively:")
    print("   python3")
    print("   >>> import tensorflow as tf")
    print("   >>> model = tf.keras.models.load_model('path/to/model.h5', compile=False)")
    sys.exit(1)