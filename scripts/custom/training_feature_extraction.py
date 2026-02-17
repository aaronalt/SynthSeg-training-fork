"""

This function is for training a network using synthetic scans generated from a set of training label maps.
See details in the docstring below.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import os
import json
import keras
import numpy as np
import tensorflow as tf
from keras import models
import keras.callbacks as KC
from keras.optimizers import Adam
import tensorflow.keras.backend as K
from inspect import getmembers, isclass
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers as keras_layers
from keras.utils import Sequence

# project imports
from SynthSeg import metrics_model as metrics
from SynthSeg.brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils, layers
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models
import threading


def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersect = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1. - (2. * intersect + smooth) / (denominator + smooth)


def save_training_params(model_dir, params):
    """Save training parameters to a JSON file in the model directory."""
    param_file = os.path.join(model_dir, 'training_params.json')
    serializable = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        elif isinstance(v, (str, int, float, bool, list, type(None))):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    with open(param_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved training parameters to {param_file}")


def training(labels_dir,
             model_dir,
             generation_labels=None,
             n_neutral_labels=None,
             segmentation_labels=None,
             subjects_prob=None,
             val_subjects_prob=None,
             batchsize=1,
             n_channels=1,
             target_res=None,
             output_shape=None,
             generation_classes=None,
             prior_distributions='uniform',
             prior_means=None,
             prior_stds=None,
             use_specific_stats_for_channel=False,
             mix_prior_and_random=False,
             flipping=True,
             scaling_bounds=.2,
             rotation_bounds=15,
             shearing_bounds=.012,
             translation_bounds=False,
             nonlin_std=4.,
             nonlin_scale=.04,
             randomise_res=True,
             max_res_iso=2.,
             max_res_aniso=4.,
             data_res=None,
             thickness=None,
             bias_field_std=.7,
             bias_scale=.025,
             return_gradients=False,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='elu',
             lr=1e-4,
             wl2_epochs=1,
             dice_epochs=50,
             steps_per_epoch=10000,
             checkpoint=None,
             val_path=None,
             skip_pretrain=False,
             finetune=False,
             validation_steps=100,
             label_weights=None):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # save all training parameters
    save_training_params(model_dir, {
        'labels_dir': labels_dir,
        'val_path': val_path,
        'checkpoint': checkpoint,
        'batchsize': batchsize,
        'n_channels': n_channels,
        'target_res': target_res,
        'output_shape': output_shape,
        'generation_labels': generation_labels,
        'segmentation_labels': segmentation_labels,
        'n_neutral_labels': n_neutral_labels,
        'generation_classes': generation_classes,
        'prior_distributions': prior_distributions,
        'flipping': flipping,
        'scaling_bounds': scaling_bounds,
        'rotation_bounds': rotation_bounds,
        'shearing_bounds': shearing_bounds,
        'translation_bounds': translation_bounds,
        'nonlin_std': nonlin_std,
        'nonlin_scale': nonlin_scale,
        'randomise_res': randomise_res,
        'max_res_iso': max_res_iso,
        'max_res_aniso': max_res_aniso,
        'bias_field_std': bias_field_std,
        'bias_scale': bias_scale,
        'n_levels': n_levels,
        'nb_conv_per_level': nb_conv_per_level,
        'conv_size': conv_size,
        'unet_feat_count': unet_feat_count,
        'feat_multiplier': feat_multiplier,
        'activation': activation,
        'lr': lr,
        'wl2_epochs': wl2_epochs,
        'dice_epochs': dice_epochs,
        'steps_per_epoch': steps_per_epoch,
        'skip_pretrain': skip_pretrain,
        'finetune': finetune,
        'label_weights': label_weights,
    })

    # get label lists
    generation_labels, _ = utils.get_list_labels(label_list=generation_labels, labels_dir=labels_dir)
    if segmentation_labels is not None:
        segmentation_labels, _ = utils.get_list_labels(label_list=segmentation_labels)
    else:
        segmentation_labels = generation_labels
    n_segmentation_labels = len(np.unique(segmentation_labels))

    # instantiate BrainGenerator objects
    brain_generator = BrainGenerator(labels_dir=labels_dir,
                                     generation_labels=generation_labels,
                                     n_neutral_labels=n_neutral_labels,
                                     output_labels=segmentation_labels,
                                     subjects_prob=subjects_prob,
                                     batchsize=batchsize,
                                     n_channels=n_channels,
                                     target_res=target_res,
                                     output_shape=output_shape,
                                     output_div_by_n=2 ** n_levels,
                                     generation_classes=generation_classes,
                                     prior_distributions=prior_distributions,
                                     prior_means=prior_means,
                                     prior_stds=prior_stds,
                                     use_specific_stats_for_channel=use_specific_stats_for_channel,
                                     mix_prior_and_random=mix_prior_and_random,
                                     flipping=flipping,
                                     scaling_bounds=scaling_bounds,
                                     rotation_bounds=rotation_bounds,
                                     shearing_bounds=shearing_bounds,
                                     translation_bounds=translation_bounds,
                                     nonlin_std=nonlin_std,
                                     nonlin_scale=nonlin_scale,
                                     randomise_res=randomise_res,
                                     max_res_iso=max_res_iso,
                                     max_res_aniso=max_res_aniso,
                                     data_res=data_res,
                                     thickness=thickness,
                                     bias_field_std=bias_field_std,
                                     bias_scale=bias_scale,
                                     return_gradients=return_gradients)

    val_brain_generator = BrainGenerator(labels_dir=val_path,
                                     generation_labels=generation_labels,
                                     n_neutral_labels=n_neutral_labels,
                                     output_labels=segmentation_labels,
                                     subjects_prob=val_subjects_prob,
                                     batchsize=batchsize,
                                     n_channels=n_channels,
                                     target_res=target_res,
                                     output_shape=output_shape,
                                     output_div_by_n=2 ** n_levels,
                                     generation_classes=generation_classes,
                                     prior_distributions=prior_distributions,
                                     prior_means=prior_means,
                                     prior_stds=prior_stds,
                                     use_specific_stats_for_channel=use_specific_stats_for_channel,
                                     mix_prior_and_random=mix_prior_and_random,
                                     flipping=flipping,
                                     scaling_bounds=scaling_bounds,
                                     rotation_bounds=rotation_bounds,
                                     shearing_bounds=shearing_bounds,
                                     translation_bounds=translation_bounds,
                                     nonlin_std=nonlin_std,
                                     nonlin_scale=nonlin_scale,
                                     randomise_res=randomise_res,
                                     max_res_iso=max_res_iso,
                                     max_res_aniso=max_res_aniso,
                                     data_res=data_res,
                                     thickness=thickness,
                                     bias_field_std=bias_field_std,
                                     bias_scale=bias_scale,
                                     return_gradients=return_gradients)

    # generation model
    labels_to_image_model = brain_generator.labels_to_image_model
    unet_input_shape = brain_generator.model_output_shape

    # prepare the segmentation model
    unet_model = nrn_models.unet(input_model=labels_to_image_model,
                                 input_shape=unet_input_shape,
                                 nb_labels=n_segmentation_labels,
                                 nb_levels=n_levels,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_size=conv_size,
                                 nb_features=unet_feat_count,
                                 feat_mult=feat_multiplier,
                                 activation=activation,
                                 batch_norm=-1,
                                 name='unet')

    val_generator = utils.build_training_generator(val_brain_generator.model_inputs_generator, batchsize)
    input_generator = utils.build_training_generator(brain_generator.model_inputs_generator, batchsize)

    # Load pretrained weights
    if checkpoint is not None:
        unet_model.load_weights(checkpoint, by_name=True, skip_mismatch=True)

    # -------------------------------------------------------------------------
    # Phase 1: WL2 Warmup (frozen encoder, train decoder/output layers only)
    # -------------------------------------------------------------------------
    if wl2_epochs > 0 and not skip_pretrain:
        print("\n=== WL2 Warmup Phase: Freezing encoder layers ===")

        # Freeze encoder layers (down arm), keep decoder trainable
        for layer in unet_model.layers:
            # Freeze layers that are part of the encoder (down path)
            # Patterns: unet_conv_downarm_*, unet_maxpool_*, unet_bn_down_*
            if '_downarm_' in layer.name or '_maxpool_' in layer.name or '_bn_down_' in layer.name:
                layer.trainable = False
            # Keep BatchNorm frozen regardless
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        # Print layer status for verification
        print("\nLayer trainability:")
        for layer in unet_model.layers:
            if hasattr(layer, 'trainable'):
                print(f"  {layer.name}: {'trainable' if layer.trainable else 'frozen'}")

        wl2_model = models.Model(unet_model.inputs, unet_model.outputs)
        wl2_model = metrics.metrics_model(wl2_model, segmentation_labels, 'wl2')

        train_model(wl2_model, input_generator, lr, wl2_epochs, steps_per_epoch, model_dir,
                    'wl2',
                    checkpoint,
                    reinitialise_momentum=True,
                    validation_data=val_generator,
                    phase='wl2',
                    validation_steps=validation_steps)

        # Update checkpoint to the WL2 trained weights
        checkpoint = os.path.join(model_dir, 'wl2_wl2_%03d_%d.h5' % (wl2_epochs, wl2_epochs))

    # -------------------------------------------------------------------------
    # Phase 2: Finetune (unfreeze all layers except BatchNorm)
    # -------------------------------------------------------------------------
    if dice_epochs > 0:
        print("\n=== Finetune Phase: Unfreezing all layers (except BatchNorm) ===")

        # Unfreeze all layers except BatchNorm
        for layer in unet_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        # Create fresh dice model with unfrozen weights
        dice_model = models.Model(unet_model.inputs, unet_model.outputs)
        dice_model = metrics.metrics_model(dice_model, segmentation_labels, 'dice', class_weights=label_weights)

        # Use lower learning rate for finetuning
        finetune_lr = lr / 10

        train_model(dice_model, input_generator, finetune_lr, dice_epochs, steps_per_epoch, model_dir,
                    'dice',
                    checkpoint,
                    reinitialise_momentum=True,
                    validation_data=val_generator,
                    phase='finetune',
                    validation_steps=validation_steps)


def train_model(model,
                generator,
                learning_rate,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                path_checkpoint=None,
                reinitialise_momentum=False,
                extra_callbacks=None,
		        validation_data=None,
                phase=None,
		        resume_epoch=None,
                validation_steps=100):

    # prepare model and log folders
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_%s_{epoch:03d}_%d.h5' % (metric_type, phase, n_epochs))
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1), KC.CSVLogger(os.path.join(log_dir, 'training.log'))]

    # early stopping on validation loss - high patience to allow small structures to learn
    callbacks.append(KC.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True))

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # TensorBoard callback
    callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch'))

    compile_model = True
    init_epoch = 0
    if path_checkpoint is not None:
        if resume_epoch:
            init_epoch =  resume_epoch
        if (not reinitialise_momentum) & (metric_type in path_checkpoint):
            custom_l2i = {key: value for (key, value) in getmembers(layers, isclass) if key != 'Layer'}
            custom_nrn = {key: value for (key, value) in getmembers(nrn_layers, isclass) if key != 'Layer'}
            custom_objects = {**custom_l2i, **custom_nrn, 'tf': tf, 'keras': keras, 'loss': metrics.IdentityLoss().loss}
            model = models.load_model(path_checkpoint, custom_objects=custom_objects)
            compile_model = False
        else:
            model.load_weights(path_checkpoint, by_name=True)

    # compile
    if compile_model or metric_type == 'dice' or not hasattr(model, 'optimizer'):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=metrics.IdentityLoss().loss)

    # fit
    model.fit(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=init_epoch,
			validation_data=validation_data,
			validation_steps=validation_steps)
