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
import keras
import numpy as np
import tensorflow as tf
from keras import models
import keras.callbacks as KC
from keras.optimizers import Adam
from scipy.ndimage import distance_transform_edt
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


alpha_tensor = K.variable(0.0, name='loss_alpha')


class AlphaScheduler(KC.Callback):
    def __init__(self, alpha_var, start_epoch=5, max_alpha=0.05, ramp_steps=10):
        super().__init__()
        self.alpha_var = alpha_var
        self.start_epoch = start_epoch
        self.max_alpha = max_alpha
        self.ramp_steps = ramp_steps

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.start_epoch:
            new_val = 0.0
        else:
            # Linear ramp from 0 to max_alpha over ramp_steps
            progress = (epoch - self.start_epoch) / self.ramp_steps
            new_val = min(self.max_alpha, progress * self.max_alpha)

        K.set_value(self.alpha_var, new_val)
        print(f"\n[Epoch {epoch + 1}] Current Alpha (Boundary Penalty): {new_val:.4f}")


class HDDiceManager:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def loss(self, y_true, y_pred):
        d_loss = dice_loss(y_true, y_pred)
        h_loss = boundary_loss(y_true, y_pred)
        return d_loss + (self.alpha * h_loss)


def dice_loss(y_true, y_pred):
    smooth = 1e-5
    intersect = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1. - (2. * intersect + smooth) / (denominator + smooth)


def boundary_loss(y_true, y_pred):
    dist_map = compute_edt_distance(y_true)
    return tf.reduce_mean(y_pred * dist_map)


def combined_loss(y_true, y_pred):
    d_loss = dice_loss(y_true, y_pred)
    b_loss = boundary_loss(y_true, y_pred)
    tf.print("Dice:", d_loss, "Boundary:", b_loss, "Alpha:", alpha_tensor)
    return d_loss + (alpha_tensor * b_loss)


class CombinedLossLayer(keras_layers.Layer):
    """Computes combined dice + boundary loss inside the model graph.
    Input: [ground_truth_onehot, predictions] both shape (batch, D, H, W, n_labels)
    Output: scalar loss value (to be used with IdentityLoss)
    """
    def call(self, inputs):
        y_true, y_pred = inputs
        d_loss = dice_loss(y_true, y_pred)
        b_loss = boundary_loss(y_true, y_pred)
        tf.print("Dice:", d_loss, "Boundary:", b_loss, "Alpha:", alpha_tensor)
        return d_loss + (alpha_tensor * b_loss)


def combined_metrics_model(input_model, label_list):
    """Replaces metrics.metrics_model — extracts GT from the generation model
    and computes combined dice+boundary loss inside the graph."""
    from keras.models import Model
    import keras.layers as KL

    # Get prediction tensor
    last_tensor = input_model.outputs[0] if isinstance(input_model.outputs, list) else input_model.output
    input_shape = last_tensor.get_shape().as_list()[1:]
    n_labels = input_shape[-1]
    label_list = np.unique(label_list)
    assert n_labels == len(label_list), 'label_list should be as long as the posteriors channels'

    # Extract GT from generation model (same as metrics_model)
    labels_out_layer = input_model.get_layer('labels_out')
    labels_gt = labels_out_layer.get_output_at(0) if hasattr(labels_out_layer, 'get_output_at') else labels_out_layer.output
    labels_gt = layers.ConvertLabels(label_list)(labels_gt)
    labels_gt = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels_gt)
    labels_gt = KL.Reshape(input_shape)(labels_gt)

    loss_tensor = CombinedLossLayer()([labels_gt, last_tensor])
    return Model(inputs=input_model.inputs, outputs=loss_tensor)


def compute_edt_distance(y_true, spacing=(1.0, 1.0, 1.0)):
    """
    Computes the Euclidean Distance Transform for a binary label.

    Args:
        y_true: Ground truth binary mask (Batch, D, H, W, 1)
        spacing: Physical distance between voxels (Anisotropy)
    """

    def _edt_numpy(y_true_np):
        # Initialize output array
        dist_map = np.zeros_like(y_true_np, dtype=np.float32)

        # Process each item in the batch
        for i in range(y_true_np.shape[0]):
            # Squeeze to 3D for scipy
            mask = np.squeeze(y_true_np[i]).astype(np.bool_)

            if not np.any(mask):
                # If no claustrum is present, return a high penalty for any prediction
                dist_map[i] = np.ones_like(y_true_np[i]) * 100.0
                continue

            # Distance to background (internal distance)
            internal_dist = distance_transform_edt(mask, sampling=spacing)
            # Distance to foreground (external distance)
            external_dist = distance_transform_edt(~mask, sampling=spacing)

            # Signed distance: positive outside, negative inside
            # Shape it back to (D, H, W, 1)
            dist_map[i] = (external_dist - internal_dist)[..., np.newaxis]

        return dist_map

    # Wrap the numpy function so TensorFlow can call it during the forward pass
    y_true = tf.cast(y_true, tf.float32)
    return tf.py_function(_edt_numpy, [y_true], tf.float32)


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
             max_res_iso=4.,
             max_res_aniso=8.,
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
             finetune=False):

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

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

    unet_model.load_weights(checkpoint, by_name=True, skip_mismatch=True)
    unet_model.trainable = False

    dice_model = combined_metrics_model(unet_model, segmentation_labels)
    dice_model.summary()

    reinitialize_momentum = True

    resume_epoch = False
    
    my_alpha_scheduler = AlphaScheduler(alpha_tensor, start_epoch=5, max_alpha=0.05)
    loss_manager = HDDiceManager(alpha_tensor)

    if resume_epoch:
        checkpoint = os.path.join('/home/aaron/SynthSeg-training/SynthSeg-training-fork/models/test/experiment_20260120_103539/dice_pretrain_052_100.h5')
        reinitialize_momentum = False
        resume_epoch = 48

    if not skip_pretrain:
        phase = 'pretrain'
        train_model(dice_model, input_generator, lr, dice_epochs, steps_per_epoch, model_dir,
                    'hd_dice',
                    checkpoint,
                    reinitialise_momentum=reinitialize_momentum,
                    validation_data=val_generator,
                    phase=phase,
                    extra_callbacks=[my_alpha_scheduler],
                    resume_epoch=resume_epoch,
                    loss_manager=loss_manager)

    # Unfreeze base model and fine-tune
    if finetune:
        phase = 'finetune'
        if skip_pretrain:
            checkpoint = os.path.join('best_pretrain_model')
        unet_model.trainable = True
        for layer in unet_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        fine_tune_lr = lr / 10
        fine_tune_epochs = int(dice_epochs / 2)
        train_model(dice_model, input_generator, fine_tune_lr, fine_tune_epochs, steps_per_epoch, model_dir,
                    'hd_dice',
                    checkpoint,
                    extra_callbacks=[my_alpha_scheduler],
                    reinitialise_momentum=True,
                    validation_data=val_generator,
                    phase=phase,
                    loss_manager=loss_manager)


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
                loss_manager=None):

    # prepare model and log folders
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_%s_{epoch:03d}_%d.h5' % (metric_type, phase, n_epochs))
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1), KC.CSVLogger(os.path.join(log_dir, 'training.log'))]

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
			validation_steps=100)
