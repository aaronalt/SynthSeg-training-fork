"""
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
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# third-party imports
from ext.lab2im import layers


def metrics_model(input_model, label_list, metrics='dice', class_weights=None,
                  loss_weights=None, hd95_weight=0.0):
    last_tensor = input_model.outputs[0]
    input_shape = last_tensor.get_shape().as_list()[1:]

    n_labels = input_shape[-1]
    label_list = np.unique(label_list)
    assert n_labels == len(label_list), 'label_list should be as long as the posteriors channels'

    labels_gt = input_model.get_layer('labels_out').output
    labels_gt = layers.ConvertLabels(label_list)(labels_gt)
    labels_gt = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels_gt)
    labels_gt = KL.Reshape(input_shape)(labels_gt)

    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    labels_gt._keras_shape = tuple(labels_gt.get_shape().as_list())

    # normalise metrics and weights
    if isinstance(metrics, str):
        metrics = [metrics]
        loss_weights = [1.0]
    loss_weights = loss_weights or [1.0] * len(metrics)
    assert len(metrics) == len(loss_weights)

    # Moment/CE need true probabilities, not logits
    needs_prob = any(m in ('moment', 'ce', 'crossentropy') for m in metrics)
    pred_prob = KL.Activation('softmax', name='prediction_softmax')(last_tensor) if needs_prob else last_tensor

    weighted_losses = []

    for metric, weight in zip(metrics, loss_weights):

        if metric == 'dice':
            loss = layers.DiceLoss(
                class_weights=class_weights,
                boundary_weights=5,
                boundary_dist=1,
                skip_background=True,
                hd95_weight=hd95_weight
            )([labels_gt, last_tensor])  # keep logits path for Dice/HD95

        elif metric in ('ce', 'crossentropy'):
            loss = layers.CrossEntropyLoss(
                class_weights=class_weights,
                boundary_weights=5,
                boundary_dist=1,
                skip_background=True
            )([labels_gt, pred_prob])  # probabilities

        elif metric == 'moment':
            loss = layers.MomentLoss(
                class_weights=class_weights,
                enable_checks=True
            )([labels_gt, pred_prob])  # probabilities

        elif metric == 'wl2':
            loss = layers.WeightedL2Loss(target_value=5)([labels_gt, last_tensor])  # logits

        else:
            raise Exception('metrics should be dice, ce, moment or wl2, got {}'.format(metric))

        weighted = KL.Lambda(lambda x, w=weight: w * x,
                             name='{}_weighted'.format(metric))(loss)
        weighted_losses.append(weighted)

    total_loss = weighted_losses[0] if len(weighted_losses) == 1 else KL.Add(name='total_loss')(weighted_losses)

    model = Model(inputs=input_model.inputs, outputs=total_loss)

    for metric, weighted_loss in zip(metrics, weighted_losses):
        model.add_metric(weighted_loss, name='{}_loss'.format(metric), aggregation='mean')

    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
