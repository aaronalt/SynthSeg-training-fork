"""

This script explains how the different parameters controlling the generation of the synthetic data.
These parameters will be reused in the training function, but we describe them here, as the synthetic images are saved,
and thus can be visualised.
Note that most of the parameters here are set to their default value, but we show them nonetheless, just to explain
their effect. Moreover, we encourage the user to play with them to get a sense of their impact on the generation.



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


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ext.lab2im import utils
import numpy as np
from SynthSeg.brain_generator import BrainGenerator

# script parameters
n_examples = 5  # number of examples to generate in this script
result_dir = './outputs'  # folder where examples will be saved


# ---------- Input label maps and associated values ----------

# folder containing label maps to generate images from (note that they must have a ".nii", ".nii.gz" or ".mgz" format)
path_label_map = './data/training_label_maps_claustrum'

# Here we specify the structures in the label maps for which we want to generate intensities.
# This is given as a list of label values, which do not necessarily need to be present in every label map.
# However, these labels must follow a specific order: first the background, and then all the other labels. Moreover, if
# 1) the label maps contain some right/left-specific label values, and 2) we activate flipping augmentation (which is
# true by default), then the rest of the labels must follow a strict order:
# first the non-sided labels (i.e. those which are not right/left specific), then all the left labels, and finally the
# corresponding right labels (in the same order as the left ones). Please make sure each that each sided label has a
# right and a left value (this is essential!!!).
#
# Example: generation_labels = [0,    # background
#                               24,   # CSF
#                               507,  # extra-cerebral soft tissues
#                               2,    # left white matter
#                               3,    # left cerebral cortex
#                               4,    # left lateral ventricle
#                               17,   # left hippocampus
#                               25,   # left lesions
#                               41,   # right white matter
#                               42,   # right cerebral cortex
#                               43,   # right lateral ventricle
#                               53,   # right hippocampus
#                               57]   # right lesions
# Note that plenty of structures are not represented here..... but it's just an example ! :)
generation_labels = np.array([0, 7, 48, 68, 79, 99, 100, 101, 103, 108, 111, 113, 117, 118, 119, 120, 125, 128, 130, 147, 149, 150, 157, 161, 181, 184, 190, 191, 192, 193, 194, 196, 199, 201, 206, 207, 208, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 234, 238, 240, 242, 243, 244, 245, 246, 252, 253, 254, 255, 256, 268, 274, 275, 276, 277, 278, 279, 282, 283, 284, 285, 286, 295, 297, 298, 301, 303, 305, 306, 307, 309, 310, 312, 313, 314, 315, 316, 320, 321, 322, 326, 339, 340, 341, 342, 343, 344, 345, 346, 347, 349, 350, 352, 354, 364, 365, 367, 368, 369, 370, 371, 372, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 393, 394, 395, 396, 397, 398, 399, 404, 405, 407, 408, 409, 410, 411, 412, 414, 418, 419, 420, 421, 422, 423, 424, 425, 426, 430, 432, 435, 441, 442, 443, 444, 454, 458, 461, 465, 478, 479, 484, 492, 493, 496, 498, 504, 506, 508, 517, 519, 521, 541, 558, 559, 561, 562, 563, 564, 565, 566, 567, 568, 569, 575, 576, 578, 580, 611, 811, 813, 843, 2001, 2002, 2003, 2006, 2007, 2009, 2012, 2014, 2015, 2016, 2018, 2019, 2020, 2022, 2023, 2024, 2026, 2027, 2028, 2030, 2031, 2033, 2034, 2035, 138])

# We also have to specify the number of non-sided labels in order to differentiate them from the labels with
# right/left values.
# Example: (continuing the previous one): in this example it would be 3 (background, CSF, extra-cerebral soft tissues).
# n_neutral_labels = len(generation_labels)
n_neutral_labels = 224

# By default, the output label maps (i.e. the target segmentations) contain all the labels used for generation.
# However, we may want not to predict all the generation labels (e.g. extra-cerebral soft tissues).
# For this reason, we specify here the target segmentation label corresponding to every generation structure.
# This new list must have the same length as generation_labels, and follow the same order.
#
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
# Note that in this example the labels 24 (CSF), and 507 (extra-cerebral soft tissues) are not predicted, or said
# differently they are segmented as background.
# Also, the left and right lesions (labels 25 and 57) are segmented as left and right white matter (labels 2 and 41).
# ---------- Shape and resolution of the outputs ----------

# number of channel to synthesise for multi-modality settings. Set this to 1 (default) in the uni-modality scenario.
n_channels = 1

# We have the possibility to generate training examples at a different resolution than the training label maps (e.g.
# when using ultra HR training label maps). Here we want to generate at the same resolution as the training label maps,
# so we set this to None.
target_res = None

# The generative model offers the possibility to randomly crop the training examples to a given size.
# Here we crop them to 160^3, such that the produced images fit on the GPU during training.
output_shape = (160, 160, 160)


# ---------- GMM sampling parameters ----------

# Here we use uniform prior distribution to sample the means/stds of the GMM. Because we don't specify prior_means and
# prior_stds, those priors will have default bounds of [0, 250], and [0, 35]. Those values enable to generate a wide
# range of contrasts (often unrealistic), which will make the segmentation network contrast-agnostic.
prior_distributions = 'uniform'

# We regroup labels with similar tissue types into K "classes", so that intensities of similar regions are sampled
# from the same Gaussian distribution. This is achieved by providing a list indicating the class of each label.
# It should have the same length as generation_labels, and follow the same order. Importantly the class values must be
# between 0 and K-1, where K is the total number of different classes.
#
# Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
#                                        generation_classes = [0,  1,   2, 3, 4, 5,  4,  6,  7,  8,  9,  8, 10]
# In this example labels 3 and 17 are in the same *class* 4 (that has nothing to do with *label* 4), and thus will be
# associated to the same Gaussian distribution when sampling the GMM.
# Tissue type groupings for your labels
generation_classes = np.array([
    0,    # 0
    1,    # 7
    2,    # 48
    3,    # 68
    2,    # 79
    4,    # 99
    4,    # 100
    2,    # 101
    5,    # 103
    5,    # 108
    1,    # 111
    1,    # 113
    5,    # 117
    2,    # 118
    2,    # 119
    2,    # 120
    4,    # 125
    5,    # 128
    1,    # 130
    5,    # 147
    5,    # 149
    5,    # 150
    4,    # 157
    1,    # 161
    5,    # 181
    1,    # 184
    6,    # 190
    6,    # 191
    5,    # 192
    5,    # 193
    5,    # 194
    5,    # 196
    1,    # 199
    1,    # 201
    2,    # 206
    5,    # 207
    1,    # 208
    6,    # 214
    6,    # 215
    6,    # 216
    6,    # 217
    6,    # 218
    6,    # 219
    6,    # 220
    6,    # 221
    6,    # 222
    6,    # 223
    6,    # 224
    6,    # 225
    6,    # 226
    6,    # 227
    5,    # 228
    5,    # 229
    5,    # 230
    1,    # 232
    6,    # 234
    6,    # 238
    6,    # 240
    6,    # 242
    6,    # 243
    6,    # 244
    6,    # 245
    6,    # 246
    5,    # 252
    5,    # 253
    5,    # 254
    5,    # 255
    6,    # 256
    6,    # 268
    6,    # 274
    6,    # 275
    6,    # 276
    6,    # 277
    6,    # 278
    6,    # 279
    6,    # 282
    6,    # 283
    6,    # 284
    6,    # 285
    6,    # 286
    6,    # 295
    5,    # 297
    1,    # 298
    7,    # 301
    6,    # 303
    6,    # 305
    6,    # 306
    6,    # 307
    1,    # 309
    6,    # 310
    6,    # 312
    6,    # 313
    6,    # 314
    6,    # 315
    6,    # 316
    7,    # 320
    1,    # 321
    1,    # 322
    7,    # 326
    7,    # 339
    7,    # 340
    7,    # 341
    7,    # 342
    7,    # 343
    7,    # 344
    7,    # 345
    7,    # 346
    7,    # 347
    2,    # 349
    6,    # 350
    6,    # 352
    1,    # 354
    7,    # 364
    7,    # 365
    7,    # 367
    7,    # 368
    7,    # 369
    7,    # 370
    7,    # 371
    7,    # 372
    7,    # 373
    7,    # 374
    7,    # 375
    6,    # 377
    6,    # 378
    6,    # 379
    6,    # 380
    6,    # 381
    6,    # 382
    8,    # 384
    8,    # 385
    2,    # 393
    6,    # 394
    6,    # 395
    6,    # 396
    6,    # 397
    6,    # 398
    6,    # 399
    7,    # 404
    7,    # 405
    7,    # 407
    7,    # 408
    7,    # 409
    7,    # 410
    7,    # 411
    1,    # 412
    8,    # 414
    7,    # 418
    7,    # 419
    7,    # 420
    7,    # 421
    7,    # 422
    6,    # 423
    6,    # 424
    6,    # 425
    6,    # 426
    6,    # 430
    7,    # 432
    8,    # 435
    6,    # 441
    6,    # 442
    6,    # 443
    6,    # 444
    6,    # 454
    6,    # 458
    1,    # 461
    8,    # 465
    6,    # 478
    6,    # 479
    6,    # 484
    6,    # 492
    1,    # 493
    8,    # 496
    8,    # 498
    8,    # 504
    6,    # 506
    8,    # 508
    6,    # 517
    6,    # 519
    8,    # 521
    8,    # 541
    7,    # 558
    7,    # 559
    7,    # 561
    7,    # 562
    7,    # 563
    7,    # 564
    7,    # 565
    7,    # 566
    7,    # 567
    7,    # 568
    7,    # 569
    7,    # 575
    7,    # 576
    6,    # 578
    8,    # 580
    9,    # 611
    6,    # 811
    6,    # 813
    5,    # 843
    10,   # 2001
    10,   # 2002
    10,   # 2003
    10,   # 2006
    10,   # 2007
    10,   # 2009
    10,   # 2012
    10,   # 2014
    10,   # 2015
    10,   # 2016
    10,   # 2018
    10,   # 2019
    10,   # 2020
    10,   # 2022
    10,   # 2023
    10,   # 2024
    10,   # 2026
    10,   # 2027
    10,   # 2028
    10,   # 2030
    10,   # 2031
    10,   # 2033
    10,   # 2034
    10,   # 2035
    11    # 138
])
# ---------- Spatial augmentation ----------

# We now introduce some parameters concerning the spatial deformation. They enable to set the range of the uniform
# distribution from which the corresponding parameters are selected.
# We note that because the label maps will be resampled with nearest neighbour interpolation, they can look less smooth
# than the original segmentations.

flipping = True  # enable right/left flipping
scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
nonlin_std = 4.  # this controls the maximum elastic deformation (higher = more deformation)
bias_field_std = 0.7  # this controls the maximum bias field corruption (higher = more bias)


# ---------- Resolution parameters ----------

# This enables us to randomise the resolution of the produces images.
# Although being only one parameter, this is crucial !!
randomise_res = True


# ------------------------------------------------------ Generate ------------------------------------------------------

# instantiate BrainGenerator object
brain_generator = BrainGenerator(labels_dir=path_label_map,
                                 generation_labels=generation_labels,
                                 n_neutral_labels=n_neutral_labels,
                                 prior_distributions=prior_distributions,
                                 generation_classes=generation_classes,
                                 output_labels=output_labels,
                                 n_channels=n_channels,
                                 target_res=target_res,
                                 output_shape=output_shape,
                                 flipping=flipping,
                                 scaling_bounds=scaling_bounds,
                                 rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds,
                                 translation_bounds=translation_bounds,
                                 nonlin_std=nonlin_std,
                                 bias_field_std=bias_field_std,
                                 randomise_res=randomise_res
                                 )

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_brain()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'image_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'labels_%s.nii.gz' % n))
