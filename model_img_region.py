import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf


def build_encoder(**params):
    # Global Image featuers (convnet output for the whole image)
    image_global = model_img.build_encoder(**params)

    # Local Image featuers (convnet output for just the bounding box)
    image_local = model_img.build_encoder(**params)

    # Context Vector input
    # normalized to [0,1] the values:
    # left, top, right, bottom, (box area / image area)
    input_ctx = layers.Input(batch_shape=(BATCH_SIZE, 5))
    ctx = layers.BatchNormalization()(input_ctx)

    x = layers.Concatenate()([image_global, image_local, ctx])
    x = layers.Dense(THOUGHT_VECTOR_SIZE)(x)
    x = layers.Activation('tanh')(x)

    return models.Model(inputs=[image_global.input, image_local.input, input_ctx], outputs=x)
