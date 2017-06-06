import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf
import model_img


def build_encoder(**params):
    thought_vector_size = params['thought_vector_size']

    # Global Image features (convnet output for the whole image)
    image_global = model_img.build_encoder(pooling='avg', **params)

    # Local Image features (convnet output for just the bounding box)
    image_local = model_img.build_encoder(pooling='avg', **params)

    # Context Vector input
    # normalized to [0,1] the values:
    # left, top, right, bottom, (box area / image area)
    ctx = layers.Input(shape=(5,))

    glob = layers.Dense(512)(image_global.output)
    loc = layers.Dense(512)(image_local.output)
    x = layers.Concatenate()([glob, loc, ctx])
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Concatenate()([x, ctx])
    x = layers.Dense(thought_vector_size)(x)
    x = layers.Activation('tanh')(x)

    return models.Model(inputs=[image_global.input, image_local.input, ctx], outputs=x)


def build_decoder(**params):
    # TODO: From a thought vector, generate a sentence, and image, and a bounding box
    pass


def build_discriminator(**params):
    # TODO: From a thought vector, generate a sentence, and image, and a bounding box
    pass
