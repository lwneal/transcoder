import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf


import model_img
import model_words


def build_encoder(dataset, **params):
    thought_vector_size = params['thought_vector_size']

    visual_model = model_img.build_encoder(**params)
    words_model = model_words.build_encoder(dataset, **params)

    x = layers.Concatenate()([visual_model.output, words_model.output])
    x = layers.BatchNormalization()(x)
    x = layers.Activation(layers.LeakyReLU())(x)
    x = layers.Dense(thought_vector_size)(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs=[visual_model.input, words_model.input], outputs=x)


def build_decoder(dataset, **params):
    return model_words.build_decoder(dataset, **params)


def build_discriminator(dataset, **params):
    return model_words.build_discriminator(dataset, **params)
