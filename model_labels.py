import os
import sys
import random
import numpy as np
from keras import layers, models


def build_encoder(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # Just an embedding from each label to a point in latent space
    x_in = layers.Input(shape=(label_count,))
    x = layers.Dense(thought_vector_size)(x)
    x = layers.Activation('tanh')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_encoder')


def build_decoder(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # Linear mapping from thought vector to labels
    x_in = layers.Input(shape=(thought_vector_size,))
    x = layers.Dense(label_count)(x_in)
    x = layers.Activation('softmax')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_decoder')


def build_discriminator(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # A discriminator on labels doesn't usually make any sense
    x_in = layers.Input(shape=(label_count,))
    x = layers.Dense(1)(x_in)
    x = layers.Activation('tanh')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_encoder')
