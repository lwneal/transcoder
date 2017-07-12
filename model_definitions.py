import os
import sys
import random
import numpy as np
from keras import layers, models, applications
from keras.layers.advanced_activations import LeakyReLU

import resnet50
import tensorflow as tf

from csr import QuadCSR

IMG_CHANNELS = 3


def simplecnn_7a(is_discriminator=False, **params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_encoder']

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same')(input_img)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(384, (3,3), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    if is_discriminator:
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        x = layers.Dense(thought_vector_size)(x)
        x = layers.BatchNormalization()(x)
    return models.Model(inputs=[input_img], outputs=x)


def stridecnn_10a(is_discriminator=False, **params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_encoder']

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same')(input_img)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    if is_discriminator:
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        x = layers.Dense(thought_vector_size)(x)
        x = layers.BatchNormalization()(x)
    return models.Model(inputs=[input_img], outputs=x)


def stridecnn_11a(is_discriminator=False, **params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_encoder']

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same')(input_img)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(384, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)


    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(512, (3,3), strides=(2,2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    if is_discriminator:
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        x = layers.Dense(thought_vector_size)(x)
        x = layers.BatchNormalization()(x)
    return models.Model(inputs=[input_img], outputs=x)


def csrnn_7a(is_discriminator=False, **params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_encoder']

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same')(input_img)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = QuadCSR(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(384, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    if is_discriminator:
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        x = layers.Dense(thought_vector_size)(x)
        x = layers.BatchNormalization()(x)
    return models.Model(inputs=[input_img], outputs=x)


def simpledeconv_a(**params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_decoder']

    x_input = layers.Input(shape=(thought_vector_size,))

    # Expand vector from 1x1 to 4x4
    N = 4
    x = layers.Reshape((1, 1, -1))(x_input)
    x = layers.Conv2DTranspose(128, (N, N), strides=(N, N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Upsample to the desired width (powers of 2 only)
    while N < img_width:
        x = layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        N *= 2

    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(inputs=[x_input], outputs=x)


def csrnn_deconv_a(**params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width_decoder']

    x_input = layers.Input(shape=(thought_vector_size,))

    # Expand vector from 1x1 to 4x4
    N = 4
    x = layers.Reshape((1, 1, -1))(x_input)
    x = layers.Conv2DTranspose(128, (N, N), strides=(N, N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Upsample to the desired width (powers of 2 only)
    while N < img_width:
        x = layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = QuadCSR(128)(x)
        N *= 2

    x = layers.Conv2D(IMG_CHANNELS, (3,3), padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(inputs=[x_input], outputs=x)


def linear_tanh(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # Just an embedding from each label to a point in latent space
    x_in = layers.Input(shape=(label_count,))
    x = layers.Dense(thought_vector_size)(x_in)
    x = layers.Activation('tanh')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_encoder')


def linear_softmax(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # Linear mapping from thought vector to labels
    x_in = layers.Input(shape=(thought_vector_size,))
    x = layers.Dense(label_count)(x_in)
    x = layers.Activation('softmax')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_decoder')


def mlp_2a(dataset, **params):
    thought_vector_size = params['thought_vector_size']
    label_count = len(dataset.name_to_idx)

    # MLP with single hidden layer
    x_in = layers.Input(shape=(thought_vector_size,))

    hidden_units = label_count + thought_vector_size
    x = layers.Dense(hidden_units)(x_in)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(label_count)(x)
    x = layers.Activation('softmax')(x)
    return models.Model(inputs=x_in, outputs=x, name='label_decoder')
