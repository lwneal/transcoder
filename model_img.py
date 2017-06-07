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


def build_encoder(is_discriminator=False, pooling=None, **params):
    thought_vector_size = params['thought_vector_size']
    img_width = params['img_width']
    pretrained_encoder = params['pretrained_encoder']
    cgru_size = params['csr_size']
    cgru_layers = params['csr_layers']

    include_top = False
    LEARNABLE_CNN_LAYERS = 1

    # Image input shape is 256x256
    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), padding='same')(input_img)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)
    # Shape is 128x128x64

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    if not is_discriminator:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    if cgru_layers >= 1:
        x = QuadCSR(cgru_size)(x)
    else:
        x = layers.Conv2D(cgru_size * 3 / 2, (3,3), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D()(x)
    # Shape is 64x64x???

    if is_discriminator:
        x = layers.Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        # Shape is 64x64xthought_vector_size
        x = layers.Conv2D(thought_vector_size, (1,1), padding='same', activation='tanh')(x)
    return models.Model(inputs=[input_img], outputs=x)


def build_decoder(**params):
    thought_vector_size = params['thought_vector_size']
    csr_size = params['csr_size']
    csr_layers = params['csr_layers']
    img_width = params['img_width']

    x_input = layers.Input(shape=(img_width/4, img_width/4, thought_vector_size,))
    x = x_input

    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(x)
    x = layers.Conv2DTranspose(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if csr_layers > 0:
        x = QuadCSR(64)(x)

    x = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(x)
    x = layers.Conv2DTranspose(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    if csr_layers > 1:
        x = QuadCSR(32)(x)

    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    return build_encoder(is_discriminator=True, **params)
