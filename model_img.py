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
    pretrained_encoder = params['pretrained_encoder']
    img_width = params['img_width']
    cgru_size = params['csr_size']
    cgru_layers = params['csr_layers']

    include_top = False
    LEARNABLE_CNN_LAYERS = 1

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)
    if pretrained_encoder:
        if pretrained_encoder == 'vgg16':
            cnn = applications.vgg16.VGG16(input_tensor=input_img, include_top=include_top)
        elif pretrained_encoder == 'resnet50':
            # Note: This is a hacked version of resnet50 with pooling removed
            cnn = resnet50.ResNet50(include_top=include_top, pooling=pooling)
        for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
            layer.trainable = False
        x = cnn(input_img)
    else:
        x = layers.Conv2D(64, (3,3), padding='same')(input_img)
        if not is_discriminator:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)

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

        x = layers.Conv2D(256, (3,3), padding='same')(x)
        if not is_discriminator:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(384, (3,3), padding='same')(x)
        if not is_discriminator:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        x = layers.Conv2D(384, (3,3), padding='same')(x)
        if not is_discriminator:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)

    if not pooling:
        x = layers.Flatten()(x)
    if is_discriminator:
        x = layers.Dense(1)(x)
        x = layers.Activation('tanh')(x)
    else:
        x = layers.Dense(thought_vector_size)(x)
        x = layers.BatchNormalization()(x)
    return models.Model(inputs=[input_img], outputs=x)


def build_decoder(**params):
    thought_vector_size = params['thought_vector_size']
    csr_size = params['csr_size']
    csr_layers = params['csr_layers']
    img_width = params['img_width']

    x_input = layers.Input(shape=(thought_vector_size,))

    # Expand vector from 1x1 to 4x4
    N = 4
    x = layers.Reshape((1, 1, -1))(x_input)
    x = layers.Conv2DTranspose(128, (N, N), strides=(N, N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(LeakyReLU())(x)

    # Upsample to the desired width (powers of 2 only)
    while N < img_width:
        x = layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        if csr_layers > 0:
            x = QuadCSR(csr_size)(x)
            csr_layers -= 1
        N *= 2

    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    return build_encoder(is_discriminator=True, **params)
