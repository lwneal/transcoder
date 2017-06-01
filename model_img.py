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


def build_encoder(is_discriminator=False, **params):
    thought_vector_size = params['thought_vector_size']
    pretrained_encoder = params['pretrained_encoder']
    img_width = params['img_width']

    include_top = False
    LEARNABLE_CNN_LAYERS = 1

    input_shape = (img_width, img_width, IMG_CHANNELS)
    input_img = layers.Input(shape=input_shape)
    if pretrained_encoder:
        if pretrained_encoder == 'vgg16':
            cnn = applications.vgg16.VGG16(input_tensor=input_img, include_top=include_top)
        elif pretrained_encoder == 'resnet50':
            # Note: This is a hacked version of resnet50 with pooling removed
            cnn = resnet50.ResNet50(include_top=include_top, pooling=None)
        for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
            layer.trainable = False
        x = cnn(input_img)
    else:
        x = layers.Conv2D(64, (3,3), padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(LeakyReLU())(x)
        x = layers.MaxPooling2D()(x)

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

    # Expand vector from 1x1 to NxN
    N = img_width / 4
    x = layers.Reshape((1, 1, -1))(x_input)
    x = layers.Conv2DTranspose(64, (N, N), strides=(N, N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(LeakyReLU())(x)

    x = SpatialCGRU(x, 256)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(LeakyReLU())(x)

    #x = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(LeakyReLU())(x)

    x = SpatialCGRU(x, 128)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(LeakyReLU())(x)

    #x = layers.Conv2DTranspose(3, (3,3), strides=(2,2), padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = SpatialCGRU(x, 32)
    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.Activation('sigmoid')(x)

    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    return build_encoder(is_discriminator=True, **params)
