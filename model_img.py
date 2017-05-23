import os
import sys
import random
import numpy as np
from keras import layers, models, applications
from keras.layers.advanced_activations import LeakyReLU

import resnet50
import tensorflow as tf

from cgru import SpatialCGRU
from dataset_img import IMG_WIDTH, IMG_HEIGHT

IMG_CHANNELS = 3


def build_encoder(**params):
    thought_vector_size = params['thought_vector_size']
    batch_size = params['batch_size']

    # TODO: more params
    CNN = 'resnet50'
    include_top = False
    LEARNABLE_CNN_LAYERS = 1
    ACTIVATION = 'relu'

    input_batch_shape = (batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    input_img = layers.Input(batch_shape=input_batch_shape)
    if CNN == 'vgg16':
        cnn = applications.vgg16.VGG16(input_tensor=input_img, include_top=include_top)
    elif CNN == 'resnet50':
        # Note: This is a hacked version of resnet50 with pooling removed
        cnn = resnet50.ResNet50(include_top=include_top, pooling=None)

    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Global Image featuers (convnet output for the whole image)
    x = cnn(input_img)

    if not include_top:
        x = layers.Flatten()(x)

    x = layers.Dense(thought_vector_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return models.Model(inputs=[input_img], outputs=x)


def build_decoder(**params):
    thought_vector_size = params['thought_vector_size']
    batch_size = params['batch_size']
    cgru_size = params['cgru_size']
    cgru_layers = params['cgru_layers']

    x_input = layers.Input(batch_shape=(batch_size, thought_vector_size))

    # Expand vector from 1x1 to NxN
    N = IMG_WIDTH / 8
    x = layers.Reshape((1, 1, -1))(x_input)
    x = layers.Conv2DTranspose(64, (N, N), strides=(N, N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(cgru_layers):
        x = SpatialCGRU(x, cgru_size)

    x = layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(3, (5,5), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)

    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    batch_size = params['batch_size']
    batch_input_shape = (batch_size, IMG_HEIGHT, IMG_WIDTH, 3)

    img_input = layers.Input(batch_shape=batch_input_shape)
    x = layers.Conv2D(64, (3,3), padding='same')(img_input)
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
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=LeakyReLU())(x)
    x = layers.Dense(1, activation='tanh')(x)

    return models.Model(inputs=img_input, outputs=x)

