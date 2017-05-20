import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf

from cgru import SpatialCGRU
from dataset_img import IMG_WIDTH, IMG_HEIGHT

IMG_CHANNELS = 3


def build_encoder(**params):
    THOUGHT_VECTOR_SIZE = params['thought_vector_size']
    BATCH_SIZE = params['batch_size']

    # TODO: more params
    CNN = 'resnet50'
    include_top = False
    LEARNABLE_CNN_LAYERS = 1
    ACTIVATION = 'relu'

    if CNN == 'vgg16':
        cnn = applications.vgg16.VGG16(include_top=include_top)
    elif CNN == 'resnet50':
        cnn = applications.resnet50.ResNet50(include_top=include_top, pooling=None)

    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Global Image featuers (convnet output for the whole image)
    input_img = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = cnn(input_img)

    if not include_top:
        x = layers.Flatten()(x)

    x = layers.Dense(THOUGHT_VECTOR_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(THOUGHT_VECTOR_SIZE)(x)

    return models.Model(inputs=[input_img], outputs=x)


def build_decoder(**params):
    thought_vector_size = params['thought_vector_size']
    batch_size = params['batch_size']
    cgru_size = params['cgru_size']

    x_input = layers.Input(batch_shape=(batch_size, thought_vector_size))

    # Expand vector from 1x1 to NxN
    N = IMG_WIDTH / 8
    x = layers.Reshape((1,1,-1))(x_input)
    x = layers.convolutional.UpSampling2D((N,N))(x)
    x = layers.Conv2D(512, (N,N), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # TODO: optional CGRU

    # Upsample from NxN to IMG_WIDTH
    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)

    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    D = models.Sequential()
    D.add(layers.Conv2D(64, (3,3), padding='same', input_shape=input_shape))
    D.add(layers.BatchNormalization())
    D.add(layers.Activation('relu'))
    D.add(layers.MaxPooling2D())
    D.add(layers.Conv2D(128, (3,3), padding='same'))
    D.add(layers.BatchNormalization())
    D.add(layers.Activation('relu'))
    D.add(layers.MaxPooling2D())
    D.add(layers.Conv2D(256, (3,3), padding='same'))
    D.add(layers.BatchNormalization())
    D.add(layers.Activation('relu'))
    D.add(layers.MaxPooling2D())
    D.add(layers.Flatten())
    D.add(layers.Dense(128, activation='relu'))
    D.add(layers.Dense(1, activation='tanh'))
    return D

