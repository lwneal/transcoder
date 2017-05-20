import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf

from cgru import SpatialCGRU

IMG_CHANNELS = 3
IMG_HEIGHT = IMG_WIDTH = 224


def build_encoder(**params):
    THOUGHT_VECTOR_SIZE = params['thought_vector_size']
    BATCH_SIZE = params['batch_size']
    CNN = 'vgg16'
    INCLUDE_TOP = True
    LEARNABLE_CNN_LAYERS = 1
    ACTIVATION = 'relu'

    if CNN == 'vgg16':
        cnn = applications.vgg16.VGG16(include_top=INCLUDE_TOP)
    elif CNN == 'resnet':
        cnn = applications.resnet50.ResNet50(include_top=INCLUDE_TOP)
        # Pop the mean pooling layer
        cnn = models.Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)

    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Global Image featuers (convnet output for the whole image)
    input_img = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = cnn(input_img)

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

    # Expand vector from 1x1 to 7x7
    x = layers.Reshape((1,1,-1))(x_input)
    x = layers.convolutional.UpSampling2D((7,7))(x)
    x = layers.Conv2D(128, (7,7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # TODO: optional CGRU

    # Upsample from 7x7 to 224x224
    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.convolutional.UpSampling2D()(x)
    x = layers.Conv2D(3, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('sigmoid')(x)

    return models.Model(inputs=[x_input], outputs=x)


def build_discriminator(**params):
    input_shape = (224, 224, 3)
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

