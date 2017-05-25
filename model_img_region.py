import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf

from dataset_img import IMG_HEIGHT, IMG_WIDTH

IMG_CHANNELS = 3


def build_encoder(**params):
    THOUGHT_VECTOR_SIZE = params['thought_vector_size']
    BATCH_SIZE = params['batch_size']
    CNN = 'vgg16'
    INCLUDE_TOP = True
    LEARNABLE_CNN_LAYERS = 1
    ACTIVATION = 'relu'

    if CNN == 'vgg16':
        cnn = applications.vgg16.VGG16(include_top=INCLUDE_TOP)
        if INCLUDE_TOP:
            # Pop the softmax layer
            cnn = models.Model(inputs=cnn.inputs, outputs=cnn.layers[-1].output)
    elif CNN == 'resnet':
        cnn = applications.resnet50.ResNet50(include_top=INCLUDE_TOP)
        # Pop the mean pooling layer
        cnn = models.Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)

    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Context Vector input
    # normalized to [0,1] the values:
    # left, top, right, bottom, (box area / image area)
    input_ctx = layers.Input(batch_shape=(BATCH_SIZE, 5))
    ctx = layers.BatchNormalization()(input_ctx)

    # Global Image featuers (convnet output for the whole image)
    input_img_global = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    image_global = cnn(input_img_global)

    # Local Image featuers (convnet output for just the bounding box)
    input_img_local = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    image_local = cnn(input_img_local)

    x = layers.Concatenate()([image_global, image_local, ctx])
    x = layers.Dense(THOUGHT_VECTOR_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return models.Model(inputs=[input_img_global, input_img_local, input_ctx], outputs=x)
