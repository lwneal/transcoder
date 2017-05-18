import os
import sys
import random
import numpy as np
from keras import layers, models, applications
import tensorflow as tf

IMG_CHANNELS = 3
IMG_HEIGHT = IMG_WIDTH = 224


def build_encoder(vocab_len, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words_encoder']
    thought_vector_size = params['thought_vector_size']

    CNN = 'vgg16'
    INCLUDE_TOP = True
    LEARNABLE_CNN_LAYERS = 1

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

    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x_img = cnn(input_img)

    input_words = layers.Input(shape=(max_words,), dtype='int32')
    x_words = layers.Embedding(vocab_len, wordvec_size, input_length=max_words, mask_zero=True)(input_words)
    for _ in range(rnn_layers - 1):
        x_words = rnn_type(rnn_size, return_sequences=True)(x_words)
    x_words = rnn_type(rnn_size)(x_words)
    x_words = layers.BatchNormalization()(x_words)
    x_words = layers.Activation('relu')(x_words)
    x_words = layers.Dense(thought_vector_size, activation='tanh')(x_words)

    x = layers.Concatenate()([x_img, x_words])
    x = layers.Dense(thought_vector_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)

    return models.Model(inputs=[input_img, input_words], outputs=x)
