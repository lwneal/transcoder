import os
import sys
import random
import numpy as np
from keras import layers, models


def build_encoder(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words']
    thought_vector_size = params['thought_vector_size']
    vocab_len = len(dataset.vocab)

    inp = layers.Input(shape=(max_words,), dtype='int32')
    x = layers.Embedding(vocab_len, wordvec_size, input_length=max_words, mask_zero=True)(inp)
    for _ in range(rnn_layers - 1):
        x = rnn_type(rnn_size, return_sequences=True)(x)
    x = rnn_type(rnn_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    encoded = layers.Dense(thought_vector_size, activation='tanh')(x)
    moo = models.Model(inputs=inp, outputs=encoded)
    return moo


def build_decoder(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words']
    thought_vector_size = params['thought_vector_size']
    vocab_len = len(dataset.vocab)

    inp = layers.Input(shape=(thought_vector_size,))
    x = layers.RepeatVector(max_words)(inp)
    for _ in range(rnn_layers - 1):
        x = rnn_type(rnn_size, return_sequences=True)(x)
    x = rnn_type(rnn_size, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(vocab_len))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('softmax')(x)
    
    return models.Model(inputs=inp, outputs=x)
