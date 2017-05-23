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
    max_words = params['max_words_encoder']
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
    moo = models.Model(inputs=inp, outputs=encoded, name='word_encoder')
    return moo


def build_decoder(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words_decoder']
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
    
    return models.Model(inputs=inp, outputs=x, name='word_decoder')


def build_discriminator(dataset, **params):
    wordvec_size = params['wordvec_size']
    max_words = params['max_words_decoder']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    vocab_len = len(dataset.vocab)

    input_shape = (max_words, vocab_len)
    x_in = layers.Input(shape=input_shape)
    x = layers.Dense(wordvec_size)(x_in)
    x = layers.LSTM(rnn_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dense(wordvec_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    return models.Model(inputs=x_in, outputs=x, name='word_discriminator')
