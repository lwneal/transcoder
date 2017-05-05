import os
import sys
import random
import numpy as np
from keras import layers, models

import model_words
import model_img
from dataset_word import WordDataset
from dataset_img import ImageRegionDataset


def get_batch(encoder_dataset, decoder_dataset, **params):
    X_list = encoder_dataset.empty_batch(**params)
    Y_list = decoder_dataset.empty_batch(**params)
    batch_size = params['batch_size']
    for i in range(batch_size):
        idx = encoder_dataset.random_idx()
        x_list = encoder_dataset.get_example(idx, **params)
        for X, x in zip(X_list, x_list):
            X[i] = x
        y_list = decoder_dataset.get_example(idx, **params)
        for Y, y in zip(Y_list, y_list):
            Y[i] = y
    # TODO: Can X and Y be the same shape?
    Y = np.expand_dims(Y, axis=-1)
    return X_list, Y


def generate(encoder_dataset, decoder_dataset, **params):
    while True:
        yield get_batch(encoder_dataset, decoder_dataset, **params)


def train(model, encoder_dataset, decoder_dataset, **params):
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)
    X, Y = next(training_gen)
    model.fit_generator(training_gen, steps_per_epoch=batches_per_epoch)


def demonstrate(model, encoder_dataset, decoder_dataset, input_text=None, **params):
    batch_size = params['batch_size']
    X_list = encoder_dataset.empty_batch(**params)
    for i in range(batch_size):
        if input_text:
            x_list = encoder_dataset.format_input(input_text, **params)
        else:
            x_list = encoder_dataset.get_example(**params)
        for X, x in zip(X_list, x_list):
            X[i] = x
    Y = model.predict(X_list)
    for x, y in zip(X_list[0], Y):
        left = encoder_dataset.unformat_input(x)
        right = decoder_dataset.unformat_output(y)
        print('{} --> {}'.format(left, right))


def build_model(encoder_dataset, decoder_dataset, **params):
    if params['encoder_type'] == 'region':
        encoder = model_img.build_encoder(encoder_dataset, **params)
    else:
        encoder = model_words.build_encoder(encoder_dataset, **params)

    decoder = model_words.build_decoder(decoder_dataset, **params)

    combined = models.Sequential()
    combined.add(encoder)
    combined.add(decoder)
    return encoder, decoder, combined


def main(**params):
    print("Loading dataset")
    if params['encoder_type'] == 'region':
        encoder_dataset = ImageRegionDataset(params['encoder_input_filename'], encoder=True, **params)
    else:
        encoder_dataset = WordDataset(params['encoder_input_filename'], encoder=True, **params)
    decoder_dataset = WordDataset(params['decoder_input_filename'], **params)
    print("Dataset loaded")

    print("Building model")
    encoder, decoder, combined = build_model(encoder_dataset, decoder_dataset, **params)
    print("Model built")

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])

    combined.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    if params['mode'] == 'train':
        for epoch in range(params['epochs']):
            train(combined, encoder_dataset, decoder_dataset, **params)
            demonstrate(combined, encoder_dataset, decoder_dataset, **params)
            encoder.save_weights(params['encoder_weights'])
            decoder.save_weights(params['decoder_weights'])
    elif params['mode'] == 'demo':
        print("Demonstration time!")
        params['batch_size'] = 1
        while True:
            inp = raw_input("Type a complete sentence in the input language: ")
            inp = inp.decode('utf-8').lower()
            demonstrate(combined, encoder_dataset, decoder_dataset, input_text=inp, **params)
