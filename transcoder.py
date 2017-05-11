import os
import sys
import random
import numpy as np
from keras import layers, models

import model_words
import model_img
import model_img_region
from dataset_word import WordDataset
from dataset_img import ImageDataset
from dataset_img_region import ImageRegionDataset


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
    return X_list, Y_list


def generate(encoder_dataset, decoder_dataset, **params):
    # HACK: X and Y should each be a numpy array...
    # Unless the model takes multiple inputs/outputs
    def unpack(Z):
        if isinstance(Z, list) and len(Z) == 1:
            return Z[0]
        return Z

    while True:
        X, Y = get_batch(encoder_dataset, decoder_dataset, **params)
        yield unpack(X), unpack(Y)


def train(model, encoder_dataset, decoder_dataset, **params):
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)
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
    X = X_list[0]
    for x, y in zip(X, Y):
        left = encoder_dataset.unformat_input(x)
        right = decoder_dataset.unformat_output(y)
        print('{} --> {}'.format(left, right))


def dataset_for_extension(ext):
    if ext == 'img':
        return ImageDataset
    elif ext == 'bbox':
        return ImageRegionDataset
    return WordDataset


def build_model(encoder_dataset, decoder_dataset, **params):
    encoder = encoder_dataset.build_model(**params)
    decoder = decoder_dataset.build_model(**params)

    if params['freeze_encoder']:
        for layer in encoder.layers:
            layer.trainable = False
    if params['freeze_decoder']:
        for layer in decoder.layers:
            layer.trainable = False

    combined = models.Sequential()
    combined.add(encoder)
    combined.add(decoder)
    return encoder, decoder, combined


def main(**params):
    print("Loading datasets...")
    ds = dataset_for_extension(params['encoder_input_filename'].split('.')[-1])
    encoder_dataset = ds(params['encoder_input_filename'], encoder=True, **params)

    ds = dataset_for_extension(params['decoder_input_filename'].split('.')[-1])
    decoder_dataset = ds(params['decoder_input_filename'], encoder=False, **params)


    print("Building models...")
    encoder, decoder, combined = build_model(encoder_dataset, decoder_dataset, **params)
    combined.summary()

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])
    combined.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if params['mode'] == 'train':
        print("Training...")
        for epoch in range(params['epochs']):
            train(combined, encoder_dataset, decoder_dataset, **params)
            demonstrate(combined, encoder_dataset, decoder_dataset, **params)
            encoder.save_weights(params['encoder_weights'])
            decoder.save_weights(params['decoder_weights'])
    elif params['mode'] == 'demo':
        print("Starting Demonstration...")
        params['batch_size'] = 1
        while True:
            inp = raw_input("Type a complete sentence in the input language: ")
            inp = inp.decode('utf-8').lower()
            demonstrate(combined, encoder_dataset, decoder_dataset, input_text=inp, **params)
