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



def train(encoder, decoder, transcoder, discriminator, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)


    avg_loss = 0
    avg_accuracy = 0
    print("Training discriminator...")
    for i in range(batches_per_epoch):
        # Generate some fake data
        X_encoder, Y_decoder = next(training_gen)
        X_generated = transcoder.predict(X_encoder)

        # Shuffle real and generated examples into a batch
        X_disc = Y_decoder
        Y_disc = np.zeros(len(X_generated), dtype=np.int)
        for j in range(len(X_generated)):
            if np.random.rand() < .5:
                # Real example, label this 1
                Y_disc[j] = 1
            else:
                # Fake example, label this 0
                X_disc[j] = X_generated[j]
                Y_disc[j] = 0
        # Convert onehot to indices
        X_disc = np.argmax(X_disc, axis=-1)
        loss, accuracy = discriminator.train_on_batch(X_disc, Y_disc)
        avg_loss = .95 * avg_loss + .05 * loss
        avg_accuracy = .95 * avg_accuracy + .05 * accuracy
        sys.stderr.write("[K\r{}/{} batches, batch size {}, loss {:.3f}, accuracy {:.3f}".format(
            i, batches_per_epoch, batch_size, avg_loss, avg_accuracy))
    sys.stderr.write('\n')

    avg_loss = 0
    avg_accuracy = 0
    print("Training encoder/decoder...")
    for i in range(batches_per_epoch):
        X, Y = next(training_gen)
        loss, accuracy = transcoder.train_on_batch(X, Y)
        avg_loss = .95 * avg_loss + .05 * loss
        avg_accuracy = .95 * avg_accuracy + .05 * accuracy
        sys.stderr.write("[K\r{}/{} batches, batch size {}, loss {:.3f}, accuracy {:.3f}".format(
            i, batches_per_epoch, batch_size, avg_loss, avg_accuracy))
    sys.stderr.write('\n')

    print("Training epoch finished")



def demonstrate(encoder, decoder, encoder_dataset, decoder_dataset, input_text=None, **params):
    batch_size = params['batch_size']

    model = models.Sequential()
    model.add(encoder)
    model.add(decoder)

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

    transcoder = models.Sequential()
    transcoder.add(encoder)
    transcoder.add(decoder)
    transcoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    discriminator = models.Sequential()
    vocab_len = len(decoder_dataset.vocab)
    wordvec_size = 512
    max_words = params['max_words']
    discriminator.add(layers.Embedding(vocab_len, wordvec_size, input_length=max_words))
    discriminator.add(layers.LSTM(512))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.Activation('tanh'))
    discriminator.add(layers.Dense(512))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.Activation('tanh'))
    discriminator.add(layers.Dense(2))
    discriminator.add(layers.Activation('softmax'))
    discriminator.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return encoder, decoder, transcoder, discriminator


def main(**params):
    print("Loading datasets...")
    ds = dataset_for_extension(params['encoder_input_filename'].split('.')[-1])
    encoder_dataset = ds(params['encoder_input_filename'], encoder=True, **params)

    ds = dataset_for_extension(params['decoder_input_filename'].split('.')[-1])
    decoder_dataset = ds(params['decoder_input_filename'], encoder=False, **params)

    print("Building models...")
    encoder, decoder, transcoder, discriminator = build_model(encoder_dataset, decoder_dataset, **params)
    encoder.summary()
    decoder.summary()

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])
    if os.path.exists(params['discriminator_weights']):
        discriminator.load_weights(params['discriminator_weights'])

    if params['mode'] == 'train':
        print("Training...")
        for epoch in range(params['epochs']):
            train(encoder, decoder, transcoder, discriminator, encoder_dataset, decoder_dataset, **params)
            demonstrate(encoder, decoder, encoder_dataset, decoder_dataset, **params)
            encoder.save_weights(params['encoder_weights'])
            decoder.save_weights(params['decoder_weights'])
            discriminator.save_weights(params['discriminator_weights'])
    elif params['mode'] == 'demo':
        print("Starting Demonstration...")
        params['batch_size'] = 1
        while True:
            inp = raw_input("Type a complete sentence in the input language: ")
            inp = inp.decode('utf-8').lower()
            demonstrate(encoder, decoder, encoder_dataset, decoder_dataset, input_text=inp, **params)
