import os
import sys
import random
import numpy as np
import tensorflow as tf
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


def train(encoder, decoder, transcoder, discriminator, cgan, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)

    train_transcoder(transcoder, training_gen, **params)
    train_gan(decoder, discriminator, cgan, training_gen, decoder_dataset, **params)
    print("Training epoch finished")


def train_transcoder(transcoder, training_gen, **params):
    batches_per_epoch = params['batches_per_epoch']
    batch_size = params['batch_size']
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


def train_gan(decoder, discriminator, cgan, training_gen, decoder_dataset, **params):
    batches_per_epoch = int(params['batches_per_epoch'] / params['training_iters_per_gan'])
    batch_size = params['batch_size']
    thought_vector_size = params['thought_vector_size']

    avg_loss = 0
    avg_accuracy = 0
    print("Training discriminator...")
    for i in range(batches_per_epoch):
        # Get some real decoding targets
        _, Y_decoder = next(training_gen)

        # Think some random thoughts
        X_decoder = np.random.normal(size=(batch_size, thought_vector_size))

        # Decode those random thoughts into hallucinations
        X_generated = decoder.predict(X_decoder)

        # Start with a batch of real ground truth targets
        X_real = Y_decoder
        Y_disc = np.ones(batch_size)

        for layer in decoder.layers:
            layer.trainable = False
        # Train discriminator layers to detect hallucinations
        loss, accuracy = discriminator.train_on_batch(X_real, -Y_disc)
        avg_loss = .95 * avg_loss + .05 * loss
        avg_accuracy = .95 * avg_accuracy + .05 * accuracy

        loss, accuracy = discriminator.train_on_batch(X_generated, Y_disc)
        avg_loss = .95 * avg_loss + .05 * loss
        avg_accuracy = .95 * avg_accuracy + .05 * accuracy
        sys.stderr.write("[K\r{}/{} batches, batch size {}, loss {:.3f}, accuracy {:.3f}".format(
            i, batches_per_epoch, batch_size, avg_loss, avg_accuracy))
        for layer in decoder.layers:
            layer.trainable = True

        # Clip discriminator weights
        for layer in discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -.01, .01) for w in weights]
            layer.set_weights(weights)

        # Generate a random thought vector
        X_encoder = np.random.uniform(-1, 1, size=(batch_size, thought_vector_size))

        # Train decoder to generate hallucinations that the discriminator thinks are real
        # HACK: how does layer.trainable work?
        for layer in discriminator.layers:
            layer.trainable = False
        # TODO: why do all weights in discriminator turn to NaN at this line?
        loss, accuracy = cgan.train_on_batch(X_encoder, -Y_disc)
        for layer in discriminator.layers:
            layer.trainable = True

        avg_loss = .95 * avg_loss + .05 * loss
        avg_accuracy = .95 * avg_accuracy + .05 * accuracy
        sys.stderr.write("[K\r{}/{} batches, batch size {}, loss {:.3f}, accuracy {:.3f}".format(
            i, batches_per_epoch, batch_size, avg_loss, avg_accuracy))


    sys.stderr.write('\n')

    # Print one of those hallucinations
    print("Hallucinated outputs:")
    for i in range(len(X_generated)):
        print(' ' + decoder_dataset.unformat_output(X_generated[i]))


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

    # TODO: fix for GAN
    if params['freeze_encoder']:
        for layer in encoder.layers:
            layer.trainable = False
    if params['freeze_decoder']:
        for layer in decoder.layers:
            layer.trainable = False

    transcoder = models.Sequential(name='transcoder')
    transcoder.add(encoder)
    transcoder.add(decoder)
    transcoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    discriminator = models.Sequential(name='discriminator')
    vocab_len = len(decoder_dataset.vocab)
    wordvec_size = 512
    max_words = params['max_words']
    input_shape = (max_words, vocab_len)
    discriminator.add(layers.Dense(wordvec_size, input_shape=input_shape))
    discriminator.add(layers.LSTM(512, return_sequences=True))
    discriminator.add(layers.LSTM(512))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.Activation('tanh'))
    discriminator.add(layers.Dense(512))
    discriminator.add(layers.BatchNormalization())
    discriminator.add(layers.Activation('tanh'))
    discriminator.add(layers.Dense(1))
    #from keras.layers.advanced_activations import LeakyReLU
    discriminator.add(layers.Activation('sigmoid'))

    from keras import backend as K
    def wgan_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)
    discriminator.compile(loss=wgan_loss, optimizer='adam', metrics=['accuracy'])

    cgan = models.Sequential()
    for layer in decoder.layers:
        cgan.add(layer)
    for layer in discriminator.layers:
        cgan.add(layer)
    cgan.compile(loss=wgan_loss, optimizer='adam', metrics=['accuracy'])

    return encoder, decoder, transcoder, discriminator, cgan


def main(**params):
    print("Loading datasets...")
    ds = dataset_for_extension(params['encoder_input_filename'].split('.')[-1])
    encoder_dataset = ds(params['encoder_input_filename'], encoder=True, **params)

    ds = dataset_for_extension(params['decoder_input_filename'].split('.')[-1])
    decoder_dataset = ds(params['decoder_input_filename'], encoder=False, **params)

    print("Building models...")
    encoder, decoder, transcoder, discriminator, cgan = build_model(encoder_dataset, decoder_dataset, **params)
    encoder.summary()
    decoder.summary()
    discriminator.summary()
    cgan.summary()

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])
    if os.path.exists(params['discriminator_weights']):
        discriminator.load_weights(params['discriminator_weights'])

    if params['mode'] == 'train':
        print("Training...")
        for epoch in range(params['epochs']):
            train(encoder, decoder, transcoder, discriminator, cgan, encoder_dataset, decoder_dataset, **params)
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
