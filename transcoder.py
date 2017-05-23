import os
import sys
import random
import numpy as np
import tensorflow as tf
import time
from keras import layers, models

import model_words
import model_img
import model_img_region
from dataset_word import WordDataset
from dataset_img import ImageDataset
from dataset_img_region import ImageRegionDataset
from dataset_visual_question import VisualQuestionDataset


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
    # TODO: freeze_encoder and freeze_decoder
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    thought_vector_size = params['thought_vector_size']

    training_gen = generate(encoder_dataset, decoder_dataset, **params)
    clipping_time = 0
    training_start_time = time.time()
    t_avg_loss = 0
    t_avg_accuracy = 0
    r_avg_loss = 0
    d_avg_loss = 0
    g_avg_loss = 0

    print("Training...")
    for i in range(batches_per_epoch):
        # Update transcoder a bunch of times
        for _ in range(int(params['training_iters_per_gan'])):
            X, Y = next(training_gen)
            loss, accuracy = transcoder.train_on_batch(X, Y)
            t_avg_loss = .95 * t_avg_loss + .05 * loss
            t_avg_accuracy = .95 * t_avg_accuracy + .05 * accuracy

        # Update Discriminator 5x per Generator update
        for _ in range(5):
            # Get some real decoding targets
            _, Y_decoder = next(training_gen)

            # Think some random thoughts
            X_decoder = np.random.normal(0, 1, size=(batch_size, thought_vector_size))

            # Decode those random thoughts into hallucinations
            X_generated = decoder.predict(X_decoder)

            # Start with a batch of real ground truth targets
            X_real = Y_decoder
            Y_disc = np.ones(batch_size)

            for layer in decoder.layers:
                layer.trainable = False
            loss, accuracy = discriminator.train_on_batch(X_real, -Y_disc)
            r_avg_loss = .95 * r_avg_loss + .05 * loss

            loss, accuracy = discriminator.train_on_batch(X_generated, Y_disc)
            d_avg_loss = .95 * d_avg_loss + .05 * loss
            for layer in decoder.layers:
                layer.trainable = True

            # Clip discriminator weights
            start_time = time.time()
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(w, -.01, .01) for w in weights]
                layer.set_weights(weights)
            clipping_time += time.time() - start_time

        # Generate a random thought vector
        X_encoder = np.random.uniform(-1, 1, size=(batch_size, thought_vector_size))

        for layer in discriminator.layers:
            layer.trainable = False
        loss, accuracy = cgan.train_on_batch(X_encoder, -Y_disc)
        g_avg_loss = .95 * g_avg_loss + .05 * loss
        for layer in discriminator.layers:
            layer.trainable = True

        sys.stderr.write("[K\r{}/{} batches \tbs {}, D_g {:.3f}, D_r {:.3f} G {:.3f} T {:.3f} Accuracy {:.3f}".format(
            i + 1, batches_per_epoch, batch_size, d_avg_loss, r_avg_loss, g_avg_loss, t_avg_loss, t_avg_accuracy))

        if i == batches_per_epoch - 1:
            print("\nHallucinated outputs:")
            for j in range(len(X_generated)):
                print(' ' + decoder_dataset.unformat_output(X_generated[j]))

    print("Trained for {:.2f} s (spent {:.2f} s clipping)".format(time.time() - training_start_time, clipping_time))
    sys.stderr.write('\n')


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
    X = zip(*X_list)
    for x, y in zip(X, Y):
        left = encoder_dataset.unformat_input(x)
        right = decoder_dataset.unformat_output(y)
        print('{} --> {}'.format(left, right))


def dataset_for_extension(ext):
    if ext == 'img':
        return ImageDataset
    elif ext == 'bbox':
        return ImageRegionDataset
    elif ext == 'vq':
        return VisualQuestionDataset
    return WordDataset


def build_model(encoder_dataset, decoder_dataset, **params):
    # HACK: Keras Bug https://github.com/fchollet/keras/issues/5221
    # Sharing a BatchNormalization layer corrupts the graph
    # Workaround: carefully call _make_train_function

    from keras import backend as K
    def wgan_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)
    transcoder_loss = 'mse' if type(decoder_dataset) is ImageDataset else 'categorical_crossentropy'


    encoder = encoder_dataset.build_encoder(**params)

    decoder = decoder_dataset.build_decoder(**params)

    discriminator = decoder_dataset.build_discriminator(**params)
    discriminator.compile(loss=wgan_loss, optimizer='adam', metrics=['accuracy'])
    discriminator._make_train_function()

    tensor_in = decoder.inputs[0]
    tensor_out = discriminator(decoder.outputs[0])
    cgan = models.Model(inputs=tensor_in, outputs=tensor_out)
    cgan.compile(loss=wgan_loss, optimizer='adam', metrics=['accuracy'])
    cgan._make_train_function()

    tensor_in = encoder.inputs[0]
    tensor_out = decoder(encoder.outputs[0])
    transcoder = models.Model(inputs=tensor_in, outputs=tensor_out)
    transcoder.compile(loss=transcoder_loss, optimizer='adam', metrics=['accuracy'])
    transcoder._make_train_function()

    return encoder, decoder, transcoder, discriminator, cgan


def main(**params):
    print("Loading datasets...")
    ds = dataset_for_extension(params['encoder_input_filename'].split('.')[-1])
    encoder_dataset = ds(params['encoder_input_filename'], is_encoder=True, **params)

    ds = dataset_for_extension(params['decoder_input_filename'].split('.')[-1])
    decoder_dataset = ds(params['decoder_input_filename'], is_encoder=False, **params)

    print("Building models...")
    encoder, decoder, transcoder, discriminator, cgan = build_model(encoder_dataset, decoder_dataset, **params)
    print("\nEncoder")
    encoder.summary()
    print("\nDecoder")
    decoder.summary()
    print("\nDiscriminator")
    discriminator.summary()

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])
    if os.path.exists(params['discriminator_weights']):
        discriminator.load_weights(params['discriminator_weights'])

    if params['mode'] == 'train':
        print("Training...")
        for epoch in range(params['epochs']):
            demonstrate(encoder, decoder, encoder_dataset, decoder_dataset, **params)
            train(encoder, decoder, transcoder, discriminator, cgan, encoder_dataset, decoder_dataset, **params)
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
