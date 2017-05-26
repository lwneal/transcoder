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


def get_batch(encoder_dataset, decoder_dataset, base_idx=None, **params):
    X_list = encoder_dataset.empty_batch(**params)
    Y_list = decoder_dataset.empty_batch(**params)
    batch_size = params['batch_size']
    for i in range(batch_size):
        if base_idx is None:
            idx = encoder_dataset.random_idx()
        else:
            idx = base_idx + i
        x_list = encoder_dataset.get_example(idx, **params)
        for X, x in zip(X_list, x_list):
            X[i] = x
        y_list = decoder_dataset.get_example(idx, **params)
        for Y, y in zip(Y_list, y_list):
            Y[i] = y
    return X_list, Y_list


def evaluate(transcoder, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']
    input_count, output_count = encoder_dataset.count(), decoder_dataset.count()
    assert input_count == output_count

    def eval_generator():
        X_list = encoder_dataset.empty_batch(**params)
        for i in range(0, input_count, batch_size):
            sys.stderr.write("\r[K{} / {}".format(i, input_count))
            yield get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)

    batch_count = input_count / batch_size
    scores = transcoder.evaluate_generator(eval_generator(), steps=batch_count)
    print("Scores: {}".format(scores))


def train_generator(encoder_dataset, decoder_dataset, **params):
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
    enable_gan = params['enable_gan']

    training_gen = train_generator(encoder_dataset, decoder_dataset, **params)
    clipping_time = 0
    training_start_time = time.time()
    t_avg_loss = 0
    t_avg_accuracy = 0
    r_avg_loss = 0
    d_avg_loss = 0
    g_avg_loss = 0
    batches_per_iter = int(params['training_iters_per_gan'])

    print("Training...")
    for i in range(0, batches_per_epoch, batches_per_iter):
        sys.stderr.write("[K\r{}/{} batches \tbs {}, D_g {:.3f}, D_r {:.3f} G {:.3f} T {:.3f} Accuracy {:.3f}".format(
            i + 1, batches_per_epoch, batch_size, d_avg_loss, r_avg_loss, g_avg_loss, t_avg_loss, t_avg_accuracy))

        # Update transcoder a bunch of times
        for _ in range(batches_per_iter):
            X, Y = next(training_gen)
            loss, accuracy = transcoder.train_on_batch(X, Y)
            t_avg_loss = .95 * t_avg_loss + .05 * loss
            t_avg_accuracy = .95 * t_avg_accuracy + .05 * accuracy

        if not enable_gan:
            continue

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

        # Update generator based on a random thought vector
        X_encoder = np.random.uniform(-1, 1, size=(batch_size, thought_vector_size))
        for layer in discriminator.layers:
            layer.trainable = False
        loss, accuracy = cgan.train_on_batch(X_encoder, -Y_disc)
        g_avg_loss = .95 * g_avg_loss + .05 * loss
        for layer in discriminator.layers:
            layer.trainable = True

    print("Trained for {:.2f} s (spent {:.2f} s clipping)".format(time.time() - training_start_time, clipping_time))
    sys.stderr.write('\n')


def demonstrate(transcoder, encoder_dataset, decoder_dataset, input_text=None, **params):
    batch_size = params['batch_size']

    X_list = encoder_dataset.empty_batch(**params)
    for i in range(batch_size):
        if input_text:
            x_list = encoder_dataset.format_input(input_text, **params)
        else:
            x_list = encoder_dataset.get_example(**params)
        for X, x in zip(X_list, x_list):
            X[i] = x
    Y = transcoder.predict(X_list)
    X = zip(*X_list)
    for x, y in zip(X, Y):
        left = encoder_dataset.unformat_input(x)
        right = decoder_dataset.unformat_output(y)
        print('{} --> {}'.format(left, right))


def hallucinate(decoder, decoder_dataset, **params):
    batch_size = params['batch_size']
    thought_vector_size = params['thought_vector_size']

    X_decoder = np.random.normal(0, 1, size=(batch_size, thought_vector_size))
    X_generated = decoder.predict(X_decoder)
    print("Hallucinated outputs:")
    for j in range(len(X_generated)):
        print(' ' + decoder_dataset.unformat_output(X_generated[j]))


def find_dataset(input_filename, dataset_type=None, **params):
    types = {
        'img': ImageDataset,
        'bbox': ImageRegionDataset,
        'vq': VisualQuestionDataset,
        'txt': WordDataset,
    }
    if dataset_type:
        return types[dataset_type]
    # If no dataset type is specified, infer based on file extension
    ext = input_filename.split('.')[-1]
    return types.get(ext, WordDataset)


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

    cgan = models.Model(inputs=decoder.inputs, outputs=discriminator(decoder.output))
    cgan.compile(loss=wgan_loss, optimizer='adam', metrics=['accuracy'])
    cgan._make_train_function()

    transcoder = models.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    transcoder.compile(loss=transcoder_loss, optimizer='adam', metrics=['accuracy'])
    transcoder._make_train_function()

    print("\nEncoder")
    encoder.summary()
    print("\nDecoder")
    decoder.summary()
    print("\nDiscriminator")
    discriminator.summary()
    return encoder, decoder, transcoder, discriminator, cgan


def main(**params):
    mode = params['mode']
    epochs = params['epochs']
    encoder_input_filename = params['encoder_input_filename']
    encoder_datatype = params['encoder_datatype']
    decoder_input_filename = params['decoder_input_filename']
    decoder_datatype = params['decoder_datatype']
    encoder_weights = params['encoder_weights']
    decoder_weights = params['decoder_weights']
    discriminator_weights = params['discriminator_weights']

    print("Loading datasets...")
    encoder_dataset = find_dataset(encoder_input_filename, encoder_datatype)(encoder_input_filename, is_encoder=True, **params)
    decoder_dataset = find_dataset(decoder_input_filename, decoder_datatype)(decoder_input_filename, is_encoder=False, **params)

    print("Building models...")
    encoder, decoder, transcoder, discriminator, cgan = build_model(encoder_dataset, decoder_dataset, **params)

    print("Loading weights...")
    if os.path.exists(encoder_weights):
        encoder.load_weights(encoder_weights)
    if os.path.exists(decoder_weights):
        decoder.load_weights(decoder_weights)
    if os.path.exists(discriminator_weights):
        discriminator.load_weights(discriminator_weights)

    print("Starting mode {}".format(mode))
    if mode == 'train':
        for epoch in range(epochs):
            train(encoder, decoder, transcoder, discriminator, cgan, encoder_dataset, decoder_dataset, **params)
            encoder.save_weights(encoder_weights)
            decoder.save_weights(decoder_weights)
            discriminator.save_weights(discriminator_weights)
            demonstrate(transcoder, encoder_dataset, decoder_dataset, **params)
            hallucinate(decoder, decoder_dataset, **params)
    elif mode == 'test':
        evaluate(transcoder, encoder_dataset, decoder_dataset, **params)
    elif mode == 'demo':
        demonstrate(transcoder, encoder_dataset, decoder_dataset, input_text=inp, **params)
