import os
import sys
import random
import numpy as np
from keras import layers, models

from dataset import WordDataset, left_pad


def get_batch(encoder_dataset, decoder_dataset, **params):
    X = encoder_dataset.empty_batch(**params)
    Y = decoder_dataset.empty_batch(**params)
    for i in range(len(X)):
        idx = encoder_dataset.random_idx()
        X[i] = encoder_dataset.get_example(idx, **params)
        Y[i] = decoder_dataset.get_example(idx, **params)
    # TODO: Can X and Y be the same shape?
    Y = np.expand_dims(Y, axis=-1)
    return X, Y


def generate(encoder_dataset, decoder_dataset, **params):
    while True:
        yield get_batch(encoder_dataset, decoder_dataset, **params)


def train(model, encoder_dataset, decoder_dataset, **params):
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)
    X, Y = next(training_gen)
    model.fit_generator(training_gen, steps_per_epoch=batches_per_epoch)


def demonstrate(model, encoder_dataset, decoder_dataset, input_text=None, **params):
    X = encoder_dataset.empty_batch(**params)
    for i in range(len(X)):
        if input_text:
            X[i] = encoder_dataset.format_input(input_text, **params)
        else:
            X[i] = encoder_dataset.get_example(**params)

    Y = model.predict(X)
    for x, y in zip(X, Y):
        left = encoder_dataset.unformat_input(x)
        right = decoder_dataset.unformat_output(y)
        print('{} --> {}'.format(left, right))


def build_model(encoder_dataset, decoder_dataset, **params):
    encoder = build_encoder(encoder_dataset, **params)
    decoder = build_decoder(decoder_dataset, **params)

    combined = models.Sequential()
    combined.add(encoder)
    combined.add(decoder)
    return encoder, decoder, combined


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
    word_preds = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))(x)
    return models.Model(inputs=inp, outputs=word_preds)


def main(**params):
    print("Loading dataset")
    # TODO: Separate datasets
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
