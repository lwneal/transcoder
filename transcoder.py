import os
import sys
import random
import numpy as np
from keras.utils import to_categorical
from keras import layers, models

from dataset import Dataset, left_pad, right_pad


def get_batch(encoder_dataset, decoder_dataset, **params):
    X = encoder_dataset.batch(**params)
    Y = decoder_dataset.batch(**params)
    for i in range(len(X)):
        idx = encoder_dataset.random_idx()
        X[i] = encoder_dataset.get_example(idx, **params)
        Y[i] = decoder_dataset.get_example(idx, **params)
    return X, Y


def generate(encoder_dataset, decoder_dataset, **params):
    while True:
        yield get_batch(encoder_dataset, decoder_dataset, **params)


def train(model, encoder_dataset, decoder_dataset, **params):
    batches_per_epoch = params['batches_per_epoch']
    training_gen = generate(encoder_dataset, decoder_dataset, **params)
    model.fit_generator(training_gen, steps_per_epoch=batches_per_epoch)


def demonstrate(model, encoder_dataset, decoder_dataset, input_text=None, **params):
    max_words = params['max_words']
    X = encoder_dataset.get_empty_batch(**params)
    for i in range(params['batch_size']):
        words = input_text or random.choice(encoder_dataset.sentences)
        X[i] = left_pad(encoder_dataset.indices(words)[:max_words], **params)
    batch_size, max_words = X.shape

    preds = model.predict(X)
    Y = np.argmax(preds, axis=-1)
    for i in range(len(Y)):
        left = ' '.join(encoder_dataset.words(X[i]))
        right = ' '.join(decoder_dataset.words(Y[i]))
        print('{} --> {}'.format(left, right))


# Actually boltzmann(log(x)) for stability
def boltzmann(pdf, temperature=1.0, epsilon=1e-5):
    if temperature < epsilon:
        return pdf / (pdf.sum() + epsilon)
    pdf = np.log(pdf) / temperature
    x = np.exp(pdf)
    sums = np.sum(x, axis=-1)[:, np.newaxis] + epsilon
    return x / sums


def sample(pdfs):
    max_words, vocab_size = pdfs.shape
    samples = np.zeros(max_words)
    for i in range(len(samples)):
        samples[i] = np.random.choice(np.arange(vocab_size), p=pdfs[i])
    return samples


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
    encoder_dataset = Dataset(params['encoder_input_filename'], encoder=True, **params)
    decoder_dataset = Dataset(params['decoder_input_filename'], **params)
    print("Dataset loaded")

    print("Building model")
    encoder, decoder, combined = build_model(encoder_dataset, decoder_dataset, **params)
    print("Model built")

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])

    combined.compile(loss='categorical_crossentropy', optimizer='adam')

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
