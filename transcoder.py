import os
import sys
import random
import numpy as np
import tensorflow as tf
import time
from keras import layers, models, optimizers

import model_definitions
import model_words
import model_img
import model_img_region
from dataset_word import WordDataset
from dataset_label import LabelDataset
from dataset_img import ImageDataset
from dataset_img_region import ImageRegionDataset
from dataset_visual_question import VisualQuestionDataset
import imutil

import train
import counterfactual


def evaluate(transcoder, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']
    input_count, output_count = encoder_dataset.count(), decoder_dataset.count()
    assert input_count == output_count

    def eval_generator():
        X_list = encoder_dataset.empty_batch(**params)
        for i in range(0, input_count, batch_size):
            sys.stderr.write("\r[K{} / {}".format(i, input_count))
            yield train.get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)
        sys.stderr.write("\n")
        # HACK: Keras is dumb and asynchronously queues up batches beyond the last one
        # Could be solved if evaluate_generator() workers used an atomic counter
        while True:
            yield train.get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)

    batch_count = input_count / batch_size
    scores = transcoder.evaluate_generator(eval_generator(), steps=batch_count)
    print("")
    print("Completed evaluation on {} items".format(batch_count))
    print("input: {}".format(params['encoder_input_filename']))
    print("output: {}".format(params['decoder_input_filename']))
    print("encoder: {}".format(params['encoder_weights']))
    print("decoder: {}".format(params['decoder_weights']))
    for name, val in zip(['loss'] + transcoder.metrics, scores):
        print("{}: {:.5f}".format(name, val))


def run_train(*args, **kwargs):
    train.train(*args, **kwargs)


def demonstrate(transcoder, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']

    X_list = encoder_dataset.empty_batch(**params)
    Y_list = decoder_dataset.empty_batch(**params)
    for i in range(batch_size):
        idx = np.random.randint(encoder_dataset.count())
        x_list = encoder_dataset.get_example(idx, **params)
        y_list = decoder_dataset.get_example(idx, **params)
        for X, x in zip(X_list, x_list):
            X[i] = x
        for Y, y in zip(Y_list, y_list):
            Y[i] = y
    X = zip(*X_list)
    Y_gen = transcoder.predict(X_list)
    Y_true = Y_list[0]
    for x, y_gen, y_true in zip(X, Y_gen, Y_true):
        x_input = encoder_dataset.unformat_input(x)
        y_generated = decoder_dataset.unformat_output(y_gen)
        y_truth = decoder_dataset.unformat_output(y_true)
        print('{} --> {} ({})'.format(x_input, y_generated, y_truth))
    fig_filename = '{}_demo.jpg'.format(int(time.time()))
    imutil.show_figure(filename=fig_filename, resize_to=None)


def hallucinate(decoder, decoder_dataset, dist='gaussian', **params):
    batch_size = params['batch_size']
    thought_vector_size = params['thought_vector_size']

    X_decoder = np.random.normal(0, 1, size=(batch_size, thought_vector_size))
    X_generated = decoder.predict(X_decoder)
    print("Hallucinated outputs:")
    for j in range(len(X_generated)):
        print(' ' + decoder_dataset.unformat_output(X_generated[j]))
    fig_filename = '{}_halluc_{}.jpg'.format(int(time.time()), dist)
    imutil.show_figure(filename=fig_filename, resize_to=None)


def dream(encoder, decoder, encoder_dataset, decoder_dataset, **params):
    batch_size = params['batch_size']
    thought_vector_size = params['thought_vector_size']
    video_filename = params['video_filename']
    dream_frames_per_example = params['dream_fps']
    dream_examples = params['dream_examples']
    if params['batch_size'] < 2:
        raise ValueError("--batch-size of {} is too low, dream() requires a larger batch size".format(params['batch_size']))

    # Select two inputs in the dataset
    start_idx = np.random.randint(encoder_dataset.count())
    end_idx = np.random.randint(encoder_dataset.count())
    for _ in range(dream_examples):
        input_start = encoder_dataset.get_example(start_idx, **params)
        input_end = encoder_dataset.get_example(end_idx, **params)
        #encoder_dataset.unformat_input(input_start)
        #encoder_dataset.unformat_input(input_end)

        # Use the encoder to get the latent vector for each example
        X_list = encoder_dataset.empty_batch(**params)
        for X, x in zip(X_list, input_start):
            X[0] = x
        for X, x in zip(X_list, input_end):
            X[1] = x

        latent = encoder.predict(X_list)
        latent_start, latent_end = latent[0], latent[1]

        # Interpolate between the two latent vectors, and output
        # the result of the decoder at each step
        # TODO: Something other than linear interpolation?
        for i in range(dream_frames_per_example):
            print("Writing img {} frame {}".format(start_idx, i))
            c = float(i) / dream_frames_per_example
            v = c * latent_end + (1 - c) * latent_start
            img = decoder.predict(np.expand_dims(v, axis=0))[0]
            #decoder_dataset.unformat_output(img)
            caption = '{} {}'.format(start_idx, i)
            imutil.show(img, video_filename=video_filename, resize_to=(512,512), display=(i % 100 == 0), caption=caption)
        print("Done")
        start_idx = end_idx
        end_idx = np.random.randint(encoder_dataset.count())


def run_counterfactual(encoder, decoder, classifier, encoder_dataset, decoder_dataset, classifier_dataset, **params):
    video_filename = params['video_filename']

    training_gen = train.train_generator(encoder_dataset, decoder_dataset, **params)
    X, _ = next(training_gen)
    X = X[:1]
    imutil.show(X)
    Z = encoder.predict(X)

    trajectory_path = []

    for _ in range(3):
        trajectory = counterfactual.compute_trajectory(
                encoder, decoder, classifier,
                Z, classifier_dataset,
                **params)
        # There and back again
        trajectory_path.extend(trajectory)
        trajectory_path.extend([trajectory[-1]] * 12)
        trajectory_path.extend(reversed(trajectory))
        trajectory_path.extend([trajectory[0]] * 12)

    def output_frame(z, display=False):
        classification = classifier.predict(z)[0]
        caption = '{:.02f} {}'.format(
                classification.max(),
                classifier_dataset.unformat_output(classification))
        imutil.show(decoder.predict(z), resize_to=(512, 512), video_filename=video_filename,
                caption=caption, font_size=20, display=display)
        print("Classification: {}".format(classifier_dataset.unformat_output(classification)))

    # First show the original image for reference
    for _ in range(24):
        imutil.show(X, resize_to=(512, 512), video_filename=video_filename, display=False)

    # Then the GAN trajectory
    for z in trajectory_path:
        output_frame(z, display=False)


def find_dataset(input_filename, dataset_type=None, **params):
    types = {
        'img': ImageDataset,
        'bbox': ImageRegionDataset,
        'vq': VisualQuestionDataset,
        'txt': WordDataset,
        'lab': LabelDataset,
    }
    if dataset_type:
        return types[dataset_type]
    # If no dataset type is specified, infer based on file extension
    ext = input_filename.split('.')[-1]
    return types.get(ext, WordDataset)


def build_model(encoder_dataset, decoder_dataset, classifier_dataset, **params):
    encoder_model = params['encoder_model']
    decoder_model = params['decoder_model']
    discriminator_model = params['discriminator_model']
    classifier_model = params['classifier_model']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']
    enable_perceptual_loss = params['enable_perceptual_loss']
    alpha = params['perceptual_loss_alpha']
    perceptual_layers = params['perceptual_loss_layers']
    decay = params['decay']

    metrics = ['accuracy']
    optimizer = optimizers.Adam(decay=decay)
    classifier_loss = 'categorical_crossentropy'

    # HACK: Keras Bug https://github.com/fchollet/keras/issues/5221
    # Sharing a BatchNormalization layer corrupts the graph
    # Workaround: carefully call _make_train_function

    from keras import backend as K
    def wgan_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    build_encoder = getattr(model_definitions, encoder_model)
    encoder = build_encoder(dataset=encoder_dataset, **params)

    build_decoder = getattr(model_definitions, decoder_model)
    decoder = build_decoder(dataset=decoder_dataset, **params)

    if enable_discriminator:
        build_discriminator = getattr(model_definitions, discriminator_model)
        discriminator = build_discriminator(is_discriminator=True, dataset=decoder_dataset, **params)
        discriminator.compile(loss=wgan_loss, optimizer=optimizer, metrics=metrics)
        discriminator._make_train_function()

        cgan = models.Model(inputs=decoder.inputs, outputs=discriminator(decoder.output))
        cgan.compile(loss=wgan_loss, optimizer=optimizer, metrics=metrics)
        cgan._make_train_function()
    else:
        discriminator = models.Sequential()
        cgan = models.Sequential()

    if enable_classifier:
        build_classifier = getattr(model_definitions, classifier_model)
        classifier = build_classifier(dataset=classifier_dataset, **params)

        transclassifier = models.Model(inputs=encoder.inputs, outputs=classifier(encoder.output))
        transclassifier.compile(loss=classifier_loss, optimizer=optimizer, metrics=metrics)
    else:
        classifier = models.Sequential()
        transclassifier = models.Sequential()

    from keras import losses
    if type(decoder_dataset) is ImageDataset:
        if enable_perceptual_loss:
            from keras import applications
            P = applications.mobilenet.MobileNet(include_top=False)
            perceptual_outputs = []
            for layer in P.layers:
                if layer.name.startswith('conv') and len(perceptual_outputs) < perceptual_layers:
                    perceptual_outputs.append(layer.output)
            print("Perceptual Loss: Using {} convolutional layers".format(len(perceptual_outputs)))

            texture = models.Model(inputs=P.inputs, outputs=perceptual_outputs)

            # A scalar value that measures the perceptual difference between two images wrt. a pretrained convnet
            def perceptual_loss(y_true, y_pred):
                T_a, T_b = texture(y_true), texture(y_pred)
                p_loss = K.mean(K.abs(T_a[0] - T_b[0]))
                for a, b in zip(T_a, T_b)[1:]:
                    p_loss += K.mean(K.abs(a - b))
                return p_loss

            transcoder_loss = lambda x, y: alpha * losses.mean_absolute_error(x, y) + (1 - alpha) * perceptual_loss(x, y)
        else:
            transcoder_loss = losses.mean_absolute_error
    else:
        transcoder_loss = losses.categorical_crossentropy

    transcoder = models.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    transcoder.compile(loss=transcoder_loss, optimizer=optimizer, metrics=metrics)
    transcoder._make_train_function()

    print("\nEncoder")
    encoder.summary()
    print("\nDecoder")
    decoder.summary()
    print("\nDiscriminator")
    discriminator.summary()
    print('\nClassifier:')
    classifier.summary()
    return encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier


def main(**params):
    mode = params['mode']
    epochs = params['epochs']
    encoder_input_filename = params['encoder_input_filename']
    encoder_datatype = params['encoder_datatype']
    decoder_input_filename = params['decoder_input_filename']
    decoder_datatype = params['decoder_datatype']
    classifier_input_filename = params['classifier_input_filename']
    classifier_datatype = params['classifier_datatype']
    encoder_weights = params['encoder_weights']
    decoder_weights = params['decoder_weights']
    discriminator_weights = params['discriminator_weights']
    classifier_weights = params['classifier_weights']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']

    print("Loading datasets...")
    encoder_dataset = find_dataset(encoder_input_filename, encoder_datatype)(encoder_input_filename, is_encoder=True, **params)
    decoder_dataset = find_dataset(decoder_input_filename, decoder_datatype)(decoder_input_filename, is_encoder=False, **params)
    classifier_dataset = None
    if enable_classifier:
        classifier_dataset = find_dataset(classifier_input_filename, classifier_datatype)(decoder_input_filename, is_encoder=False, **params)

    print("Building models...")
    encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier = build_model(encoder_dataset, decoder_dataset, classifier_dataset, **params)

    print("Loading weights...")
    if os.path.exists(encoder_weights):
        encoder.load_weights(encoder_weights)
    if os.path.exists(decoder_weights):
        decoder.load_weights(decoder_weights)
    if enable_discriminator and os.path.exists(discriminator_weights):
        discriminator.load_weights(discriminator_weights)
    if enable_classifier and os.path.exists(classifier_weights):
        classifier.load_weights(classifier_weights)

    print("Starting mode {}".format(mode))
    if mode == 'train':
        for epoch in range(epochs):
            run_train(encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier, encoder_dataset, decoder_dataset, classifier_dataset, **params)
            encoder.save_weights(encoder_weights)
            decoder.save_weights(decoder_weights)
            if enable_discriminator:
                discriminator.save_weights(discriminator_weights)
            if enable_classifier:
                classifier.save_weights(classifier_weights)
            demonstrate(transcoder, encoder_dataset, decoder_dataset, **params)
            if enable_discriminator:
                hallucinate(decoder, decoder_dataset, **params)
    elif mode == 'evaluate':
        evaluate(transcoder, encoder_dataset, decoder_dataset, **params)
    elif mode == 'demo':
        demonstrate(transcoder, encoder_dataset, decoder_dataset, **params)
        if enable_discriminator:
            hallucinate(decoder, decoder_dataset, **params)
    elif mode == 'dream':
        dream(encoder, decoder, encoder_dataset, decoder_dataset, **params)
    elif mode == 'counterfactual':
        run_counterfactual(encoder, decoder, classifier, encoder_dataset, decoder_dataset, classifier_dataset, **params)
