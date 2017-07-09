import os
import sys
import random
import numpy as np
import tensorflow as tf
import time
from keras import layers, models

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


def get_batch(encoder_dataset, decoder_dataset, base_idx=None, **params):
    batch_size = params['batch_size']
    # The last batch might need to be a smaller partial batch
    example_count = min(encoder_dataset.count(), decoder_dataset.count())
    if base_idx is not None and base_idx + batch_size > example_count:
        batch_size = example_count - base_idx
        params['batch_size'] = batch_size
        #print("\nFinal batch at base idx {} with odd size: {}".format(base_idx, batch_size))
    X_list = encoder_dataset.empty_batch(**params)
    Y_list = decoder_dataset.empty_batch(**params)
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
        sys.stderr.write("\n")
        # HACK: Keras is dumb and asynchronously queues up batches beyond the last one
        # Could be solved if evaluate_generator() workers used an atomic counter
        while True:
            yield get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)


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


def train_generator(encoder_dataset, decoder_dataset, **params):
    # Datasets return a list of arrays
    # Keras requires either a list of arrays or a single array
    def unpack(Z):
        if isinstance(Z, list) and len(Z) == 1:
            return Z[0]
        return Z
    while True:
        X, Y = get_batch(encoder_dataset, decoder_dataset, **params)
        yield unpack(X), unpack(Y)


def train(encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier, encoder_dataset, decoder_dataset, classifier_dataset, **params):
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    thought_vector_size = params['thought_vector_size']
    enable_gan = params['enable_gan']
    enable_classifier = params['enable_classifier']
    batches_per_iter = int(params['training_iters_per_gan'])
    freeze_encoder = params['freeze_encoder']
    freeze_decoder = params['freeze_decoder']
    enable_transcoder = not (freeze_encoder and freeze_decoder)

    training_gen = train_generator(encoder_dataset, decoder_dataset, **params)
    if enable_classifier:
        classifier_gen = train_generator(encoder_dataset, classifier_dataset, **params)
    clipping_time = 0
    training_start_time = time.time()
    t_avg_loss = 0
    t_avg_accuracy = 0
    r_avg_loss = 0
    d_avg_loss = 0
    g_avg_loss = 0
    c_avg_loss = 0
    c_avg_accuracy = 0

    print("Training...")
    for i in range(0, batches_per_epoch, batches_per_iter):
        sys.stderr.write("\r[K\r{}/{} bs {}, DG_loss {:.3f}, DR_loss {:.3f} G_loss {:.3f} T_loss {:.3f} T_acc {:.3f} C_loss {:.3f} C_acc {:.3f}".format(
            i + 1, batches_per_epoch, batch_size,
            d_avg_loss,
            r_avg_loss,
            g_avg_loss,
            t_avg_loss,
            t_avg_accuracy,
            c_avg_loss,
            c_avg_accuracy
        ))

        # Train encoder and decoder on labeled X -> Y pairs
        if enable_transcoder:
            for _ in range(batches_per_iter):
                X, Y = next(training_gen)
                if freeze_encoder:
                    for layer in encoder.layers:
                        layer.trainable = False
                if freeze_decoder:
                    for layer in decoder.layers:
                        layer.trainable = False
                loss, accuracy = transcoder.train_on_batch(X, Y)
                for layer in encoder.layers + decoder.layers:
                    layer.trainable = True
                t_avg_loss = .95 * t_avg_loss + .05 * loss
                t_avg_accuracy = .95 * t_avg_accuracy + .05 * accuracy

        if enable_gan:
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

        if enable_classifier:
            X, Y = next(classifier_gen)
            if freeze_encoder:
                for layer in encoder.layers:
                    layer.trainable = False
            if freeze_decoder:
                for layer in decoder.layers:
                    layer.trainable = False
            loss, accuracy = transclassifier.train_on_batch(X, Y)
            for layer in encoder.layers + decoder.layers:
                layer.trainable = True
            c_avg_loss = .95 * c_avg_loss + .05 * loss
            c_avg_accuracy = .95 * c_avg_accuracy + .05 * accuracy

        if enable_gan:
            # Update generator based on a random thought vector
            X_encoder = np.random.uniform(-1, 1, size=(batch_size, thought_vector_size))
            for layer in discriminator.layers:
                layer.trainable = False
            loss, accuracy = cgan.train_on_batch(X_encoder, -Y_disc)
            g_avg_loss = .95 * g_avg_loss + .05 * loss
            for layer in discriminator.layers:
                layer.trainable = True

    sys.stderr.write('\n')
    print("Trained for {:.2f} s (spent {:.2f} s clipping)".format(time.time() - training_start_time, clipping_time))
    sys.stderr.write('\n')


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
    dream_fps = params['dream_fps']

    # Select two inputs in the dataset
    img_idx = 41
    for _ in range(10):
        input_start = encoder_dataset.get_example(img_idx, **params)
        input_end = encoder_dataset.get_example(img_idx + 1, **params)
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
        for i in range(dream_fps):
            print("Writing img {} frame {}".format(img_idx, i))
            c = float(i) / dream_fps
            v = c * latent_end + (1 - c) * latent_start
            img = decoder.predict(np.expand_dims(v, axis=0))[0]
            #decoder_dataset.unformat_output(img)
            imutil.show(img, video_filename=video_filename, resize_to=(512,512))
        print("Done")
        img_idx += 1


def counterfactual(encoder, decoder, classifier, encoder_dataset, decoder_dataset, classifier_dataset, **params):
    thought_vector_size = params['thought_vector_size']
    video_filename = params['video_filename']

    # Randomly choose a class to mutate toward
    selected_class = np.random.randint(0, len(classifier_dataset.idx_to_name))

    training_gen = train_generator(encoder_dataset, decoder_dataset, **params)
    X, _ = next(training_gen)
    X = X[:1]
    imutil.show(X)

    Z = encoder.predict(X)

    classification = classifier.predict(Z)[0]
    print("Classification: {}".format(classification))
    original_class = np.copy(classification)

    # This will contain our latent vector
    from keras import backend as K
    latent_value = K.placeholder((1, thought_vector_size))

    loss = K.variable(0.)
    loss += K.sum(classifier.outputs[0][0])

    loss -= K.sum(classifier.outputs[0][0][selected_class])

    grads = K.gradients(loss, classifier.inputs[0])

    compute_gradient = K.function(classifier.inputs, grads)

    # Perform gradient descent on the classification loss
    step_size = .01
    classification = classifier.predict(Z)[0]
    momentum = None
    NUM_FRAMES = 240
    def output_frame():
        caption = '{:.02f} {}'.format(
                classification.max(),
                classifier_dataset.unformat_output(classification))
        imutil.show(decoder.predict(Z), resize_to=(512, 512), video_filename=video_filename, caption=caption, font_size=20)
        print("Classification: {}".format(classifier_dataset.unformat_output(classification)))

    for _ in range(10):
        output_frame()
    for i in range(10 * NUM_FRAMES):
        gradient = compute_gradient([Z])[0]
        if momentum is None:
            momentum = gradient
        momentum += gradient
        momentum *= .99
        # Clip magnitude to 1.0
        if np.linalg.norm(momentum) > 1.0:
            momentum /= np.linalg.norm(momentum)
        Z -= momentum * step_size
        classification = classifier.predict(Z)[0]
        # L2 regularization?
        #Z *= 0.99
        if i % 10 == 0:
            output_frame()
    for _ in range(5):
        output_frame()
    print('\n')
    print("Original Image:")
    img = decoder.predict(encoder.predict(X))[0]
    imutil.show(img, filename='{}_counterfactual_orig.jpg'.format(int(time.time())))
    imutil.add_to_figure(img)
    print("Original Classification: {}".format(classifier_dataset.unformat_output(original_class)))

    print("Counterfactual Image:")
    img = decoder.predict(Z)[0]
    imutil.show(img, filename='{}_counterfactual_{}.jpg'.format(int(time.time()), selected_class))
    imutil.add_to_figure(img)
    print("Counterfactual Classification: {}".format(classifier_dataset.unformat_output(classification)))
    imutil.show_figure(filename='{}_counterfactual.jpg'.format(int(time.time())), resize_to=None)

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
    enable_gan = params['enable_gan']
    enable_classifier = params['enable_classifier']
    enable_perceptual_loss = params['enable_perceptual_loss']
    alpha = params['perceptual_loss_alpha']

    metrics = ['accuracy']
    optimizer = 'adam'
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

    if enable_gan:
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
            vgg = applications.vgg16.VGG16(include_top=False)
            texture = models.Sequential()
            for i in range(4):
                texture.add(vgg.layers[i])
            def perceptual_loss(y_true, y_pred):
                return K.mean(K.square(texture(y_true) - texture(y_pred)))
            transcoder_loss = lambda x, y: alpha * losses.mean_squared_error(x, y) + (1 - alpha) * perceptual_loss(x, y)
        else:
            transcoder_loss = losses.mean_squared_error
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
    enable_gan = params['enable_gan']
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
    if enable_gan and os.path.exists(discriminator_weights):
        discriminator.load_weights(discriminator_weights)
    if enable_classifier and os.path.exists(classifier_weights):
        classifier.load_weights(classifier_weights)

    print("Starting mode {}".format(mode))
    if mode == 'train':
        for epoch in range(epochs):
            train(encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier, encoder_dataset, decoder_dataset, classifier_dataset, **params)
            encoder.save_weights(encoder_weights)
            decoder.save_weights(decoder_weights)
            if enable_gan:
                discriminator.save_weights(discriminator_weights)
            if enable_classifier:
                classifier.save_weights(classifier_weights)
            demonstrate(transcoder, encoder_dataset, decoder_dataset, **params)
            if enable_gan:
                hallucinate(decoder, decoder_dataset, **params)
    elif mode == 'evaluate':
        evaluate(transcoder, encoder_dataset, decoder_dataset, **params)
    elif mode == 'demo':
        demonstrate(transcoder, encoder_dataset, decoder_dataset, **params)
        if enable_gan:
            hallucinate(decoder, decoder_dataset, **params)
    elif mode == 'dream':
        dream(encoder, decoder, encoder_dataset, decoder_dataset, **params)
    elif mode == 'counterfactual':
        counterfactual(encoder, decoder, classifier, encoder_dataset, decoder_dataset, classifier_dataset, **params)
