import sys
import time
import numpy as np


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


def get_batch(encoder_dataset, decoder_dataset, base_idx=None, **params):
    batch_size = params['batch_size']
    # The last batch might need to be a smaller partial batch
    example_count = min(encoder_dataset.count(), decoder_dataset.count())
    if base_idx is not None and base_idx + batch_size > example_count:
        batch_size = example_count - base_idx
        params['batch_size'] = batch_size
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


def train(encoder, decoder, transcoder, discriminator, cgan, classifier, transclassifier, encoder_dataset, decoder_dataset, classifier_dataset, **params):
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    thought_vector_size = params['thought_vector_size']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']
    batches_per_iter = int(params['training_iters_per_gan'])
    freeze_encoder = params['freeze_encoder']
    freeze_decoder = params['freeze_decoder']
    enable_transcoder = not (freeze_encoder and freeze_decoder)
    discriminator_iters = params['discriminator_iters']

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

        if enable_discriminator:
            for _ in range(discriminator_iters):
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

                # WGAN: Clip discriminator weights
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

        if enable_discriminator:
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
