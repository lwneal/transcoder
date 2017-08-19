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


def train(models, datasets, **params):
    batch_size = params['batch_size']
    batches_per_epoch = params['batches_per_epoch']
    thought_vector_size = params['thought_vector_size']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']
    freeze_encoder = params['freeze_encoder']
    freeze_decoder = params['freeze_decoder']
    enable_transcoder = not (freeze_encoder and freeze_decoder)
    discriminator_iters = params['discriminator_iters']

    encoder = models['encoder']
    decoder = models['decoder']
    discriminator = models['discriminator']
    transcoder = models['transcoder']
    cgan = models['cgan']
    transclassifier = models['transclassifier']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']
    classifier_dataset = datasets.get('classifier')

    training_gen = train_generator(encoder_dataset, decoder_dataset, **params)
    if enable_classifier:
        classifier_gen = train_generator(encoder_dataset, classifier_dataset, **params)
    clipping_time = 0
    training_start_time = time.time()
    t_avg_loss = 0
    t_avg_accuracy = 0
    d_avg_loss = 0
    g_avg_loss = 0
    c_avg_loss = 0
    c_avg_accuracy = 0

    check_weights(models)

    print("Training...")
    for i in range(0, batches_per_epoch):
        sys.stderr.write('\n')
        sys.stderr.write("\r[K\r{}/{} bs {}, D_loss {:.3f}, G_loss {:.3f} T_loss {:.3f} T_acc {:.3f} C_loss {:.3f} C_acc {:.3f}".format(
            i + 1, batches_per_epoch, batch_size,
            d_avg_loss,
            g_avg_loss,
            t_avg_loss,
            t_avg_accuracy,
            c_avg_loss,
            c_avg_accuracy
        ))
        sys.stderr.write('\n')

        # Train encoder and decoder on labeled X -> Y pairs
        if enable_transcoder:
            X, Y = next(training_gen)
            t_weights = transcoder.get_weights()
            loss, accuracy = transcoder.train_on_batch(X, Y)
            check_gradient(t_weights, transcoder.get_weights(), name='Transcoder')
            t_avg_loss = .95 * t_avg_loss + .05 * loss
            t_avg_accuracy = .95 * t_avg_accuracy + .05 * accuracy

        if enable_discriminator:
            for _ in range(discriminator_iters):
                # Think some random thoughts
                X_decoder = np.random.normal(0, 1, size=(batch_size, thought_vector_size))

                # Decode those random thoughts into hallucinations
                X_generated = decoder.predict(X_decoder)

                # Get some real examples to contrast with the generated ones
                _, X_real = next(training_gen)
                Y_disc = np.ones(batch_size)
                Y_dummy = np.zeros(batch_size)

                # Discriminator optimizes three loss functions:
                # Wasserstein loss for batch of real inputs
                # Wasserstein loss for batch of generated inputs
                # Gradient penalty loss for random interpolation of real and generated inputs
                disc_inputs = [X_real,  X_generated]
                disc_targets = [Y_disc, -Y_disc, Y_dummy]

                disc_weights = discriminator.get_weights()
                outputs = discriminator.train_on_batch(disc_inputs, disc_targets)
                check_gradient(disc_weights, discriminator.get_weights(), name='Discriminator')
                loss = outputs[0]
                d_avg_loss = .95 * d_avg_loss + .05 * loss


        if enable_classifier:
            # Update the encoder and classifier
            X, Y = next(classifier_gen)
            c_weights = transclassifier.get_weights()
            loss, accuracy = transclassifier.train_on_batch(X, Y)
            check_gradient(c_weights, transclassifier.get_weights(), name='Classifier')
            c_avg_loss = .95 * c_avg_loss + .05 * loss
            c_avg_accuracy = .95 * c_avg_accuracy + .05 * accuracy

        if enable_discriminator:
            # Update generator based on a random thought vector
            X_encoder = np.random.normal(size=(batch_size, thought_vector_size))
            Y_disc = np.ones(batch_size)

            decoder_weights_before = decoder.get_weights()
            loss, accuracy = cgan.train_on_batch(X_encoder, Y_disc)
            check_gradient(decoder_weights_before, decoder.get_weights(), name='Decoder')
            g_avg_loss = .95 * g_avg_loss + .05 * loss

    sys.stderr.write('\n')
    print("Trained for {:.2f} s (spent {:.2f} s clipping)".format(time.time() - training_start_time, clipping_time))
    sys.stderr.write('\n')


def check_gradient(old_weights, new_weights, name):
    #print("Weight Update for {}".format(name))
    differences = [new - old for new, old in zip(new_weights, old_weights)]
    minnest = 0
    maxest = 0
    for w, diff in zip(new_weights, differences):
        d_min = diff.min()
        d_max = diff.max()
        minnest = min(minnest, d_min)
        maxest = max(maxest, d_max)
        #print("\t{} min {:.5f}  max {:.5f}  mean {:.5f}".format(w.shape, d_min, d_max, abs(diff).mean()))
        if np.isnan(d_min):
            raise ValueError("NaN in model {} layer {}".format(name, w.shape))
    print("{} min {:.4f} max {:.4f}".format(name, minnest, maxest))


def check_weights(models):
    print("Model Weights:")
    danger_zone = 1000.0
    for name in ['encoder', 'decoder', 'discriminator', 'classifier']:
        model = models[name]
        weights = model.get_weights()
        min_weight = min([w.min() for w in weights])
        max_weight = max([w.max() for w in weights])
        print("{:32} \tmin {:.04f} \tmax {:.04f}".format(name, min_weight, max_weight))
        if max_weight > danger_zone or min_weight < -danger_zone:
            print("Warning: Weight for model {} above limit {}".format(name, danger_zone))
            for i, layer in enumerate(weights):
                print("{:4}\t shape {:32} \tmin {:.04f} \tmax {:.04f}".format(
                    i, layer.shape, layer.min(), layer.max()))


def check_latent_distribution(encoder, X_real):
    # Aside: What is the distribution of E(X)? Is it Gaussian?
    latent = encoder.predict(X_real)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figsize = (30,30)
    from imutil import show

    import scipy
    for i in range(10):
        c = scipy.stats.pearsonr(latent[:,0], latent[:,i])
        print("Correlation between axes {} and {} is {}".format(0, i, c))
        plt.scatter(latent[:,0], latent[:,i])
    plt.savefig('/tmp/foobar.png')
    show('/tmp/foobar.png', resize_to=None)
    import pdb; pdb.set_trace()
