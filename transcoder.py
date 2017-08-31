import os
import sys
import numpy as np
import time
import util
import imutil

import dataset_builder
import model_builder
import training
import latent_space


def main(**params):
    mode = params['mode']
    util.chdir_to_experiment(params['experiment_name'])
    # TODO: tee stdout without breaking interactive tools like pdb
    #util.redirect_stdout_stderr(params['stdout_filename'])

    datasets = dataset_builder.build_datasets(**params)
    models = model_builder.build_models(datasets, **params)
    model_builder.load_weights(models, **params)

    func = globals().get(mode)
    if not func:
        raise ValueError("Mode {} is not implemented in {}".format(mode, __name__))
    func(models, datasets, **params)


def train(models, datasets, **params):
    epochs = params['epochs']
    encoder_weights = params['encoder_weights']
    decoder_weights = params['decoder_weights']
    discriminator_weights = params['discriminator_weights']
    classifier_weights = params['classifier_weights']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']

    encoder = models['encoder']
    decoder = models['decoder']
    discriminator = models['discriminator']
    classifier = models['classifier']

    try:
        for epoch in range(epochs):
            training.train(models, datasets, **params)
            encoder.save_weights(encoder_weights)
            decoder.save_weights(decoder_weights)
            if enable_discriminator:
                discriminator.save_weights(discriminator_weights)
            if enable_classifier:
                classifier.save_weights(classifier_weights)
            demonstrate(models, datasets, **params)
    except KeyboardInterrupt:
        print("\n\nQuitting at epoch {} due to ctrl+C".format(epoch))


def visualize_trajectory(models, datasets, **params):
    from tqdm import tqdm
    thought_vector_size = params['thought_vector_size']
    encoder = models['encoder']
    decoder = models['decoder']
    classifier = models['classifier']

    print("Encoding training data into latent space...")
    encoder_dataset = datasets['encoder']
    latent_vectors = []
    for i in tqdm(range(encoder_dataset.count())):
        img = encoder_dataset.get_example(i)
        img = np.array(img)
        z = encoder.predict(img)
        latent_vectors.append(z)
    latent_vectors = np.array(latent_vectors).reshape((-1, thought_vector_size))

    for j in range(10):
        trajectory = latent_space.random_trajectory(z, length=100)

        def closest_example(z, candidates):
            from scipy.spatial.distance import cdist
            idx = np.argmin(cdist(z, candidates))
            return encoder_dataset.get_example(idx)[0]

        from imutil import show
        timestamp = str(int(time.time()))
        filename = 'trajectory_examples_{}_{:02d}.mjpeg'.format(timestamp, j)
        for z_val in trajectory:
            closest_real_img = closest_example(z_val, latent_vectors)
            hallucinated_img = decoder.predict(np.array(z_val))[0]
            combined_img = np.array([closest_real_img, hallucinated_img])
            show(combined_img, video_filename=filename, resize_to=None)
        os.system('ffmpeg -i {0} {1} && rm {0}'.format(filename, filename.replace('mjpeg', 'mp4')))


def evaluate(models, datasets, **params):
    batch_size = params['batch_size']

    transcoder = models['transcoder']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']

    input_count, output_count = encoder_dataset.count(), decoder_dataset.count()
    assert input_count == output_count

    def eval_generator():
        for i in range(0, input_count, batch_size):
            sys.stderr.write("\r[K{} / {}".format(i, input_count))
            yield training.get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)
        sys.stderr.write("\n")
        # HACK: Keras is dumb and asynchronously queues up batches beyond the last one
        # Could be solved if evaluate_generator() workers used an atomic counter
        while True:
            yield training.get_batch(encoder_dataset, decoder_dataset, base_idx=i, **params)

    batch_count = input_count / batch_size
    scores = transcoder.evaluate_generator(eval_generator(), steps=batch_count)
    print("")
    print("Completed evaluation on {} input items ({} batches)".format(input_count, batch_count))
    print("input: {}".format(params['encoder_input_filename']))
    print("output: {}".format(params['decoder_input_filename']))
    print("encoder: {}".format(params['encoder_weights']))
    print("decoder: {}".format(params['decoder_weights']))
    for name, val in zip(['loss'] + transcoder.metrics, scores):
        print("{}: {:.5f}".format(name, val))


def demonstrate(models, datasets, **params):
    enable_discriminator = params['enable_discriminator']
    batch_size = params['batch_size']

    transcoder = models['transcoder']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']

    X_list = encoder_dataset.empty_batch(**params)
    Y_list = decoder_dataset.empty_batch(**params)
    np.random.seed(123)
    for i in range(batch_size):
        idx = np.random.randint(encoder_dataset.count())
        x_list = encoder_dataset.get_example(idx, **params)
        y_list = decoder_dataset.get_example(idx, **params)
        for X, x in zip(X_list, x_list):
            X[i] = x
        for Y, y in zip(Y_list, y_list):
            Y[i] = y
    np.random.seed()
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

    if enable_discriminator:
        hallucinate(models, datasets, **params)


def hallucinate(models, datasets, dist='gaussian', **params):
    batch_size = params['batch_size']
    thought_vector_size = params['thought_vector_size']

    decoder = models['decoder']

    decoder_dataset = datasets['decoder']

    np.random.seed(123)
    X_decoder = np.random.normal(0, 1, size=(batch_size, thought_vector_size))
    np.random.seed()
    X_generated = decoder.predict(X_decoder)
    print("Hallucinated outputs:")
    for j in range(len(X_generated)):
        print(' ' + decoder_dataset.unformat_output(X_generated[j]))
    fig_filename = '{}_halluc_{}.jpg'.format(int(time.time()), dist)
    imutil.show_figure(filename=fig_filename, resize_to=None)


def dream(models, datasets, **params):
    video_filename = params['video_filename']
    dream_frames_per_example = params['dream_fps']
    dream_examples = params['dream_examples']
    if params['batch_size'] < 2:
        raise ValueError("--batch-size of {} is too low, dream() requires a larger batch size".format(params['batch_size']))

    encoder = models['encoder']
    decoder = models['decoder']

    encoder_dataset = datasets['encoder']

    # Select two inputs in the dataset
    start_idx = np.random.randint(encoder_dataset.count())
    end_idx = np.random.randint(encoder_dataset.count())
    for _ in range(dream_examples):
        input_start = encoder_dataset.get_example(start_idx, **params)
        input_end = encoder_dataset.get_example(end_idx, **params)

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
            c = float(i) / dream_frames_per_example
            v = c * latent_end + (1 - c) * latent_start
            img = decoder.predict(np.expand_dims(v, axis=0))[0]
            # Uncomment for slow/verbose logging
            # decoder_dataset.unformat_output(img)
            caption = '{} {}'.format(start_idx, i)
            imutil.show(img, video_filename=video_filename, resize_to=(512,512), display=(i % 100 == 0), caption=caption)
        print("Done")
        start_idx = end_idx
        end_idx = np.random.randint(encoder_dataset.count())


def counterfactual(models, datasets, **params):
    video_filename = params['video_filename']

    encoder = models['encoder']
    decoder = models['decoder']
    classifier = models['classifier']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']
    classifier_dataset = datasets.get('classifier')

    training_gen = training.train_generator(encoder_dataset, decoder_dataset, **params)
    X, _ = next(training_gen)
    X = X[:1]
    imutil.show(X)
    Z = encoder.predict(X)

    trajectory_path = []

    for _ in range(3):
        trajectory = latent_space.counterfactual_trajectory(
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


