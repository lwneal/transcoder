#!/usr/bin/env python
"""
Usage:
    counterfactual [options]

Options:
    --dataset=<n>            Dataset with images and labels eg. mnist, cub200 [default: mnist]
    --thought-vector=<n>     Size of the latent space [default: 16]
    --encoder=<n>            Name of the encoder model [default: stridecnn_10a]
    --decoder=<n>            Name of the decoder model [default: simpledeconv_a]
    --classifier=<n>         Name of the classifier model [default: mlp_2a]
    --epochs=<n>             Number of epochs [default: 100]
    --decay=<n>              Training rate decay [default: .0001]
    --learning-rate=<n>      Initial training rate [default: .001]
    --gan-weight=<n>         GAN training per autoencoder training [default: 3.0]
    --perceptual-layers=<n>  Perceptual loss depth [default: 3]
    --img-width=<n>          Width of images through transcoder [default: 32]
    --timestamp=<n>          Timestamp of previously-trained network (new ts if left default) [default: None]
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import docopt
import os
import time
import json
from main import get_params
from util import pushd


def save_params(params, filename=None):
    if not filename:
        filename = os.path.expanduser('~/results/{}'.format(filename))
    with open(filename, 'w') as fp:
        fp.write(json.dumps(params, indent=2))


def counterfactual():
    import transcoder
    arguments = docopt.docopt(__doc__)

    dataset = arguments['--dataset']
    thought_vector_size = int(arguments['--thought-vector'])
    encoder_model = arguments['--encoder']
    decoder_model = arguments['--decoder']
    classifier_model = arguments['--classifier']
    epochs = int(arguments['--epochs'])
    decay = float(arguments['--decay'])
    learning_rate = float(arguments['--learning-rate'])
    gan_weight = float(arguments['--gan-weight'])
    perceptual_layers = int(arguments['--perceptual-layers'])
    img_width = int(arguments['--img-width'])
    try:
        experiment_timestamp = int(arguments['--timestamp'])
    except ValueError:
        experiment_timestamp = int(time.time())

    experiment_name = '-'.join(str(x) for x in [
        dataset,
        thought_vector_size,
        encoder_model,
        decoder_model,
        classifier_model,
        experiment_timestamp,
    ])

    save_params(arguments)

    timestamp = int(time.time())

    #Download the dataset if it doesn't already exist
    # TODO: security lol
    os.system('scripts/download_{}.py'.format(dataset))

    train_dataset = os.path.expanduser('~/data/{}_train.dataset'.format(dataset))
    test_dataset = os.path.expanduser('~/data/{}_test.dataset'.format(dataset))

    # Fill params with defaults
    import main
    defaults = {opt.long: opt.value for opt in docopt.parse_defaults(main.__doc__)}
    params = {main.argname(k): main.argval(defaults[k]) for k in defaults}

    params['experiment_name'] = experiment_name
    params['encoder_input_filename'] = test_dataset
    params['decoder_input_filename'] = test_dataset
    params['classifier_input_filename'] = test_dataset
    params['vocabulary_filename'] = train_dataset
    params['encoder_datatype'] = 'img'
    params['decoder_datatype'] = 'img'
    params['classifier_datatype'] = 'lab'
    params['encoder_model'] = encoder_model
    params['decoder_model'] = decoder_model
    params['discriminator_model'] = encoder_model
    params['classifier_model'] = classifier_model
    params['thought_vector_size'] = thought_vector_size
    params['img_width_encoder'] = img_width
    params['img_width_decoder'] = img_width
    params['epochs'] = epochs
    params['learning_rate'] = learning_rate
    params['decay'] = decay
    params['perceptual_loss_layers'] = perceptual_layers
    params['batches_per_epoch'] = 200
    params['enable_classifier'] = True
    params['enable_discriminator'] = True
    params['enable_perceptual_loss'] = True
    params['encoder_weights'] = 'encoder_{}.h5'.format(encoder_model)
    params['decoder_weights'] = 'decoder_{}.h5'.format(decoder_model)
    params['classifier_weights'] = 'classifier_{}.h5'.format(classifier_model)
    params['discriminator_weights'] = 'discriminator_{}.h5'.format(encoder_model)
    params['training_iters_per_gan'] = gan_weight

    # First train a manifold
    train_params = params.copy()
    train_params['stdout_filename'] = 'stdout_train_{}.txt'.format(timestamp)
    train_params['encoder_input_filename'] = train_dataset
    train_params['decoder_input_filename'] = train_dataset
    train_params['classifier_input_filename'] = train_dataset
    train_params['mode'] = 'train'
    if train_params['epochs'] > 0:
        transcoder.main(**train_params)

    # Evaluate the classifier
    eval_params = params.copy()
    eval_params['decoder_model'] = params['classifier_model']
    eval_params['decoder_datatype'] = params['classifier_datatype']
    eval_params['decoder_weights'] = params['classifier_weights']
    eval_params['enable_classifier'] = False
    eval_params['enable_discriminator'] = False
    eval_params['stdout_filename'] = 'stdout_eval_{}.txt'.format(timestamp)
    eval_params['mode'] = 'evaluate'
    transcoder.main(**eval_params)

    # Generate a "dream" video
    dream_params = params.copy()
    dream_params['video_filename'] = 'dream_output_{}.mjpeg'.format(timestamp)
    dream_params['stdout_filename'] = 'stdout_dream_{}.txt'.format(timestamp)
    dream_params['enable_discriminator'] = False
    dream_params['mode'] = 'dream'
    transcoder.main(**dream_params)

    # Re-encode the video to mp4 for storage
    encode_video(experiment_name, dream_params['video_filename'])

    # Add counterfactuals
    counter_params = params.copy()
    counter_params['video_filename'] = 'counterfactual_output_{}.mjpeg'.format(timestamp)
    counter_params['stdout_filename'] = 'stdout_counterfactual_{}.txt'.format(timestamp)
    counter_params['enable_discriminator'] = False
    counter_params['mode'] = 'counterfactual'
    for _ in range(3):
        transcoder.main(**counter_params)

    # Re-encode the video to mp4 for storage
    encode_video(experiment_name, counter_params['video_filename'])

    # Touch a file to mark the experiment as finished
    filename = os.path.expanduser('~/results/{}/finished'.format(experiment_name))
    open(filename, 'w').write('OK')


def encode_video(experiment_name, video_name):
    # TODO: security lol
    dirname = '~/results/{}'.format(experiment_name)
    input_name = os.path.join(dirname, video_name)
    output_name = input_name.replace('.mjpeg', '.mp4')
    with pushd(dirname):
        os.system('ffmpeg -y -i {} {}'.format(input_name, output_name))
        os.system('rm {}'.format(input_name))


if __name__ == '__main__':
    counterfactual()
