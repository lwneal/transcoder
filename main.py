"""
Usage:
        main.py [options]

Options:
      --experiment-name=<n>             Name of the experiment [Required]
      --stdout-filename=<f>             Filename for copy of stdout [default: stdout.txt]
      --encoder-input-filename=<f>      Dataset or text file for encoder
      --decoder-input-filename=<f>      Dataset or text file for decoder
      --classifier-input-filename=<f>   Optional dataset or text file for classifier [default: None]
      --encoder-datatype=<type>         One of: img, bbox, vq, txt, lab [default: None]
      --decoder-datatype=<type>         One of: img, bbox, vq, txt, lab [default: None]
      --classifier-datatype=<type>      One of: img, bbox, vq, txt, lab [default: None]
      --encoder-weights=<name>          Filename for saved model [default: None]
      --decoder-weights=<name>          Filename for saved model [default: None]
      --discriminator-weights=<name>    Filename for saved model [default: None]
      --classifier-weights=<name>       Filename for saved model [default: None]
      --encoder-model=<m>               Model name for encoder (see models.py) [default: None]
      --decoder-model=<m>               Model name for decoder (see models.py) [default: None]
      --discriminator-model=<m>         Model name for discriminator (see models.py) [default: None]
      --classifier-model=<m>            Model name for classifier (see models.py) [default: None]
      --epochs=<epochs>                 Number of epochs to train [default: 2000].
      --batches-per-epoch=<b>           Number of batches per epoch [default: 1000].
      --batch-size=<size>               Batch size for training [default: 16]
      --training-iters-per-gan=<iters>  Iterations of normal training per iteration of GAN [default: 3.0]
      --discriminator-per-generator=<n> Iterations of D updates per G update [default: 3]
      --max-words-encoder=<ewords>      Number of words of context (the N in N-gram) [default: 12]
      --max-words-decoder=<dwords>      Number of words of context (the N in N-gram) [default: 12]
      --thought-vector-size=<size>      Size of encoder output (decoder input) vector [default: 2048]
      --img-width=<width>               Resize images to this width in pixels [default: 64]
      --enable-perceptual-loss=<n>      Enable VGG16 3-layer perceptual loss [default: True]
      --perceptual-loss-layers=<n>      Number of VGG16 layers to use for P-loss [default: 5]
      --perceptual-loss-alpha=<n>       Value [0,1] modulating perceptual loss [default: .5]
      --wordvec-size=<size>             Number of units in word embedding [default: 1024]
      --rnn-type=<type>                 One of LSTM, GRU [default: GRU]
      --rnn-size=<size>                 Number of output units in RNN [default: 1024]
      --rnn-layers=<layers>             Number of layers of RNN to use [default: 1]
      --tokenize=<tokenize>             If True, input text will be tokenized [default: False]
      --lowercase=<lower>               If True, lowercase all words [default: True]
      --vocab-rarity=<v>                Minimum number of occurrences of a word [default: 1]
      --load-encoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --load-decoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --mode=<mode>                     One of train, evaluate, demo, dream, counterfactual [default: train]
      --max-temperature=<temp>          Sampling temperature for log-Boltzmann distribution [default: 1.0]
      --freeze-encoder=<freeze>         Freeze weights for the encoder [default: False]
      --freeze-decoder=<freeze>         Freeze weights for the decoder [default: False]
      --enable-gan=<bool>               If False, no GAN training will be applied [default: True]
      --enable-classifier=<bool>        If True, train with a classifier for counterfactuals [default: False]
      --video-filename=<fn>             Output video filename for dream mode [default: output.mjpeg]
      --dream-fps=<n>                   Integer, number of frames between dream examples [default: 30]
      --vocabulary-filename=<n>         Filename to draw vocabulary from, to match label indices in test/train folds [default: None]
"""
import sys
import os
from docopt import docopt
from pprint import pprint


def get_params():
    args = docopt(__doc__)
    return {argname(k): argval(args[k]) for k in args}


def argname(k):
    return k.strip('<').strip('>').strip('--').replace('-', '_')


def argval(val):
    if hasattr(val, 'lower') and val.lower() in ['true', 'false']:
        return val.lower().startswith('t')
    try:
        return int(val)
    except:
        pass
    try:
        return float(val)
    except:
        pass
    if val == 'None':
        return None
    return val


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    import re
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    return unicode(re.sub('[-\s]+', '-', value))


class Logger(object):
    def __init__(self, name='stdout.log'):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


if __name__ == '__main__':
    params = get_params()

    for var in ['experiment_name', 'encoder_input_filename', 'decoder_input_filename']:
        if not params[var]:
            print(__doc__)
            raise ValueError("Empty value for required option {}".format(var))

    name = slugify(unicode(params['experiment_name']))
    os.chdir(os.path.expanduser('~/results'))
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    if params['encoder_weights'] is None:
        params['encoder_weights'] = 'encoder_{}.h5'.format(params['encoder_model'])
    if params['decoder_weights'] is None:
        params['decoder_weights'] = 'decoder_{}.h5'.format(params['decoder_model'])
    if params['discriminator_weights'] is None:
        params['discriminator_weights'] = 'disc_{}.h5'.format(params['discriminator_model'])
    if params['classifier_weights'] is None:
        params['classifier_weights'] = 'classifier_{}.h5'.format(params['classifier_model'])

    if params['stdout_filename']:
        sys.stdout = Logger(params['stdout_filename'])

    import transcoder
    transcoder.main(**params)
