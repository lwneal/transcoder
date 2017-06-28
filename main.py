"""
Usage:
        main.py [options]

Options:
      --experiment-name=<n>             Name of the experiment [Required]
      --stdout-filename=<f>             Filename for copy of stdout [default: stdout.txt]
      --encoder-input-filename=<txt>    Input text file the encoder will read (eg. English sentences)
      --decoder-input-filename=<txt>    Input text file the decoder will try to copy (eg. German sentences)
      --encoder-datatype=<type>         One of: img, bbox, vq, txt, lab [default: None]
      --decoder-datatype=<type>         One of: img, bbox, vq, txt, lab [default: None]
      --encoder-weights=<name>          Filename for saved model [default: encoder.h5]
      --decoder-weights=<name>          Filename for saved model [default: decoder.h5]
      --discriminator-weights=<name>    Filename for saved model [default: discriminator.h5]
      --epochs=<epochs>                 Number of epochs to train [default: 2000].
      --batches-per-epoch=<b>           Number of batches per epoch [default: 1000].
      --batch-size=<size>               Batch size for training [default: 16]
      --training-iters-per-gan=<iters>  Iterations of normal training per iteration of GAN [default: 5.0]
      --max-words-encoder=<ewords>      Number of words of context (the N in N-gram) [default: 12]
      --max-words-decoder=<dwords>      Number of words of context (the N in N-gram) [default: 12]
      --thought-vector-size=<size>      Size of encoder output (decoder input) vector [default: 2048]
      --img-width=<width>               Resize images to this width in pixels [default: 64]
      --wordvec-size=<size>             Number of units in word embedding [default: 1024]
      --rnn-type=<type>                 One of LSTM, GRU [default: GRU]
      --rnn-size=<size>                 Number of output units in RNN [default: 1024]
      --rnn-layers=<layers>             Number of layers of RNN to use [default: 1]
      --pretrained-encoder=<name>       For image encoders, one of: vgg16, resnet50 [default: None]
      --csr-size=<size>                 Number of output units in conv spatial recurrent [default: 256]
      --csr-layers=<layers>             Number of conv spatial RNN layers to use [default: 0]
      --tokenize=<tokenize>             If True, input text will be tokenized [default: False]
      --lowercase=<lower>               If True, lowercase all words [default: True]
      --vocab-rarity=<v>                Minimum number of occurrences of a word [default: 1]
      --load-encoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --load-decoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --mode=<mode>                     One of train, test, demo, dream [default: train]
      --max-temperature=<temp>          Sampling temperature for log-Boltzmann distribution [default: 1.0]
      --freeze-encoder=<freeze>         Freeze weights for the encoder [default: False]
      --freeze-decoder=<freeze>         Freeze weights for the decoder [default: False]
      --enable-gan=<bool>               If False, no GAN training will be applied [default: True]
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
    name = params['experiment_name']
    if not name:
        raise ValueError("Empty value for required option --experiment-name")
    name = slugify(unicode(name))
    os.chdir(os.path.expanduser('~/results'))
    os.mkdir(name)
    os.chdir(name)

    if params['stdout_filename']:
        sys.stdout = Logger(params['stdout_filename'])

    import transcoder
    transcoder.main(**params)
