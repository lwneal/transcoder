"""
Usage:
        main.py [options]

Options:
      --encoder-input-filename=<txt>    Input text file the encoder will read (eg. English sentences)
      --decoder-input-filename=<txt>    Input text file the decoder will try to copy (eg. German sentences)
      --encoder-datatype=<type>         One of: img, bbox, vq, text [default: None]
      --decoder-datatype=<type>         One of: img, bbox, vq, text [default: None]
      --encoder-weights=<name>          Filename for saved model [default: default_encoder.h5]
      --decoder-weights=<name>          Filename for saved model [default: default_decoder.h5]
      --discriminator-weights=<name>    Filename for saved model [default: default_discriminator.h5]
      --epochs=<epochs>                 Number of epochs to train [default: 2000].
      --batches-per-epoch=<b>           Number of batches per epoch [default: 1000].
      --batch-size=<size>               Batch size for training [default: 16]
      --training-iters-per-gan=<iters>  Iterations of normal training per iteration of GAN [default: 5.0]
      --thought-vector-size=<size>      Size of encoder output (decoder input) vector [default: 204
      --max-words-encoder=<ewords>      Number of words of context (the N in N-gram) [default: 12]
      --max-words-decoder=<dwords>      Number of words of context (the N in N-gram) [default: 12]
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
      --load-encoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --load-decoder-vocab=<vocab>      Filename, save/load vocabulary from this file [default: None]
      --mode=<mode>                     One of train, test, demo [default: train]
      --max-temperature=<temp>          Sampling temperature for log-Boltzmann distribution [default: 1.0]
      --freeze-encoder=<freeze>         Freeze weights for the encoder [default: False]
      --freeze-decoder=<freeze>         Freeze weights for the decoder [default: False]
      --enable-gan=<bool>               If False, no GAN training will be applied [default: True]
"""
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


if __name__ == '__main__':
    params = get_params()
    import transcoder
    transcoder.main(**params)
