#!/usr/bin/env python
"""
Usage:
    enhancer [options]

Options:
    --dataset=<n>            Dataset with images and labels eg. cub200 [default: cub200]
    --thought-vector=<n>     Size of the latent space [default: 32]
    --encoder=<n>            Name of the counterfactual encoder model [default: simplecnn_7a]
    --decoder=<n>            Name of the counterfactual decoder model [default: simpledeconv_a]
    --upscaler=<n>           Name of the upscaler model [default: upscale_2a]
    --epochs=<n>             Number of epochs [default: 10]
    --perceptual-layers=<n>  Perceptual loss depth [default: 2]
    --timestamp=<n>          Timestamp of previously-trained network [default: None]
"""
import time

if __name__ == '__main__':
    import docopt
    arguments = docopt.docopt(__doc__)
    if timestamp == 'None':
        print("Error: --timestamp required")
        exit()

    # TODO: Load the already-trained encoder/decoder, train an upscaler
    print("Load the already-trained encoder/decoder, train an upscaler")
