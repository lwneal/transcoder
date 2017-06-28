import os
import json
import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical

import model_img
import imutil

DATA_DIR = os.path.expanduser('~/data/')


class ImageDataset(object):
    def __init__(self, input_filename=None, **params):
        if not input_filename:
            raise ValueError("No input filename supplied. See options with --help")
        lines = open(input_filename).readlines()

        self.regions = map(json.loads, lines)
        print("Input file {} contains {} image regions".format(input_filename, len(self.regions)))

    def random_idx(self):
        return np.random.randint(0, len(self.regions))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        if idx > len(self.regions):
            raise ValueError("Example index {} out of bounds (number of examples is {})".format(idx, len(self.regions)))
        region = self.regions[idx]
        return self.format_input(region, **params)

    def count(self):
        return len(self.regions)

    def format_input(self, region, **params):
        img_width = params['img_width']
        filename = os.path.join(DATA_DIR, str(region['filename']))
        img = imutil.decode_jpg(filename, resize_to=(img_width, img_width))
        img = img * 1.0 / 255
        return [img]

    def unformat_input(self, X, **params):
        pixels = X[0]
        imutil.add_to_figure(pixels)
        return 'Image input shape: {}'.format(pixels.shape)

    def unformat_output(self, Y, **params):
        imutil.add_to_figure(Y)
        return 'Output image shape: {}'.format(Y.shape)

    def empty_batch(self, **params):
        img_width = params['img_width']
        batch_size = params['batch_size']
        return [np.zeros((batch_size, img_width, img_width, 3), dtype=float)]

    def build_encoder(self, **params):
        return model_img.build_encoder(**params)

    def build_decoder(self, **params):
        return model_img.build_decoder(**params)

    def build_discriminator(self, **params):
        return model_img.build_discriminator(**params)
