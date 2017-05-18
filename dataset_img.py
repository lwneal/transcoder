import os
import json
import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical

import model_img
from imutil import decode_jpg, show

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
        region = self.regions[idx]
        return self.format_input(region, **params)

    def format_input(self, region, **params):
        filename = os.path.join(DATA_DIR, str(region['filename']))
        img = decode_jpg(filename)
        return [img]

    def unformat_input(self, X, **params):
        pixels = X[0]
        show(pixels)
        return 'Image input shape: {}'.format(pixels.shape)

    def unformat_output(self, preds, **params):
        return 'Image region prediction shape: {}'.format(preds.shape)

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        return [np.zeros((batch_size, 224, 224, 3), dtype=float)]

    def build_model(self, **params):
        return model_img.build_encodet(**params)
