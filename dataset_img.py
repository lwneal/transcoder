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
    def __init__(self, input_filename=None, is_encoder=False, **params):
        if not input_filename:
            raise ValueError("No input filename supplied. See options with --help")
        lines = open(input_filename).readlines()

        if is_encoder:
            self.img_shape = (params['img_width_encoder'], params['img_width_encoder'])
        else:
            self.img_shape = (params['img_width_decoder'], params['img_width_decoder'])

        self.regions = [json.loads(l) for l in lines]
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

    def format_input(self, region, use_box=True, **params):
        filename = os.path.join(DATA_DIR, str(region['filename']))
        box = region.get('box') if use_box else None
        img = imutil.decode_jpg(filename, resize_to=self.img_shape, crop_to_box=box)
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
        batch_size = params['batch_size']
        return [np.zeros((batch_size, self.img_shape[0], self.img_shape[1], 3), dtype=float)]

