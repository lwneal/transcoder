import os
import json
import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical

import model_img_region
from imutil import decode_jpg, show

DATA_DIR = os.path.expanduser('~/data/')


class ImageRegionDataset(object):
    def __init__(self, input_filename=None, **params):
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
        image_global = decode_jpg(filename)
        image_local = decode_jpg(filename, crop_to_box=region['bbox'])
        x0, x1, y0, y1 = region['bbox']
        ctx_vector = np.array([x0, x1, y0, y1, (x1 - x0) * (y1 - y0)], dtype=float)
        return [image_global, image_local, ctx_vector]

    def unformat_input(self, X, **params):
        pixels = X[0]
        show(pixels)
        return 'Image input shape: {}'.format(pixels.shape)

    def unformat_output(self, preds, **params):
        return 'Image region prediction shape: {}'.format(preds.shape)

    def empty_batch(self, **params):
        img_width = params['img_width']
        batch_size = params['batch_size']
        return [np.zeros((batch_size, img_width, img_width, 3), dtype=float),
                np.zeros((batch_size, img_width, img_width, 3), dtype=float),
                np.zeros((batch_size, 5), dtype=float)]

    def build_encoder(self, **params):
        return model_img_region.build_encoder(**params)

    def build_decoder(self, **params):
        return model_img_region.build_decoder(**params)

    def build_discriminator(self, **params):
        return model_img_region.build_discriminator(**params)
