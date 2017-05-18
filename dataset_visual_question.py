import os
import json
import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical

from dataset_word import WordDataset
import model_visual_question
from imutil import decode_jpg, show

DATA_DIR = os.path.expanduser('~/data/')


class VisualQuestionDataset(object):
    def __init__(self, input_filename=None, **params):
        lines = open(input_filename).readlines()
        self.questions = [json.loads(l) for l in lines]
        text = '\n'.join(q['question'] for q in self.questions)
        self.words = WordDataset(input_text=text, is_encoder=True)
        print("Input file {} contains {} visual questions".format(input_filename, len(self.questions)))

    def random_idx(self):
        return np.random.randint(0, len(self.questions))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        question = self.questions[idx]
        return self.format_input(question, **params)

    def format_input(self, question, **params):
        filename = os.path.join(DATA_DIR, str(question['filename']))
        x_img = decode_jpg(filename)
        question = question['question'].lower()
        x_words = self.words.format_input(question, **params)[0]
        return [x_img, x_words]

    def unformat_input(self, pixels, **params):
        show(pixels)
        return 'Image input shape: {}'.format(pixels.shape)

    def unformat_output(self, preds, **params):
        return 'Image region prediction shape: {}'.format(preds.shape)

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        images = np.zeros((batch_size, 224, 224, 3), dtype=float)
        words = self.words.empty_batch(**params)[0]
        return [images, words]

    def build_model(self, **params):
        return model_visual_question.build_encoder(len(self.words.vocab), **params)
