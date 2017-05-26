import json
import random
import numpy as np
import model_labels
from keras.utils import to_categorical

class LabelDataset(object):
    def __init__(self, input_filename=None, **params):
        if not input_filename:
            raise ValueError("No input filename supplied. See options with --help")
        lines = open(input_filename).readlines()

        self.idx_to_name = {}
        self.name_to_idx = {}
        self.labels = []
        for line in lines:
            item = json.loads(line)
            self.labels.append(item['label'])
        
        distinct_labels = sorted(list(set(self.labels)))
        for idx, label in enumerate(distinct_labels):
            self.idx_to_name[idx] = label
            self.name_to_idx[label] = idx
        print("Input file {} contains {} labels".format(input_filename, len(self.labels)))

    def random_idx(self):
        return np.random.randint(0, len(self.labels))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        label = self.labels[idx]
        return self.format_input(label, **params)
    
    def count(self):
        return len(self.labels)

    def format_input(self, label, **params):
        num_classes = len(self.name_to_idx)
        onehot = to_categorical(self.name_to_idx[label], num_classes)
        return onehot

    def unformat_input(self, X, **params):
        index = X[0]
        return self.idx_to_name[index]

    def unformat_output(self, preds, **params):
        index = np.argmax(preds, axis=-1)
        return self.idx_to_name[index]

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        num_classes = len(self.name_to_idx)
        return [np.zeros((batch_size, num_classes))]

    def build_encoder(self, **params):
        return model_labels.build_encoder(self, **params)

    def build_decoder(self, **params):
        return model_labels.build_decoder(self, **params)

    def build_discriminator(self, **params):
        return model_labels.build_discriminator(self, **params)
