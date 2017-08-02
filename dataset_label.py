import json
import random
import numpy as np
import model_labels
from keras.utils import to_categorical

class LabelDataset(object):
    def __init__(self, input_filename=None, **params):
        vocab_filename = params['vocabulary_filename']

        lines = open(input_filename).readlines()
        self.idx_to_name = {}
        self.name_to_idx = {}
        self.labels = []
        for line in lines:
            item = json.loads(line)
            self.labels.append(item['label'])
        
        # HACK: on training set, use test set as the source of class indices
        # Fixes crash when test set is small and is missing some classes
        distinct_labels = sorted(list(set(self.labels)))
        if vocab_filename:
            print("Drawing classes from file {}".format(vocab_filename))
            vocab_labels = [json.loads(line)['label'] for line in open(vocab_filename).readlines()]
            distinct_labels = sorted(list(set(vocab_labels)))

        for idx, label in enumerate(distinct_labels):
            self.idx_to_name[idx] = label
            self.name_to_idx[label] = idx
        print("Input file {} contains {} labels out of {} classes".format(input_filename, len(self.labels), len(distinct_labels)))

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
