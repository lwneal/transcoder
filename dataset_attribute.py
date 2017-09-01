import json
import random
import numpy as np
import model_labels
from keras.utils import to_categorical

# TODO: This is a hacked up dataset that combines Labels (1 of N) with Attributes (M separate boolean classifications)
# We train the Label with categorical crossentropy and the Attributes with binary crossentropy
# The idea is to learn relationships between classes and attributes
class AttributeDataset(object):
    def __init__(self, input_filename=None, is_encoder=False, **params):
        if is_encoder:
            vocab_filename = params['load_encoder_vocab']
        else:
            vocab_filename = params['load_decoder_vocab']
        if vocab_filename is None:
            vocab_filename = input_filename

        lines = open(input_filename).readlines()
        self.idx_to_name = {}
        self.name_to_idx = {}

        self.labels = []
        self.attributes = []
        self.attribute_names = set()
        
        # Load vocabulary of possible classes
        print("Drawing classes from file {}".format(vocab_filename))
        distinct_labels = sorted(list(set(self.labels)))
        vocab_labels = [json.loads(line)['label'] for line in open(vocab_filename).readlines()]
        distinct_labels = sorted(list(set(vocab_labels)))
        for idx, label in enumerate(distinct_labels):
            self.idx_to_name[idx] = label
            self.name_to_idx[label] = idx

        # Load vocabulary of possible attributes
        for line in open(vocab_filename).readlines():
            item = json.loads(line)
            for attr_name in item:
                if attr_name.startswith('is'):
                    self.attribute_names.add(attr_name)
        self.attribute_names = sorted(list(self.attribute_names))

        # Load labels and attributes
        for line in lines:
            item = json.loads(line)
            self.labels.append(item['label'])
            attrs = {k: item[k] for k in item if k.startswith('is')}
            self.attributes.append(attrs)

        print("Input file {} with {} examples\n\tLabels out of {} classes\n\t {} binary attributes ".format(
            input_filename, len(self.labels), len(distinct_labels), len(self.attribute_names)))

    def random_idx(self):
        return np.random.randint(0, len(self.labels))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        example = (self.labels[idx], self.attributes[idx])
        return self.format_input(example, **params)
    
    def count(self):
        return len(self.labels)

    def format_input(self, example, **params):
        label, attrs = example
        num_classes = len(self.name_to_idx)
        onehot = to_categorical(self.name_to_idx[label], num_classes)
        binary = np.zeros(len(self.attribute_names))
        for i, name in enumerate(sorted(self.attribute_names)):
            value = float(attrs[name])
            # TODO: What if an attribute is not specified? Set to 0.5?
            binary[i] = value
        return [onehot, binary]

    def unformat_input(self, X, **params):
        index = X[0]
        attrs = X[1]
        return [self.idx_to_name[index], attrs]

    def unformat_output(self, preds, **params):
        class_preds = preds[0][0]
        attr_preds = preds[1][0]
        class_idx = np.argmax(class_preds, axis=-1)
        attrs = {}
        for i, name in enumerate(sorted(self.attribute_names)):
            attrs[name] = attr_preds[i]
        return [self.idx_to_name[class_idx], attrs]

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        num_classes = len(self.name_to_idx)
        num_attributes = len(self.attribute_names)
        return [np.zeros((batch_size, num_classes)), np.zeros((batch_size, num_attributes))]
