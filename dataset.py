import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical


PAD_TOKEN = ' '
UNK_TOKEN = '<unk>'
START_TOKEN = '<bos>'
END_TOKEN = '<eos>'

class Dataset(object):
    def __init__(self, input_filename=None, encoder=False, **params):
        if not input_filename:
            raise ValueError("No input filename supplied. See options with --help")
        text = open(input_filename).read()

        if params.get('tokenize'):
            text = remove_unicode(text)
            text = tokenize_text(text)
        
        if params.get('lowercase'):
            text = text.lower()

        self.encoder = encoder
        self.sentences = text.splitlines()
        print("Input file {} contains {} sentences".format(input_filename, len(self.sentences)))
        self.vocab = get_vocab(text)
        print("Vocabulary contains {} words from {} to {}".format(len(self.vocab), self.vocab[4], self.vocab[-1]))
        self.word_to_idx = {}
        self.idx_to_word = {}
        for i, word in enumerate(self.vocab):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

    def random_idx(self):
        return np.random.randint(0, len(self.sentences))

    def get_example(self, idx, **params):
        sentence = self.sentences[idx]
        indices = self.indices(sentence)
        if self.encoder:
            return left_pad(indices, **params)
        else:
            return to_categorical(right_pad(indices, **params), len(self.vocab))

    def batch(self, **params):
        batch_size = params['batch_size']
        max_words = params['max_words']
        if self.encoder:
            return np.zeros((batch_size, max_words), dtype=int)
        else:
            return np.zeros((batch_size, max_words, len(self.vocab)))

    def indices(self, words):
        # TODO: Properly tokenize?
        unk = self.word_to_idx[PAD_TOKEN]
        return [self.word_to_idx.get(w, unk) for w in words.split()]

    def words(self, indices):
        # TODO: Properly detokenize and join
        return [self.idx_to_word.get(i) for i in indices]

    def get_empty_batch(self, batch_size=1, max_words=12, **params):
        X = np.zeros((batch_size, max_words))
        X[:, -1] = self.word_to_idx[START_TOKEN]
        return X


def get_vocab(text, n=3):
    word_count = {}
    for word in text.split():
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
    words = [w for w in text.split() if word_count[w] > n]
    return sorted(list(set(words + [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN])))


def remove_unicode(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)


def left_pad(indices, max_words=10, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res


def right_pad(indices, max_words=10, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[:max_words]
    res[:len(indices)] = indices
    return res
