import re
import random
import numpy as np
from tokenizer import tokenize_text
from keras.utils import to_categorical
import model_words


PAD_TOKEN = ' '
UNK_TOKEN = '<unk>'
START_TOKEN = '<bos>'
END_TOKEN = '<eos>'

class WordDataset(object):
    def __init__(self, input_filename=None, input_text=None, is_encoder=False, **params):
        if not input_filename and not input_text:
            raise ValueError("No input filename supplied. See options with --help")
        if input_filename:
            text = open(input_filename).read()
        else:
            text = input_text

        if params.get('tokenize'):
            text = remove_unicode(text)
            text = tokenize_text(text)
        
        if params.get('lowercase'):
            text = text.lower()

        self.is_encoder = is_encoder
        self.sentences = text.splitlines()
        print("Input file {} contains {} sentences".format(input_filename, len(self.sentences)))
        self.vocab, self.word_to_idx, self.idx_to_word = get_vocab(text)
        print("Vocabulary contains {} words from {} to {}".format(len(self.vocab), self.vocab[4], self.vocab[-1]))

    def random_idx(self):
        return np.random.randint(0, len(self.sentences))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        sentence = self.sentences[idx]
        return self.format_input(sentence, **params)

    def format_input(self, sentence, **params):
        max_words = params['max_words']
        if self.is_encoder:
            indices = self.indices(sentence)
            return [left_pad(indices[:max_words], **params)]
        else:
            num_classes = len(self.vocab)
            indices = self.indices(sentence + ' <eos>')
            return [to_categorical(right_pad(indices, **params), num_classes)]

    def unformat_input(self, indices, **params):
        return ' '.join(self.words(indices))

    def unformat_output(self, preds, **params):
        indices = np.argmax(preds, axis=-1)
        return ' '.join(self.words(indices))

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        max_words = params['max_words']
        if self.is_encoder:
            return [np.zeros((batch_size, max_words), dtype=int)]
        else:
            return [np.zeros((batch_size, max_words, len(self.vocab)), dtype=float)]

    def indices(self, words):
        # TODO: Properly tokenize?
        unk = self.word_to_idx[UNK_TOKEN]
        return [self.word_to_idx.get(w, unk) for w in words.split()]

    def words(self, indices):
        # TODO: Properly detokenize and join
        return [self.idx_to_word.get(i) for i in indices]

    def build_model(self, **params):
        if self.is_encoder:
            return model_words.build_encoder(self, **params)
        else:
            return model_words.build_decoder(self, **params)


def get_vocab(text, n=3):
    word_count = {}
    for word in text.split():
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
    words = [w for w in text.split() if word_count[w] > n]
    vocab = sorted(list(set(words + [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN])))

    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(vocab):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return vocab, word_to_idx, idx_to_word


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
