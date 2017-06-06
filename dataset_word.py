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
            input_filename = "<from memory>"

        if params.get('tokenize'):
            text = remove_unicode(text)
            text = tokenize_text(text)
        
        if params.get('lowercase'):
            text = text.lower()

        self.is_encoder = is_encoder
        if self.is_encoder:
            self.max_words = params['max_words_encoder']
            vocab_filename = params['load_encoder_vocab']
        else:
            self.max_words = params['max_words_decoder']
            vocab_filename = params['load_decoder_vocab']
        self.sentences = text.splitlines()

        print("Input file {} contains {} sentences".format(input_filename, len(self.sentences)))

        if vocab_filename:
            print("Loading vocabulary from file...")
            self.vocab, self.word_to_idx, self.idx_to_word = load_vocab_from_file(vocab_filename)
        else:
            print("Building vocabulary...")
            rarity = params['vocab_rarity']
            self.vocab, self.word_to_idx, self.idx_to_word = build_vocab(text, n=rarity)
        print("Vocabulary contains {} words from {} to {}".format(len(self.vocab), self.vocab[4], self.vocab[-1]))
        if self.is_encoder:
            open('last_used_encoder_vocab.txt', 'w').write('\n'.join(self.vocab))
        else:
            open('last_used_decoder_vocab.txt', 'w').write('\n'.join(self.vocab))

    def random_idx(self):
        return np.random.randint(0, len(self.sentences))

    def get_example(self, idx=None, **params):
        if idx is None:
            idx = self.random_idx()
        sentence = self.sentences[idx]
        return self.format_input(sentence, **params)
    
    def count(self):
        return len(self.sentences)

    def format_input(self, sentence, **params):
        if self.is_encoder:
            indices = self.indices(sentence)
            return [left_pad(indices[:self.max_words], self.max_words, **params)]
        else:
            num_classes = len(self.vocab)
            indices = self.indices(sentence + ' <eos>')
            return [to_categorical(right_pad(indices, self.max_words, **params), num_classes)]

    def unformat_input(self, X, **params):
        indices = X[0]
        return ' '.join(self.words(indices))

    def unformat_output(self, preds, **params):
        indices = np.argmax(preds, axis=-1)
        return ' '.join(self.words(indices))

    def empty_batch(self, **params):
        batch_size = params['batch_size']
        if self.is_encoder:
            return [np.zeros((batch_size, self.max_words), dtype=int)]
        else:
            return [np.zeros((batch_size, self.max_words, len(self.vocab)), dtype=float)]

    def indices(self, words):
        # TODO: Properly tokenize?
        unk = self.word_to_idx[UNK_TOKEN]
        return [self.word_to_idx.get(w, unk) for w in words.split()]

    def words(self, indices):
        # TODO: Properly detokenize and join
        return [self.idx_to_word.get(i) for i in indices]

    def build_encoder(self, **params):
        return model_words.build_encoder(self, **params)

    def build_decoder(self, **params):
        return model_words.build_decoder(self, **params)

    def build_discriminator(self, **params):
        return model_words.build_discriminator(self, **params)


def load_vocab_from_file(filename):
    text = open(filename).read()
    return build_vocab(text, n=0)


def build_vocab(text, n=2):
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


def left_pad(indices, max_words=12, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res


def right_pad(indices, max_words=12, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[:max_words]
    res[:len(indices)] = indices
    return res
