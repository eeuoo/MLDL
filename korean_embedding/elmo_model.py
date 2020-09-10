import glob
import random
import numpy as np
from typing import List

class Vocabulary(object) :
    def __init__(self, filename, validate_file=False):
        self._id_to_word = []
        self._word_to_id = []
        self._unk = -1
        self._bos = -1
        self._eos = -1

        with open(filename) as f :
            idx = 0
            for line in f :
                word_name = line.strip()
                if word_name == '<S>' :
                    self._bos = idx
                elif word_name == '</S>' :
                    self._eos = idx
                elif word_name == '<UNK>' :
                    self._unk = idx
                if word_name == '!!!MAXTERMID' :
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file :
            if self._bos == -1 or self._eos == -1 or self._unk == -1 :
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        if split:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)

