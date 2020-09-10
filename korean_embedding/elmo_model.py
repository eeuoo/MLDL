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
