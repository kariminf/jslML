#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2023 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2023	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

class JslBERTPreprocessor:
    def __init__(self, ordinal_encoder, max_length):
        self.ordinal_encoder = ordinal_encoder
        self.max_length = max_length
        self.PAD = 0
        self.CLS = 2
        self.SEP = 3
        self.seg1_length = int(max_length * 0.75)

    def train_encode(self, word1, word2):
        length1 = len(word1)
        length2 = len(word2)
        if length1 + length2 + 3 > self.max_length:
            if (length1 + 2) > self.seg1_length:
                if (length2 + 1) <= (self.max_length - self.seg1_length):
                    length1 = self.max_length - length2 - 3
                else:
                    length1 = self.seg1_length - 2
                    length2 = self.max_length - self.seg1_length - 1
            else:
                length2 = self.max_length - length1 - 3
        length3 = self.max_length - (length1 + length2 + 3)
        # First word
        tok = [self.CLS] + self.ordinal_encoder(word1, length1) + [self.SEP]
        seg = [0] * (length1 + 2)
        pos = list(range(length1+1)) + [0]
        # Second word
        tok = tok + self.ordinal_encoder(word2, length2) + [self.SEP]
        seg = seg + [1] * (length2 + 1)
        pos = pos + list(range(1, length2+1)) + [0]
        # Padding
        if length3 > 0:
            tok = tok + [self.PAD] * length3
            seg = seg + [0] * length3
            pos = pos + [0] * length3
        return tok, pos, seg

    def predict_encode(self, word):
        length = len(word)
        if length + 2 > self.max_length:
            length = self.max_length - 2
        seg = [0] * (self.max_length)
        tok = [self.CLS] + self.ordinal_encoder(word, length) + [self.SEP]
        pos = list(range(length+2))
        if length + 2 < self.max_length: # Padding
            length = self.max_length - length - 2
            tok = tok + [self.PAD] * length
            pos = pos + [0] * length
        return tok, pos, seg


cls_code = 2
sep_code = 3

def generate_next(sents):
    result = []
    for sent in sents:
        for i, word in enumerate(sent):
            if i > 0:
                result.append((sent[i-1], word, 1))
            notpast = np.random.choice(len(sent), 1)[0]
            if notpast == i-1:
                notpast = i
            result.append((sent[notpast], word, 0))

    return result

def prepare_data_char_bert(df, char_encoder, word_length):
    X = []
    Y = []
    preprocess = JslBERTPreprocessor(char_encoder, word_length)
    for index, row in df.iterrows():
        word1 = row[0].replace("#@%", '"')
        word2 = row[1].replace("#@%", '"')
        rowcode, poscode, segcode = preprocess.train_encode(word1, word2)
        X.append([rowcode, poscode, segcode])
        Y.append(row[2])
    return X, Y
