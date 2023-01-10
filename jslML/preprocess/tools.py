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
    for index, row in df.iterrows():
        word1code = char_encoder(row[0].replace("#@%", '"'), max_length=word_length)
        word2code = char_encoder(row[1].replace("#@%", '"'), max_length=word_length)
        rowcode = [cls_code] + word1code + [sep_code] + word2code
        poscode = list(range(15)) + list(range(15))
        segcode = ([0] * 15) + ([1] * 15)
        X.append([rowcode, poscode, segcode])
        Y.append(row[2])
    return X, Y