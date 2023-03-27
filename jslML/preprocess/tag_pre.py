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
from typing import List, Tuple


def generate_pos(sents: List[List[Tuple[str, str, str]]], wb:int=1, wa:int=1, tp:int=1):
    """Generate part of speach training/testing dataset.

    Args:
        sents (List[List[Tuple[str, str, str]]]): A list of sentences, each contains a list of words, 
        each is a tuple word/lemma/pos
        wb (int, optional): number of words before the current one. Defaults to 1.
        wa (int, optional): numer of words after the current one. Defaults to 1.
        tp (int, optional): number of tags before the current one. Defaults to 1.

    Returns:
        List[List[str]]: a list of words and tags (input/output)
    """

    result = []

    for sent in sents:
        past_words = ["<s>"] * wb
        past_tags  = ["<t>"] * tp
        next_words = []
        for i in range(wa):
            if i + 1 < len(sent):
                next_words.append(sent[i+1][0])
            else: 
                next_words.append("</s>")
        for i, word_info in enumerate(sent):
            e = past_words + next_words + past_tags + [word_info[0], word_info[2]]
            result.append(e)
            past_words = past_words[1:] + [word_info[0]]
            past_tags = past_tags[1:] + [word_info[2]]
            if i + wa + 1 < len(sent):
                next_words = next_words[1:] + [sent[i + wa + 1][0]]
            else:
                next_words = next_words[1:] + ["</s>"]

    return result


def encode_tag(tag, tag_list):
    return tag_list.index(tag)


def prepare_data_memm_tag(df, word_encoder, pos_encoder):
    X = []
    Y = []
    for index, row in df.iterrows():
        wordp = row[0].replace("#@%", '"')
        wordn = row[1].replace("#@%", '"')
        posp  = row[2]
        wordc = row[3].replace("#@%", '"')
        posc  = row[4]
        X.append(word_encoder.encode(wordp, wordn, posp, wordc))
        Y.append(pos_encoder.encode(posc))
    return X, Y
