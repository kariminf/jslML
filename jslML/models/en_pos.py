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

from jslML.structures.jslbert import JslBERT
from jslML.preprocess.list_encoders import create_onehot_encoder, create_onehot_pref_encoder, create_onehot_suff_encoder
import re 

tag_list = [
    "<s>", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", 
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
    ]

prefixes = [
    "ambi", "anti", "astro", "bi", "co", "con", "de", "dis", "em", "extra", "fore" "hetero"
    "hind", "homo", "im", "in", "inter", "mal", "mid", "mis", "mono", "non", "on", "pan",
    "ped", "post", "pre", "pro", "re", "semi", "sub", "sur", "trans", "tri", "twi", "ultra",
    "un", "uni", "under", "up"
]

suffixes = [
    "able", "ac", "ize", "age", "al", "an", "ant", "ary", "cracy", "cycle", "dom", "eer", "en",
    "er", "ess", "est", "ette", "ful", "hood", "ible", "ic", "ify", "ion", "ish", "ism", "ity",
    "less", "like", "log", "ment", "ness", "or", "ous", "ship", "th", "ure", "ward", "wise", "y"
]

encode_prefix = create_onehot_pref_encoder(5, prefixes)
encode_suffix = create_onehot_suff_encoder(5, suffixes)
encode_tag    = create_onehot_encoder(tag_list)

RE_D = re.compile(r'\d')
RE_H = re.compile(r'-')
RE_U = re.compile(r'[A-Z]')
RE_P = re.compile(r'\.')

def encode_surface(word):
    result = []
    #contains a number
    result.append(int(bool(RE_D.search(word))))
    #contains a hyphen
    result.append(int(bool(RE_H.search(word))))
    #contains an uppercase
    result.append(int(bool(RE_U.search(word))))
    #contains a point (dot)
    result.append(int(bool(RE_P.search(word))))
    return result

def eng_word_srf_encoder(word):
    return encode_prefix(word) + encode_suffix(word) + encode_surface(word)

def new_eng_char_bert(blocks_nbr=1, d_model=10, heads_nbr=2, vocab_size = 100, max_length = 30, d_mha=5, masked=True):
    brt = JslBERT(blocks_nbr, d_model, heads_nbr, vocab_size, max_length, d_mha, masked)
    brt.compile(run_eagerly=True)
    return brt



def train_eng_pos(brt, X, Y, epochs=100, batch_size=4000):
    brt.fit(X, Y, epochs=epochs, batch_size=batch_size)

    print("tokEmb")
    print(brt.tokEmb.weights)

    print("posEmb")
    print(brt.posEmb.weights)

    print("segEmb")
    print(brt.segEmb.weights)

    for block in brt.blocks:
        attention_weights = block.lma.get_weights()
        print("Query")
        print(block.lma._query_dense.weights)
        print("Keys")
        print(block.lma._key_dense.weights)
        print("Values")
        print(block.lma._value_dense.weights)
        print("Proj")
        print(block.lma._output_dense.weights)
        print("FFP")
        print(block.ffp.weights)


# weight_names = ['query', 'keys',  'values', 'proj']
# for name, out in zip(weight_names,layer.get_weights()):
#     print(name, out.shape)
# query (5, 2, 4) # (embed_dim, num_heads, key_dim)
# keys (5, 2, 4)  # (embed_dim, num_heads, key_dim)
# values (5, 2, 4) # (embed_dim, num_heads, value_dim/key_dim)
# proj (2, 4, 5)  # (num_heads, key_dim, embed_dim)