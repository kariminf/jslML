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

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jslbert import JslBERT



blocks_nbr = 2
d_model    = 10
heads_nbr  = 3
vocab_size = 96
max_length = 30
brt = JslBERT(blocks_nbr, d_model, heads_nbr, vocab_size, max_length)

brt.compile()

X = [[[5, 2, 1, 5] + [0] * 26, [1] * 30, [0] * 30 ]]

Y = [[1]]

brt.fit(X, Y, epochs=1)

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