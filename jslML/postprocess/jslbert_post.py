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

def save_block_embeding(f, weights, i_block, etype):
    heads_nbr = weights[1].shape[0]
    for h in range(heads_nbr):
        w = weights[0].numpy()[:, h,:].T.tolist()
        b = weights[0].numpy()[h,:].tolist()
        f.write("  block" + str(i_block) + "_h" + str(h) + "_" + str(etype) + "_w = " + str(w))
        f.write(",\n")
        f.write("  block" + str(i_block) + "_h" + str(h) + "_" + str(etype) + "_b = " + str(b))
        f.write(",\n")

def weights_to_js(brt, URL):
    f = open(URL, "w")

    f.write("  const \n")

    w  = brt.tokEmb.weights[0].numpy().T.tolist()
    b  = brt.tokEmb.weights[1].numpy().tolist()
    f.write("  tok_emb_w = " + str(w))
    f.write(",\n")
    f.write("  tok_emb_b = " + str(b))
    f.write(",\n")

    w  = brt.posEmb.weights[0].numpy().T.tolist()
    b  = brt.posEmb.weights[1].numpy().tolist()
    f.write("  pos_emb_w = " + str(w))
    f.write(",\n")
    f.write("  pos_emb_b = " + str(b))
    f.write(",\n")

    w  = brt.segEmb.weights[0].numpy().T.tolist()
    b  = brt.segEmb.weights[1].numpy().tolist()
    f.write("  seg_emb_w = " + str(seg_emb_w))
    f.write(",\n")
    f.write("  seg_emb_b = " + str(seg_emb_b))
    f.write(",\n")

    hps1, hps2, _ = brt.blocks[0].lma._output_dense.weights[0].numpy().shape
    hp_out_size = hps1 * hps2

    for i in range(len(brt.blocks)):
        block = brt.blocks
        save_block_embeding(f, brt.blocks[i].lma._query_dense.weights, i, "q")
        save_block_embeding(f, brt.blocks[i].lma._key_dense.weights, i, "k")
        save_block_embeding(f, brt.blocks[i].lma._value_dense.weights, i, "v")

        w  = block.lma._output_dense.weights[0].numpy().transpose(2, 0, 1).reshape([-1, hp_out_size]).tolist()
        b  = block.lma._output_dense.weights[1].numpy().T.tolist()
        f.write("  block" + str(i) + "_hp_w = " + str(w))
        f.write(",\n")
        f.write("  block" + str(i) + "_hp_b = " + str(b))
        f.write(",\n")

        w  = block.ffp.weights[0].numpy().T.tolist()
        b  = block.ffp.weights[1].numpy().T.tolist()
        f.write("  block" + str(i) + "_ffp_w = " + str(w))
        f.write(",\n")
        f.write("  block" + str(i) + "_ffp_b = " + str(b))
        f.write(",\n")

        ln1_gamma = block.norm1.weights[0].numpy().tolist()
        ln1_beta = block.norm1.weights[1].numpy().tolist()
        ln2_gamma = block.norm2.weights[0].numpy().tolist()
        ln2_beta = block.norm2.weights[1].numpy().tolist()

        f.write("  block" + str(i) + "_ln1_beta = " + str(ln1_beta))
        f.write(",\n")
        f.write("  block" + str(i) + "_ln1_gamma = " + str(ln1_gamma))
        f.write(",\n")
        f.write("  block" + str(i) + "_ln2_beta = " + str(ln2_beta))
        f.write(",\n")
        f.write("  block" + str(i) + "_ln2_gamma = " + str(ln2_gamma))

        if i < len(brt.blocks) - 1:
            f.write(",\n")
        else:
            f.write(";\n")
         