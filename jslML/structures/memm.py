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

import tensorflow as tf 
import numpy as np
from tensorflow import keras
from keras.layers import Layer, LayerNormalization, Dense, MultiHeadAttention

class DefEmb:
    def predict(self, x):
        return x 

class TagEncoder:
    def __init__(self, tag_list, embedding=DefEmb()):
        self.tag_list = np.array(tag_list)
        self.embedding = embedding

    def encode(self, tag):
        return self.embedding.predict((self.tag_list == tag).astype(int))


def get_k_max(X, k):
    return sorted(X, key=lambda x: x[0], reverse=True)[:k]

class BeamMEMM(tf.keras.Model):
    def __init__(self, k:int, tg:TagEncoder):
        self.k = k
        self.maxent = Dense(len(tg.tag_list), activation="softmax", name="tags")
        self.tg = tg
        self.cls_names = tg.tag_list
    
    def train_step(self, data):
        X, Y = data
        M = tf.cast(tf.shape(Y)[0], tf.float32)

        with tf.GradientTape() as tape:
            logits = self.maxent(X)
            loss = tf.reduce_sum(self.loss(logits, Y))/M
        
        variables = self.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
          
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
    def init(self, x):
        self.BV = []
        newx = self.tg.encode("<t>").concat(x);
        p = np.log(self.maxent.predict(newx)[0])
        past_i = [-1] * len(p)
        # k_max = sorted(Z, key=lambda x: x[0], reverse=True)
        self.BV.append(get_k_max(zip(self.cls_names, past_i, p), self.k))

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
    def step(self, x):
        past = self.BV[len(self.BV)-1]
        choices = []

        for i, e in enumerate(past):
            past_tag = self.tg.encode(e[0])
            newx = np.concatenate(self.tg.encode(past_tag), x)
            p = np.log(self.maxent.predict(newx)[0])
            past_i = [i] * len(p)
            choices += zip(self.cls_names, past_i, p)

        self.BV.append(get_k_max(choices, self.k))

    @tf.function
    def final(self):
        i = len(self.BV) - 1 
        result = []
        j = 0 
        while i > 0:
            next_e = self.BV[i][j]
            result = [next_e[0]] + result
            j = next_e[1]
            i -= 1
        return result