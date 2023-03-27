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
from keras.layers import Dense

class SimpleAE(tf.keras.Model):
    def __init__(self, k:int, n:int):
        super(self.__class__, self).__init__()
        self.enc = Dense(k, name="enc")
        self.dec = Dense(n, name="dec", activation='sigmoid')
        self.loss = tf.keras.losses.CategoricalCrossentropy()
    
    def train_step(self, X):
        X = tf.cast(X, tf.float32)
        with tf.GradientTape() as tape:
            M = tf.cast(tf.shape(X)[0], tf.float32)
            logits = self.dec(self.enc(X))
            loss = self.loss(logits, X)
            loss = tf.reduce_sum(loss)/M
        
        variables = self.trainable_variables 
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
          
        return {"loss": loss}

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def encode(self, X):
        return self.enc(X)
