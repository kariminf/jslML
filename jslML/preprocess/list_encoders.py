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

def create_onehot_encoder(tag_list):
    def onehot_encoder(tag):
        result = [0] * len(tag_list)
        result[tag_list.index(tag)] = 1
        return result
    return onehot_encoder


def create_onehot_pref_encoder(size, pref_list):
    def onehot_encoder(word):
        word = word.lower()
        result = [0] * len(pref_list)
        if len(word) > size:
            idx = -1
            for i in range(size, 0, -1):
                pref = word[:i]
                try:
                    idx = pref_list.index(pref)
                    break
                except ValueError:
                    pass
            if idx > 0 :
                result[idx] = 1
        return result
    return onehot_encoder


def create_onehot_suff_encoder(size, suff_list):
    def onehot_encoder(word):
        word = word.lower()
        result = [0] * len(suff_list)
        if len(word) > size:
            idx = -1
            for i in range(size, 0, -1):
                suff = word[-i:]
                try:
                    idx = suff_list.index(suff)
                    break
                except ValueError:
                    pass
            if idx > 0 :
                result[idx] = 1
        return result
    return onehot_encoder