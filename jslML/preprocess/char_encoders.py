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


eng_char_voc_size = 100
eng_char_start = 33
eng_char_end = 126

unknown_code = 1
padding_code = 0
char_shift = 5

def eng_char_encode(word, slice_length=None):
    code = []
    if slice_length:
        slice_length -= 1
    for i in range(len(word)):
        char_code = ord(word[i])
        if eng_char_start <= char_code <= eng_char_end:
            char_code = char_code - eng_char_start + char_shift
        else:
            char_code = unknown_code
        code.append(char_code)

        #better than slicing after completion
        if (slice_length) and i == slice_length:
            break
    return code