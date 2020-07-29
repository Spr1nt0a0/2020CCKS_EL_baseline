#!/usr/bin/env python
# encoding: utf-8
'''
@author: Spr1nt
@contact: baochen.fly@gmail.com
@software: pycharm
@file: cards.py
@time: 2020/7/22 22:43
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


def get_cards():
    """
    get gpu cards number
    """
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num