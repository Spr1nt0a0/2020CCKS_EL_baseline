#!/usr/bin/env python
# encoding: utf-8
'''
@author: Spr1nt
@contact: baochen.fly@gmail.com
@software: pycharm
@file: init.py
@time: 2020/7/22 22:45
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ast
import copy
import logging
import os

import numpy as np
import paddle.fluid as fluid
import six

log = logging.getLogger(__name__)


def cast_fp32_to_fp16(exe, main_program):
    """cast_fp32_to_fp16"""
    log.info("Cast parameters to float16 data format.")
    for param in main_program.global_block().all_parameters():
        if not param.name.endswith(".master"):
            param_t = fluid.global_scope().find_var(param.name).get_tensor()
            data = np.array(param_t)
            if param.name.startswith("encoder_layer") \
                    and "layer_norm" not in param.name:
                param_t.set(np.float16(data).view(np.uint16), exe.place)

            # load fp32
            master_param_var = fluid.global_scope().find_var(
                param.name +
                ".master",
            )
            if master_param_var is not None:
                master_param_var.get_tensor().set(data, exe.place)


def init_checkpoint(exe, init_checkpoint_path, main_program, use_fp16=False):
    """init_checkpoint"""
    assert os.path.exists(
        init_checkpoint_path,
    ), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        """existed_persitables"""
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables,
    )
    log.info("Load model from {}".format(init_checkpoint_path))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)


def init_pretraining_params(
    exe,
    pretraining_params_path,
    main_program,
    use_fp16=False,
):
    """init_pretraining_params"""
    assert os.path.exists(
        pretraining_params_path,
    ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        """existed_params"""
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params,
    )
    log.info("Load pretraining parameters from {}.".format(
        pretraining_params_path,
    ))

    if use_fp16:
        cast_fp32_to_fp16(exe, main_program)