#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/15 20:52
# @Author  : cxf
import tensorflow as tf


def get_shape_list(input_tensor):
    """
    获得tensor各维度的值，用作reshape等操作
    需要区分开动态shape和静态shape
    :param input_tensor:
    :return:
    """
    shape = input_tensor.shape.as_list()
    dynamic_index = []
    for (index, ndim) in enumerate(shape):
        if ndim is None:
            dynamic_index.append(index)
    dynamic_shape = tf.shape(input_tensor)
    for index in dynamic_index:
        shape[index] = dynamic_shape[index]
    return shape


def dropout(input_tensor, dropout_prob):
    """
    :param input_tensor
    :param dropout_prob:
    :return:
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    return tf.nn.dropout(input_tensor, keep_prob=1-dropout_prob)
