#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/9 0:46
# @Author  : cxf
import tensorflow as tf
from tf_utils import get_shape_list, dropout


def multi_head_attention(query, key, value, num_heads, head_size, value_mask, dropout_prob):
    with tf.variable_scope("mulit_head_attetion"):
        query_embedding = tf.layers.dense(query, num_heads*head_size,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))  # 暂时不使用激活函数
        key_embedding = tf.layers.dense(key, num_heads*head_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        value_embedding = tf.layers.dense(value, num_heads*head_size,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        from_shape = get_shape_list(query)
        to_shape = get_shape_list(value)

        query_embedding = tf.transpose(tf.reshape(query_embedding, [-1, from_shape[1], num_heads, head_size]), [0, 2, 1, 3])
        key_embedding = tf.transpose(tf.reshape(key_embedding, [-1, to_shape[1], num_heads, head_size]), [0, 2, 1, 3])
        value_embedding = tf.transpose(tf.reshape(value_embedding, [-1, to_shape[1], num_heads, head_size]), [0, 2, 1, 3])

        attetion_scores = tf.matmul(query_embedding, key_embedding, transpose_b=True)
        attetion_scores = attetion_scores/head_size**0.5

        if value_mask is not None:
            # value_mask的shape为[batch_size, to_len]
            for _ in range(attetion_scores.shape.ndims - value_mask.shape.ndims):
                value_mask = tf.expand_dims(value_mask, axis=1)
            attetion_scores = value_mask*attetion_scores + (1-value_mask)*(-1e12)

        attetion_scores = tf.nn.softmax(attetion_scores)
        attetion_scores = dropout(attetion_scores, dropout_prob)
        output_tensor = tf.matmul(attetion_scores, value_embedding)
        output_tensor = tf.reshape(tf.transpose(output_tensor, [0, 2, 1, 3]), [-1, from_shape[1], num_heads*head_size])
        return output_tensor


def attention(from_tensor, to_tenor):
    """
    直接使用点积进行attention
    :param from_tensor:
    :param to_tenor:
    :return:
    """
    with tf.variable_scope("dot_attention"):
        attention_scores = tf.matmul(from_tensor, to_tenor, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores)
        output_tensor = tf.matmul(attention_scores, to_tenor)
        return output_tensor
