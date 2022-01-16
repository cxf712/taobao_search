#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/9 0:13
# @Author  : cxf
import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM(object):
    def __init__(self, hidden_unit, cell_type, num_layers, dropout_rate, use_last):
        self.hidden_unit = hidden_unit  # rnn hidden size
        self.cell_type = cell_type  # rnn cell type
        self.num_layers = num_layers  # num layers of rnn
        self.dropout = dropout_rate   # dropout rate in rnn
        self.use_last = use_last  # whether output the last

    def rnn_cell(self):
        cell_tmp = None
        if self.cell_type == "lstm":
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == "gru":
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def si_dir_rnn(self, input):
        """
        单向rnn
        :param input:
        :return:
        """
        cell_fw = self.rnn_cell()
        if self.dropout is not None:
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=1 - self.dropout)
        if self.num_layers > 1:
            cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell_fw, input, dtype=tf.float32)
        if self.use_last:
            return outputs[:, -1, :]
        else:
            return outputs  # [batch_size, text_len, hidden_size]

    def bi_dir_rnn(self, input):
        """
        双向rnn
        :param input:
        :return:
        """
        cell_fw = self.rnn_cell()
        cell_bw = self.rnn_cell()
        if self.dropout is not None:
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=1 - self.dropout)
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=1 - self.dropout)

        if self.num_layers > 1:
            cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
            cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)

        if self.use_last:
            return outputs[:, -1, :]
        else:
            return outputs  # [batch_size, text_len, hidden_size*2]


