#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 11:57
# @Author  : cxf
import tensorflow as tf
from lstm import LSTM
from tf_utils import get_shape_list, dropout
from attention import multi_head_attention, attention


class MgdspModel(object):
    def __init__(self, unigram_max_len, bigram_max_len, word_max_len, unigram_size, bigram_size, word_size,
                 embedding_dim, real_term_max_len, short_term_max_len, long_term_max_len, history_features_embedding,
                 history_features_size, history_feature_nums, rnn_hidden_unite, rnn_cell_type, rnn_num_layers,
                 num_heads, head_size, negtive_nums, hard_nums, temperature, hard_ratio, use_hard):
        """

        :param unigram_max_len: query的unigram序列的长度
        :param bigram_max_len: query的bigram序列的长度
        :param word_max_len: query的词序列长度
        :param unigram_size: 整个unigram表的size
        :param bigram_size: 整个bigram表的size
        :param word_size:  整个词表的size
        :param embedding_dim: 各种输入的embedding维度
        :param real_term_max_len: 实时行为序列的最大长度
        :param short_term_max_len: 短时行为序列的最大长度
        :param long_term_max_len:  长时行为序列的最大长度
        :param history_features_embedding: 历史行为序列的特征的embedding维度
        :param history_features_size: 历史行为序列的特征表的size(包含itemid，brand，leaf category,shop)
        :param history_feature_nums: 历史行为序列使用的特征数目
        :param rnn_hidden_unite: rnn的隐藏层unite数目
        :param rnn_cell_type: rnn的cell type
        :param rnn_num_layers: rnn的层数
        :param num_heads: multi-head attention的head的数目
        :param head_size: multi-head attention每个head的embedding维度
        :param negtive_nums: 负样本的数目
        :param hard_nums:  hard负样本的数目
        :param temperature: softmax的温度系数
        :param hard_ratio: hard negtivate 插值的权重
        :param use_hard: 是否使用hard negtivate
        """
        self.unigram_max_len = unigram_max_len
        self.bigram_max_len = bigram_max_len
        self.word_max_len = word_max_len
        self.unigram_size = unigram_size
        self.bigram_size = bigram_size
        self.word_size = word_size
        self.embedding_dim = embedding_dim
        self.real_term_max_len = short_term_max_len
        self.short_term_max_len = real_term_max_len
        self.long_term_max_len = long_term_max_len
        self.history_feature_nums = history_feature_nums
        self.history_features_embedding = history_features_embedding
        self.history_features_size = history_features_size
        self.rnn_hidden_unite = rnn_hidden_unite
        self.rnn_cell_type = rnn_cell_type
        self.rnn_num_layers = rnn_num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.negtive_nums = negtive_nums
        self.use_hard = use_hard
        self.hard_nums = hard_nums
        self.temperature = temperature
        self.hard_ratio = hard_ratio

    def build_model(self):
        query_unigram_input_ids = tf.placeholder(shape=[None, self.unigram_max_len], dtype=tf.int32,
                                                 name="query_unigram_input_ids")
        query_bigram_input_ids = tf.placeholder(shape=[None, self.bigram_max_len], dtype=tf.int32,
                                                name="query_bigram_input_ids")
        query_word_input_ids = tf.placeholder(shape=[None, self.word_max_len], dtype=tf.int32,
                                              name="query_word_input_ids")
        real_term_input = tf.placeholder(shape=[None, self.real_term_max_len, self.history_feature_nums], dtype=tf.int32,
                                         name="real_term_input")
        short_term_input = tf.placeholder(shape=[None, self.short_term_max_len, self.history_feature_nums], dtype=tf.int32,
                                          name="short_term_input")
        long_term_input = tf.placeholder(shape=[None, self.long_term_max_len, self.history_feature_nums], dtype=tf.int32,
                                          name="long_term_input")
        item_id_input = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="item_id")
        item_title_input = tf.placeholder(shape=[None, self.word_max_len], dtype=tf.int32, name="item_title")
        batch_negtive_id = tf.placeholder(shape=[self.negtive_nums, 1], dtype=tf.int32, name="batch_negtive_id")
        batch_negtive_title = tf.placeholder(shape=[self.negtive_nums, self.word_max_len], dtype=tf.int32,
                                             name="batch_negtive_title")
        # historical query目前不清楚使用的范围，暂时不考虑计算
        with tf.variable_scope("query"):
            with tf.variable_scope("embedding"):
                query_unigram_embedding = tf.reduce_mean(embedding_lookup(query_unigram_input_ids, self.unigram_size, self.embedding_dim, "unigram_embedding"),
                                                         axis=1)
                query_bigram_embedding = tf.reduce_mean(embedding_lookup(query_bigram_input_ids, self.bigram_size,
                                                                         self.embedding_dim,"unigram_embedding"),
                                                        axis=1)
                query_word_embedding = tf.reduce_mean(embedding_lookup(query_word_input_ids, self.word_size,
                                                                       self.embedding_dim,"word_embedding"), axis=1)
                query_word_transformer_encode = tf.reduce_mean(transformer_encode(query_word_input_ids), axis=1)

                query_embedding_mix = query_unigram_embedding + query_bigram_embedding + query_word_embedding +\
                                      query_word_transformer_encode
                query_embedding = tf.concat([query_unigram_embedding, query_bigram_embedding, query_word_embedding,
                                             query_word_transformer_encode, query_embedding_mix])
            with tf.variable_scope("user_behavior"):
                behavior_embedding_table = tf.get_variable(shape=[self.history_features_size,
                                                                  self.history_features_embedding],
                                                           name="behavior_embedding_table",
                                                           initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                           trainable=True)
                real_term_embedding = self.personalized_short_term_embedding("real_term", real_term_input,
                                                                             query_embedding, behavior_embedding_table,
                                                                             self.real_term_max_len, True)
                short_term_embedding = self.personalized_short_term_embedding("short_term", short_term_input,
                                                                              behavior_embedding_table, query_embedding,
                                                                              self.short_term_max_len, False)
                long_term_embedding = self.personalized_long_term_embedding("logn_term", long_term_input, query_embedding,
                                                                            behavior_embedding_table, self.long_term_max_len)
                first_token = tf.get_variable(name="cls", shape=[1, self.embedding_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=True)
                fusion_user_embedding = tf.concat([first_token, query_embedding, real_term_embedding, short_term_embedding,
                                                   long_term_embedding], axis=1)
                fusion_user_embedding = multi_head_attention(fusion_user_embedding, fusion_user_embedding,
                                                             fusion_user_embedding, 1, self.embedding_dim, None, 0)
                fusion_user_embedding = fusion_user_embedding[:, 0, :]
            with tf.variable_scope("positive"):
                item_embedding = self.get_item_embedding(item_id_input, item_title_input, behavior_embedding_table)
            with tf.variable_scope("negtive_smaples"):
                batch_negtive_embedding = self.get_item_embedding(batch_negtive_id, batch_negtive_title,
                                                                  behavior_embedding_table)
            with tf.variable_scope("loss"):
                pos_score = tf.reduce_sum(tf.multiply(fusion_user_embedding, item_embedding), axis=-1)
                neg_score = tf.matmul(fusion_user_embedding, batch_negtive_embedding, transpose_b=True)
                if self.use_hard:
                    # 选取neg_score 得分的topk作为每个query的hard_negtivate embedding
                    topk_neg_score, topk_neg_indice = tf.nn.top_k(neg_score, self.hard_nums)  # [batch_size, hard_nums]
                    flat_topk_indice = tf.reshape(topk_neg_indice, [-1])
                    topk_embedding = tf.gather(batch_negtive_embedding, flat_topk_indice)
                    topk_embedding = tf.reshape(topk_embedding, [-1, self.hard_nums, self.embedding_dim])
                    # 使用插值方式得到topk hard negtivate的embedding
                    topk_embedding = self.hard_ratio*tf.expand_dims(fusion_user_embedding, axis=1)+(1-self.hard_ratio)*topk_embedding
                    hard_neg_score = tf.matmul(tf.expand_dims(fusion_user_embedding, axis=1), topk_embedding, transpose_b=True)
                    hard_neg_score = tf.squeeze(hard_neg_score, axis=1)
                    score = tf.concat([tf.expand_dims(pos_score, axis=1), neg_score, hard_neg_score], axis=1)
                else:
                    score = tf.concat([tf.expand_dims(pos_score, axis=1), neg_score], axis=1)
                log_logits = tf.nn.log_softmax(score/self.temperature)
                label = tf.zeros(shape=[get_shape_list(item_embedding)[0], 1], dtype=tf.int32)
                label_one_hot = tf.one_hot(label, self.negtive_nums+1)
                total_loss = -tf.reduce_sum(label_one_hot * log_logits, axis=-1)
                loss = tf.reduce_mean(total_loss)
        return loss

    def get_item_embedding(self, item_id_input, item_title_input, behavior_embedding_table):
        """

        :param item_id_input
        :param item_title_input
        :param behavior_embedding_table
        :return:
        """
        with tf.variable_scope("item"):
            item_id_embedding = tf.nn.embedding_lookup(behavior_embedding_table, item_id_input)
            item_title_embedding = tf.nn.embedding_lookup(behavior_embedding_table, item_title_input)
            item_id_embedding = tf.squeeze(item_id_embedding, axis=1)
            item_embedding = item_id_embedding + tf.layers.dense(tf.reduce_mean(item_title_embedding),
                                                                 self.embedding_dim, use_bias=False,
                                                                 activation=tf.nn.tanh,
                                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                                     stddev=0.02))
        return item_embedding

    def personalized_short_term_embedding(self, name, term_input, query_embedding, behavior_embedding_table,
                                          term_max_len, use_lstm):
        """
        real-time 和short-time的行为序列建模
        :param name:
        :param term_input: 为历史行为序列，shape=[batch_size,term_max_len,num_features]
        :param query_embedding: 用户输入的query embedding
        :param behavior_embedding_table: itemId和side information 合起来的embedding table
        :param term_max_len: 序列的最大长度
        :param use_lstm: 是否使用lstm对行为序列进行建模
        :return:
        """
        with tf.variable_scope(name):
            input_shape = get_shape_list(term_input)
            term_embedding = tf.nn.embedding_lookup(behavior_embedding_table,
                                                    tf.reshape(term_input, [-1, term_max_len*input_shape[2]]))
            # cocnat itemId embedding and side information embedding
            term_embedding = tf.reshape(term_embedding,
                                        [-1, term_max_len, input_shape[2]*self.history_features_embedding])
            if use_lstm:
                lstm_model = LSTM(self.rnn_hidden_unite, self.rnn_cell_type, self.rnn_num_layers, 0.2, True)
                # use lstm to capture evolution
                term_embedding = lstm_model.si_dir_rnn(term_embedding)
            # use multi-head self-attention to aggregate potential points of interst
            term_embedding = multi_head_attention(term_embedding, term_embedding, term_embedding, self.num_heads,
                                                  self.head_size, None, 0)
            zeros = tf.zeros(shape=[input_shape[0], 1, get_shape_list(term_embedding)[-1]])
            term_embedding = tf.concat([zeros, term_embedding], axis=1)
            personalized_embedding = attention(query_embedding, term_embedding)
            return personalized_embedding

    def personalized_long_term_embedding(self, name, term_input, query_embedding, behavior_embedding_table, term_max_len):
        """
        long-time的行为序列建模
        :param name:
        :param term_input: 为历史行为序列，shape=[batch_size,type, term_max_len,num_features],type包含点击、收藏、购买
        :param query_embedding: 用户输入的query embedding
        :param behavior_embedding_table: itemId和side information 合起来的embedding table
        :param term_max_len: 序列的最大长度
        :return:
        """
        with tf.variable_scope(name):
            input_shape = get_shape_list(term_input)
            term_embedding = tf.nn.embedding_lookup(behavior_embedding_table,
                                                    tf.reshape(term_input, [-1, term_max_len*self.history_feature_nums]))
            term_embedding = tf.reshape(term_embedding,
                                        [-1, input_shape[1], term_max_len, self.history_feature_nums, self.history_features_embedding])
            term_embedding = tf.reduce_mean(term_embedding, axis=2)
            item_embedding, shop_embedding, leaf_embedding, brand_embedding = tf.split(term_embedding, 4)

            def get_attr_embedding(attr_embedding):
                zeros = tf.zeros(shape=[1, self.history_features_embedding])
                attr_embedding = tf.concat([zeros, attr_embedding], axis=1)
                attr_embedding = attention(query_embedding, attr_embedding)
                return attr_embedding
            item_embedding = get_attr_embedding(item_embedding)
            shop_embedding = get_attr_embedding(shop_embedding)
            leaf_embedding = get_attr_embedding(leaf_embedding)
            brand_embedding = get_attr_embedding(brand_embedding)
            personalized_embedding = item_embedding + shop_embedding + leaf_embedding + brand_embedding

            return personalized_embedding


def embedding_lookup(input, max_size, embedding_dim, name):
    """
    将input 进行embedding
    :param input:
    :param max_size:
    :param embedding_dim:
    :param name:
    :return:
    """
    embedding_table = tf.get_variable(name=name, shape=[max_size, embedding_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02), trainable=True)
    output = tf.nn.embedding_lookup(embedding_table, input)

    return output


def transformer_encode(input_tensor, num_layers, num_heads, head_size, large_hidden_size, value_mask, dropout_prob,
                       return_all_layers):
    """
    使用transformer 对输入进行encode
    :param input_tensor shape = [batch_size, sen_len, dim]
    :param num_layers
    :param num_heads
    :param head_size
    :param large_hidden_size
    :param value_mask
    :param dropout_prob
    :param return_all_layers
    :return:
    """
    all_layers_output = []
    prev_output = input_tensor
    for i in range(num_layers):
        input_layer = prev_output
        attention_output = multi_head_attention(input_layer, input_tensor, input_tensor, num_heads, head_size, value_mask,
                                                dropout_prob)
        with tf.variable_scope("add_norm"):
            attention_output = tf.layers.dense(attention_output, num_heads*head_size,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            attention_output = dropout(attention_output, dropout_prob)
            attention_output = layer_norm(attention_output+input_layer)

        with tf.variable_scope("feed_forward"):
            output = tf.layers.dense(attention_output, large_hidden_size,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            output = tf.layers.dense(output, num_heads*head_size,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            output = dropout(output, dropout_prob)
            output = layer_norm(output+attention_output)
        all_layers_output.append(output)
        prev_output = output
    if return_all_layers:
        return all_layers_output
    else:
        return prev_output


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(input_tensor, begin_norm_axis=-1, begin_params_axis=-1, name=name)
