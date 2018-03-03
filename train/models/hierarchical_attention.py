#!/usr/bin/env python
#-*-coding:utf-8-*-



import numpy as np
import tensorflow as tf
from tensorflow.contrib  import rnn


class HierarchicalAttention:

    def __init__(self, config):
        self.embedding_dim = config.embedding_dim
        self.sequence_length = config.sequence_length
        self.num_hidden = config.num_hidden
        self.num_classes = config.num_classes
        self.vocab_sizes = config.vocab_size
        self.l2_lambda = config.l2_lambda
        self.optimizer = config.optimizer
        self.clip_gradients = config.clip_gradients
        self.learning_rate = config.learning_rate
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate

        self.word_attention_size = config.word_attention_size


        self.input_x = tf.placeholder(dtype = tf.int32, shape = [None, self.sequence_length], name = "input_x")
        self.input_y = tf.placeholder(dtype = tf.int32, shape = [None, self.num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")


        self.global_step = tf.Variable(0, trainable=False, name = "global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name = "epoch_Step")

        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))


        self.initializer = tf.random_normal_initializer(stddev = 0.1)

        self.init_weight()

        self.forward()

        self.accuracy = self.get_accuracy()

        self.loss_val  = self.loss()
        self.train_op = self.train()



    def init_weight(self):
         self.embedding = tf.get_variable(name = "embedding",
                                          shape = [self.vocab_sizes, self.embedding_dim],
                                          initializer = self.initializer)

         self.w_projection = tf.get_variable(name = "w_projection",
                                             shape = [2 * self.num_hidden, self.num_classes],
                                             initializer = tf.contrib.layers.xavier_initializer())

         self.b_projection = tf.get_variable(name = "b_projection", shape = [self.num_classes])



    def word_attention(self, inputs):

        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        attention_size = self.word_attention_size

        W_omega = tf.Variable(tf.random_normal([self.num_hidden, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output, alphas


    def forward(self):
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.input_x)

        rnn_fw_cell = rnn.GRUCell(self.num_hidden)
        rnn_bw_cell = rnn.GRUCell(self.num_hidden)
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, self.embedded_words, dtype = tf.float32)

        attention_output, alphas = self.word_attention(rnn_outputs)
        print(attention_output)
        attention_drop = tf.nn.dropout(attention_output, self.dropout_keep_prob)

        print(attention_drop)
        exit(0)
        # output_rnn = tf.concat(attention_output, axis = 2)
        # self.output_rnn_last = tf.reduce_mean(output_rnn, axis = 1)

        with tf.name_scope("output"):
            self.logits = tf.nn.xw_plus_b(self.output_rnn_last, self.w_projection, self.b_projection, name = "logits")
            self.predictions = tf.argmax(self.logits, axis = 1, name = "predictions")
            self.pred_prob = tf.nn.softmax(self.logits, name = "predprob")


    def loss(self):
        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.logits);

            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
            return loss

    def get_accuracy(self):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.cast(tf.argmax(self.input_y, 1), tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")
            return accuracy

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   self.decay_steps,
                                                   self.decay_rate,
                                                   staircase = True)

        train_op = tf.contrib.layers.optimize_loss(self.loss_val,
                                                   global_step = self.global_step,
                                                   learning_rate = learning_rate,
                                                   optimizer = self.optimizer,
                                                   clip_gradients = self.clip_gradients)
        return train_op


