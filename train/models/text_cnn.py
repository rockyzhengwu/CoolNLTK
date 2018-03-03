#!/usr/bin/env python
# -*-coding=utf-8-*-


import tensorflow as tf

class TextCNN(object):
    """
    cnn text cassfication
    """

    def __init__(self, config, embed):

        self.embedding_size = config["embedding_dim"]
        self.num_classes = config["num_classes"]
        self.filter_sizes = config["filter_sizes"]
        self.num_filters = config["num_filters"]
        self.vocab_size = config["vocab_size"]
        self.decay_steps = config["decay_steps"]
        self.decay_rate = config["decay_rate"]
        self.decay_rate_big = config["decay_rate_big"]
        self.clip_gradients = config["clip_gradients"]
        self.l2_lamba = config["l2_lambda"]
        self.optimizer = config["optimizer"]
        self.initializer = config["initializer"]
        self.sequence_length = config["sequence_length"]

        self.learning_rate = tf.Variable(config["learning_rate"], trainable = False, name = "learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * self.decay_rate_big)

        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        if not self.initializer:
            self.initializer = tf.random_normal_initializer(stddev = 0.1)

        self.input_x = tf.placeholder(tf.int32, [None, None], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, None], name = "input_y")

        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name = "dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name = "global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name = "epoch_Step")

        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.embedding = tf.get_variable(name="char_embeding", initializer=embed)

        # init weights
        self.init_weights()
        # forward
        self.forward()
        # accuracy
        self.accuracy = self.get_accuracy()
        # loss
        self.loss_val = self.loss(l2_lambda=self.l2_lamba)

        # back forward
        self.train_op = self.train()


    def init_weights(self):
        with tf.name_scope("embedding"):

            self.W_projection = tf.get_variable(name = "w_projection",
                                                shape = [self.num_filters_total, self.num_classes],
                                                initializer = tf.contrib.layers.xavier_initializer())

            self.b_projection = tf.get_variable(name = "b_projection", shape = [self.num_classes])


    def forward(self):
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.input_x)
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(self.filter_sizes):
            filter = tf.get_variable("filter-%s" % filter_size,
                                     [filter_size, self.embedding_size, 1, self.num_filters],
                                     initializer = self.initializer)

            conv = tf.nn.conv2d(self.sentence_embeddings_expanded,
                                filter,
                                strides = [1, 1, 1, 1],
                                padding = "VALID",
                                name = "conv")

            b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
            h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

            pooled = tf.nn.max_pool(h,
                                    ksize = [1, self.sequence_length - filter_size + 1, 1, 1],
                                    strides = [1, 1, 1, 1],
                                    padding = 'VALID',
                                    name = "pool")

            pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])


        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob = self.dropout_keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.nn.xw_plus_b(self.h_drop, self.W_projection, self.b_projection, name="logits")
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
            self.pred_prob = tf.nn.softmax(self.logits, name="predprob")


    def loss(self, l2_lambda = 0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
            return loss

    def get_accuracy(self):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.cast(tf.argmax(self.input_y, 1), tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
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
