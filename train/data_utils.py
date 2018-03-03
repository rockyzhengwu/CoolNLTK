#!/usr/bin/env python
#-*-coding:utf-8-*-

import math
import random
import json
import tensorflow as tf


def load_size_file(size_filename):
    with open(size_filename, 'r') as f:
        print(size_filename)
        num_obj = json.load(f)
        return num_obj


def load_vocab(vocab_filename):
    vocab = json.load(open(vocab_filename))
    id_to_word = {v:k for k, v in vocab.items()}
    return vocab, id_to_word


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            targets.append(target + padding)
        return [strings, targets]


    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)

        for idx in range(self.len_data):
            yield self.batch_data[idx]



class SegBatcher(object):
    def __init__(self, record_file_name, batch_size,  num_epochs=None):
        self._batch_size = batch_size
        self._epoch = 0
        self._step = 1.
        self.num_epochs = num_epochs
        print("SEG batcher :", record_file_name)
        self.next_batch_op = self.input_pipeline(record_file_name, self._batch_size, self.num_epochs)


    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        features = {
            'label': tf.FixedLenSequenceFeature([], tf.int64),
            'words': tf.FixedLenSequenceFeature([], tf.int64),
        }

        _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)
        label = example['label']
        words = example['words']
        return words, label

    def input_pipeline(self, filenames, batch_size,  num_epochs=None):
        filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=True)
        words, label = self.example_parser(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        print(capacity)

        next_batch = tf.train.batch([words, label], batch_size=batch_size, capacity=capacity,
                                    dynamic_pad=True, allow_smaller_final_batch=True)

        return next_batch

if __name__ == '__main__':
    batcher = SegBatcher("/home/wuzheng/dl/CoolNLTK/train/datasets/dbpedia/train.tfrecord", 32, 1)

    sess = tf.Session()
    tf.global_variables_initializer()
    threads = tf.train.start_queue_runners(sess=sess)

    tf.local_variables_initializer()
    batch = sess.run(batcher.next_batch_op)
    words, label = batch
    print(words.shape)
    print(label.shape)


