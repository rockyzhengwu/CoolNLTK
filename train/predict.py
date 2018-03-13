#!/usr/bin/env python
# -*-coding:utf-8-*-




import tensorflow as tf
import numpy as np
import json


def load_vocab(vocab_path):
    f = open(vocab_path)
    vocab = json.loads(f.read())
    return vocab


def list_to_array(data_list, dtype=np.int32):
    array = np.array(data_list, dtype).reshape(1, len(data_list))
    return array


def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class Predictor(object):
    def __init__(self, model_file, char_to_id, id_to_tag):
        self.char_to_id = char_to_id
        self.id_to_tag = {int(k): v for k, v in id_to_tag.items()}
        self.graph = load_graph(model_file)

        self.input_x = self.graph.get_operation_by_name("prefix/input_x").outputs[0]
        self.dropout_keep_prob = self.graph.get_operation_by_name("prefix/dropout_keep_prob").outputs[0]
        self.predictions = self.graph.get_operation_by_name("prefix/output/predictions").outputs[0]
        self.scores = self.graph.get_operation_by_name("prefix/output/predprob").outputs[0]
        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id_to_tag)

    def predict(self, words):
        inputx = np.array([[vocab.get(w, vocab.get("<OOV>")) for w in words]])
        feed_dict = {
            self.input_x: inputx,
            self.dropout_keep_prob: 1.0,
        }
        predictions, prob = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        predict = [p + 1 for p in predictions]
        print(predict)



if __name__ == '__main__':
    model_file = "/home/wuzheng/GitHub/CoolNLTK/train/results/dbpedia/clstm/modle.pb"
    vocab_path = "/home/wuzheng/GitHub/CoolNLTK/train/datasets/dbpedia/vocab.json"
    vocab = load_vocab(vocab_path)
    id_to_tag = {v: k for k, v in vocab.items()}
    predictor = Predictor(model_file=model_file, char_to_id=vocab, id_to_tag=id_to_tag)

    text = "palaquium canaliculatum , palaquium canaliculatum is a species of plant in the sapotaceae family . it is endemic to sri lanka"
    words = text.split()
    predictor.predict(words)
