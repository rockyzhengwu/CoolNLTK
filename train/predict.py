#!/usr/bin/env python
# -*-coding:utf-8-*-




import tensorflow as tf
import numpy as np
import json


def load_vocab(vocab_path):
    f = open(vocab_path)
    vocab = json.loads(f.read())
    return vocab



def load_size(size_file):
    f = open(size_file)
    size_map = json.loads(f.read())
    return size_map


def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def list_to_array(data_list, dtype=np.int32):
    array = np.array(data_list, dtype).reshape(1, len(data_list))
    return array


class Predictor(object):
    # cnn  fix_length 需要需要等于True


    def __init__(self, model_file, char_to_id, fix_length=False):
        self.char_to_id = char_to_id
        self.fix_length  = fix_length

        self.graph = load_graph(model_file)

        self.input_x = self.graph.get_operation_by_name("prefix/input_x").outputs[0]
        self.dropout_keep_prob = self.graph.get_operation_by_name("prefix/dropout_keep_prob").outputs[0]
        self.predictions = self.graph.get_operation_by_name("prefix/output/predictions").outputs[0]
        self.scores = self.graph.get_operation_by_name("prefix/output/predprob").outputs[0]
        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()


    def predict(self, words):

        inputx = [[vocab.get(w, vocab.get("<OOV>")) for w in words]]
        if self.fix_length:
            # cnn padding
            new_inputx = []
            for x in inputx:
                if len(x)>100:
                    x = x[100]
                else:
                    x = x + [0]*(100 - len(x))
                new_inputx.append(x)
            inputx = np.array(new_inputx)
        else:
            inputx = np.array(inputx)
        feed_dict = {
            self.input_x: inputx,
            self.dropout_keep_prob: 1.0,
        }
        predictions, prob = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        predict = [p + 1 for p in predictions]
        print(predict)



if __name__ == '__main__':
    #######
    # cnn,clstm 是一定要固定句子长度的，bilstm可以随意, 但不能为空
    #####

    model_file = "./results/dbpedia/bilstm/modle.pb"
    vocab_path = "./datasets/dbpedia/vocab.json"
    map_file = "./datasets/dbpedia/size.json"
    vocab = load_vocab(vocab_path)
    predictor = Predictor(model_file=model_file, char_to_id=vocab, fix_length=False)
    text = "palaquium canaliculatum , palaquium canaliculatum is a species of plant in the sapotaceae family . it is endemic to sri lanka"
    text = "Șipotu river ( râul mare ) , the Șipotu river is a tributary of the râul mare in romania"
    words = text.split()
    predictor.predict(words)
