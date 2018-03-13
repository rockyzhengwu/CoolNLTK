#!/usr/bin/env python
#-*-coding:utf-8-*-


import tensorflow as tf
from tensorflow.python.platform import gfile

def save_to_binary(checkpoints_path, out_model_path):
    checkpoint_dir = checkpoints_path

    graph = tf.Graph()
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint_file)

    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        # input_x = graph.get_operation_by_name("input_x").outputs[0]
        # dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # scores = graph.get_operation_by_name("output/predprob").outputs[0]

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def,
            output_node_names=['output/predictions', 'output/predprob']
        )
        with tf.gfile.FastGFile(out_model_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


import pickle
def save_vocab_to_text(path, outf_path):
    outf = open(outf_path, 'w')
    with gfile.Open(path, 'rb') as f:
        data = pickle.loads(f.read())
    print(type(data.vocabulary_._mapping))
    word_map = data.vocabulary_._mapping
    for k, v in word_map.items():
        outf.write("\t".join([k, str(v)])+"\n")



if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="checkpoint dir")
    parser.add_argument("--out_dir", required=True, help="out dir ")
    args = parser.parse_args()
    print("export model from: %s, save to :%s"%(args.checkpoint_dir, args.out_dir) )
    save_to_binary(args.checkpoint_dir, os.path.join(args.out_dir, "modle.pb"))
