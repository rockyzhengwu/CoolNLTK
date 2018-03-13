#!/usr/bin/env python
# -*-coding:utf-8-*-



import numpy as np
import datetime
import tensorflow as tf
from sklearn.metrics.classification import classification_report
from sklearn.metrics.classification import  accuracy_score

from models.text_cnn import TextCNN
from models.text_rnn import TextRNN
from models.c_lstm import CLSTM

from data_utils import load_size_file
from data_utils import load_vocab
from data_utils import SegBatcher
import word2vec
import os


def init_config(vocab_size, num_class):
    config = {}
    config['clip'] = FLAGS.clip
    config['optimizer'] = FLAGS.optimizer
    # cnn
    config['filter_sizes'] = [int(i) for i in FLAGS.filters.split(",")]
    config['num_filter'] = FLAGS.num_filters
    config['embedding_dim'] = FLAGS.embedding_dim

    config["num_classes"] = num_class

    config["num_filters"] = FLAGS.num_filters
    config["vocab_size"] = vocab_size
    config['learning_rate'] = FLAGS.learning_rate

    config["decay_steps"] = FLAGS.decay_steps
    config["decay_rate"] = FLAGS.decay_rate
    config["decay_rate_big"] = FLAGS.decay_rate_big
    config["clip_gradients"] = FLAGS.clip_gradients
    config["l2_lambda"] = FLAGS.l2_lambda
    config["optimizer"] = FLAGS.optimizer
    config["initializer"] = None
    config["sequence_length"] = FLAGS.sequence_length
    config['num_hidden'] = FLAGS.num_hidden

    # clstm
    config['cnn_filter_size'] = FLAGS.cnn_filter_size
    config['cnn_num_filter'] = FLAGS.cnn_num_filter
    config['cnn_pool_size'] = FLAGS.cnn_pool_size

    return config


def get_model():

    model_type = FLAGS.model
    if model_type == "cnn":
        model_class = TextCNN
    elif model_type == "bilstm":
        model_class = TextRNN
    elif model_type == "clstm":
        model_class = CLSTM
    else:
        raise Exception("model_type {} not implement".format(model_type))
    return model_class


def main(argv):
    model_class = get_model()

    size_map = load_size_file(FLAGS.size_file)
    vocab_size = size_map.get('vocab_size')
    num_class = size_map.get("num_tag")
    num_train = size_map.get("train_num")


    model_conf = init_config(vocab_size, num_class)
    print(model_conf)

    _, id_to_word = load_vocab(FLAGS.vocab_file)


    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        wv = word2vec.Word2vec()
        embed = wv.load_w2v_array(FLAGS.embedding_file, id_to_word)
        print("embeding shape:", embed.shape)
        word_embedding = tf.constant(embed, dtype=tf.float32)
        model = model_class(model_conf, word_embedding)

        train_batcher = SegBatcher(FLAGS.train_file, FLAGS.batch_size, num_epochs=FLAGS.max_epoch)
        test_batcher = SegBatcher(FLAGS.test_file, FLAGS.batch_size, num_epochs=1)

        print("train_file ====> ", FLAGS.train_file)
        print("test_file =====>", FLAGS.test_file)
        print("batch size =====>", FLAGS.batch_size)
        print("most epoch ======>", FLAGS.max_epoch)

        loss_summary = tf.summary.scalar("loss", model.loss_val)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])

        tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=FLAGS.out_dir, save_model_secs=0, save_summaries_secs=0)
        with sv.managed_session(config=session_conf) as sess:
            threads = tf.train.start_queue_runners(sess=sess)
            test_batches = []
            done = False

            print("load all dev batches to memory")

            while not done:
                try:
                    words, labels = sess.run(test_batcher.next_batch_op)
                    test_batches.append((words, labels))
                except Exception as e:
                    done = True
            def run_eval(batchs):
                print("eval....")
                true_labels = []
                pred_labels = []
                for words, label in batchs:
                    feed_dict = {
                        model.input_x: words,
                        model.input_y:label,
                        model.dropout_keep_prob:1.0,
                    }
                    predictions,acc = sess.run([model.predictions, model.accuracy], feed_dict=feed_dict)
                    pred_labels.append(predictions)
                    label  = np.argmax(label, axis=1)
                    true_labels.append(label)
                true_labels = np.concatenate(true_labels, axis=0)
                pred_labels = np.concatenate(pred_labels, axis=0)
                report = classification_report(true_labels, pred_labels)
                print(report)
                acc = accuracy_score(true_labels, pred_labels)
                return acc

            best_acc = 0.0
            for epoch in range(FLAGS.max_epoch):
                if sv.should_stop():
                    # todo test
                    print("stop.......")
                    break
                examples = 0
                while examples < num_train:
                    try:
                        batch = sess.run(train_batcher.next_batch_op)
                    except Exception as e:
                        print(e)
                        exit(0)

                    words, label = batch
                    feed_dict = {
                        model.input_x: words,
                        model.input_y: label,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

                    _, step, loss, accuracy, summaries = sess.run(
                    [model.train_op, model.global_step, model.loss_val, model.accuracy, train_summary_op],
                    feed_dict)
                    examples += len(words)

                    if step % FLAGS.eval_step == 0:
                        # todo evaliate
                        acc = run_eval(test_batches)

                        if acc< best_acc:
                            # todo finish
                            pass
                        else:
                            best_acc = acc
                            sv.saver.save(sess, os.path.join(FLAGS.out_dir, "model"), global_step=step,)
                        print("{}: test{:g} best acc :{}".format(time_str, acc, best_acc))
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))



if __name__ == '__main__':
    ###############################################
    #  common config
    ###############################################

    tf.app.flags.DEFINE_string("train_file", "", "train file with tfrecord")
    tf.app.flags.DEFINE_string("test_file", "", "test file with tfrecod format")
    tf.app.flags.DEFINE_string("embedding_file", "", "pre embeding file")
    tf.app.flags.DEFINE_string("out_dir", "", "log path of the supervisor")
    tf.app.flags.DEFINE_string("size_file", "", "size file create before")
    tf.app.flags.DEFINE_string("vocab_file", "", "vocab file ")

    tf.app.flags.DEFINE_integer("max_epoch", 100, "max epoch")
    tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
    tf.app.flags.DEFINE_integer("eval_step", 10, "evaluation step size")
    tf.app.flags.DEFINE_string("optimizer", "Adam", "optimizer ")
    tf.app.flags.DEFINE_integer("clip", 5, "clip  ")
    tf.app.flags.DEFINE_string("model", "bilstm", "model")
    tf.app.flags.DEFINE_integer("embedding_dim", 100, "pre embedding dim")
    tf.app.flags.DEFINE_integer("save_model_secs", 30, "save model seconds")
    tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "keep dropout prob")

    tf.app.flags.DEFINE_integer("decay_steps", 5, "")
    tf.app.flags.DEFINE_float("decay_rate", 0.65, "decay rate")
    tf.app.flags.DEFINE_float("decay_rate_big", 0.95, "decay rate big")
    tf.app.flags.DEFINE_float("clip_gradients", 0.05, "")
    tf.app.flags.DEFINE_float("l2_lambda", 0.05, "")
    tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    tf.app.flags.DEFINE_integer("sequence_length", 100, "sequence length")


    ##############################################
    # text cnn
    ##############################################

    tf.app.flags.DEFINE_string("filters", "3,5,7", "filter sizes for text_cnn")
    tf.app.flags.DEFINE_integer("num_filters", 128, "num of filter for text_cnn")
    ##############################################
    #bilstm
    ##############################################
    tf.app.flags.DEFINE_integer("num_hidden", 256, "num of hidden for text_rnn")

    #############################################
    #clstm
    #############################################
    tf.app.flags.DEFINE_integer("cnn_filter_size", 3, "CNN filter size for clstm")
    tf.app.flags.DEFINE_integer("cnn_num_filter", 256, "num filter for clstm")
    tf.app.flags.DEFINE_integer("cnn_pool_size", 2,  "pool size for clstm")


    FLAGS = tf.flags.FLAGS
    tf.app.run()