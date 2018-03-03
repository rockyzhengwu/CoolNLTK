#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import pickle
import os
import json
import re


def load_map_file(vocab_file):
    vocab = json.load(open(vocab_file))
    return vocab


def seg_to_tfrecords(text_file, out_dir, map_file, out_name):
    out_filename = os.path.join(out_dir, out_name + ".tfrecord")
    vocab = load_map_file(map_file)

    writer = tf.python_io.TFRecordWriter(out_filename)
    num_sample = 0
    all_oov = 0
    total_word = 0
    oov_count = 0

    with open(text_file) as f:
        for lineno, line in enumerate(f):
            num_sample += 1
            line = line.strip("\n")
            if not line:
                continue
            word_count,  oov = create_one_seg_sample(writer, line, vocab)
            total_word += word_count
            all_oov += oov

    print("oov rate : %f" % (1.0 * oov_count / total_word))

    return num_sample


def create_one_seg_sample(writer, line, vocab):
    label = re.match("__label__\d+", line)
    label = label.group()
    line = line[len(label+' , '):]
    label = int(label.replace("__label__", ""))
    line = line.strip()
    words = line.split(" ")

    word_list = []
    oov_count = 0
    word_count = 0
    for w in words:
        word_count +=1
        if w in vocab:
            word_list.append(vocab.get(w))
        else:
            word_list.append(vocab.get("<OOV>"))
            oov_count +=1

    label_list = [0] * NUM_CLASS
    label_list[label-1] = 1

    example = tf.train.SequenceExample()
    sent_len = len(words)
    if sent_len == 0:
        print("line: ", line)
        print('words:', words)
    assert sent_len != 0

    # todo padding all words
    if sent_len > MAX_LENGTH:
        pad_word_list = word_list[:MAX_LENGTH]
    else:
        pad_list = [0] * (MAX_LENGTH - sent_len)
        pad_word_list = word_list+pad_list

    fl_labels = example.feature_lists.feature_list["words"]
    for w in pad_word_list:
        fl_labels.feature.add().int64_list.value.append(w)

    fl_tokens = example.feature_lists.feature_list["label"]
    for l in label_list:
        fl_tokens.feature.add().int64_list.value.append(l)

    writer.write(example.SerializeToString())
    return word_count, oov_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train file path")
    parser.add_argument("--dev_file", required=False, default="", help="dev file path")
    parser.add_argument("--vocab_file", required=True, help="map file path")
    parser.add_argument("--test_file", required=True, help="test file path")
    parser.add_argument("--out_dir", required=True, help=" out dir for tfrecord ")
    parser.add_argument("--size_file", required=True, help="the size file create by create map step")
    parser.add_argument("--max_length", default=100, type=int, help="max length of sent")
    args = parser.parse_args()

    MAX_LENGTH = args.max_length

    size_filename = args.size_file
    print(size_filename)

    with open(size_filename, 'r') as f:
        size_obj = json.load(f)

    NUM_CLASS = size_obj.get("num_tag")


    train_num = seg_to_tfrecords(args.train_file, args.out_dir, args.vocab_file, "train")
    test_num = seg_to_tfrecords(args.test_file, args.out_dir, args.vocab_file, "test")

    dev_num = 0
    if args.dev_file:
        dev_num = seg_to_tfrecords(args.dev_file, args.out_dir, args.vocab_file, "dev")

    print("train sample : %d" % (train_num))
    print("test sample : %d" % (test_num))
    print("dev sample :%d" % (dev_num))

    with open(os.path.join(size_filename), 'w') as f:
        size_obj['train_num'] = train_num
        size_obj['dev_num'] = dev_num
        size_obj['test_num'] = test_num
        json.dump(size_obj, f)