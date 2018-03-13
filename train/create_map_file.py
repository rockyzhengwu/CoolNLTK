#!/usr/bin/env python
# -*-coding:utf-8-*-

import argparse
import json


OOV_STR = "<OOV>"
PAD_STR = "<PAD>"


def create_vocab(embeding_file):

    vocab_dict = {}
    vocab_dict[PAD_STR] = 0
    vocab_dict[OOV_STR] = 1

    f = open(embeding_file, errors="ignore")
    m, n = f.readline().split(" ")
    n = int(n)
    m = int(m)
    print("preembeding size : %d"%(m))

    for i, line in enumerate(f):
        word = line.split()[0]
        if not word:
            continue
        if word not in vocab_dict:
            vocab_dict[word] = len(vocab_dict)
    print("vocab size : %d" % len(vocab_dict))
    return vocab_dict


def tag_to_map(train_file):
    f = open(train_file)
    tags =  set()

    for i, line in enumerate(f):
        line = line.strip("\n")
        if not line:
            continue
        data = line.split(" , ")
        label = data[0]
        label_index = int(label.replace("__label__", ""))
        tags.add(label_index)
    return list(tags)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train file path")
    parser.add_argument("--embeding_file", required=True, help="embeding file path")
    parser.add_argument("--map_file", required=True, help="the out dir of map file")
    parser.add_argument("--size_file", required=True, help="save size to")
    args = parser.parse_args()

    f = open(args.map_file, 'w')
    vocab = create_vocab(embeding_file=args.embeding_file)
    tags = tag_to_map(train_file=args.train_file )
    print("tag map result: ")
    print(tags)
    json.dump(vocab, f, indent=4)
    vocab_size = len(vocab)
    num_class = len(tags)
    print("vocab size : %d, num of tag : %d"%(vocab_size, num_class))
    size_file = open(args.size_file, 'w')
    json.dump({"vocab_size": vocab_size, "num_tag":num_class}, size_file)
