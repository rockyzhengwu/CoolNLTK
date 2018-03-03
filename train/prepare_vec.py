#!/usr/bin/env python
# -*- coding:utf-8 -*-



def load_file(file_name):
    f = open(file_name)
    for line_no, line in enumerate(f):
        line = line.strip("\n")
        if not line:
            continue
        else:
            content = " ".join(line.split(" , ")[1:])
            words = content.split(" ")
            yield words




def papre_char_vec(train_file, dev_file, test_file, out_filename):
    sent_counter = 0
    outf = open(out_filename, 'w')

    for file_name in [train_file, dev_file, test_file]:
        if not file_name:
            continue

        print("papre char vec train data from : %s" % (file_name))
        for words in load_file(file_name):
            sent_counter += 1
            outf.write(" ".join(words) + "\n")

    print("all sent count %d" % (sent_counter))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train_file path")
    parser.add_argument("--dev_file", default="", required=False, help="dev file path")
    parser.add_argument("--test_file", required=True, help="test file path")
    parser.add_argument("--out_file", required=True, help="out dir for vec path")
    args = parser.parse_args()
    papre_char_vec(args.train_file, args.dev_file, args.test_file, args.out_file)
