#!/usr/bin/env bash


TRAIN_FILE=./datasets/dbpedia/dbpedia.train
TEST_FILE=./datasets/dbpedia/dbpedia.test

DATA_OUT_DIR=./datasets/dbpedia/

MODEL_OUT_DIR=./results/dbpedia/

WORD2VEC=./third_party/word2vec/word2vec

EMBEDDING_SIZE=100

MAX_LENGTH=100

MODEL=clstm

############################################################
VEC_PATH="$DATA_OUT_DIR"/vec

PRE_VEC_TRAIN="$DATA_OUT_DIR"/vec/pre_vec_train.txt
EMBEDDING_FILE="$DATA_OUT_DIR"/vec/vec_"$EMBEDDING_SIZE".txt

VOCAB_FILE="$DATA_OUT_DIR"/vocab.json
SIZE_FILE="$DATA_OUT_DIR"/size.json

TRAIN_TF="$DATA_OUT_DIR"train.tfrecord
TEST_TF="$DATA_OUT_DIR"test.tfrecord

CHECKPOINT_DIR="$MODEL_OUT_DIR/$MODEL"/ckpt/



if [ ! -d $VEC_PATH ];
    then mkdir -p $VEC_PATH;
fi;

if [ "$1" = "pre" ] ; then
python prepare_vec.py --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" --out_file "$PRE_VEC_TRAIN"


elif [ "$1" = "vec" ]; then
time $WORD2VEC -train "$PRE_VEC_TRAIN" -output "$EMBEDDING_FILE" -cbow 1 -size "$EMBEDDING_SIZE" -window 8 -negative 25 -hs 0 \
  -sample 1e-4 -threads 4 -binary 0 -iter 15 -min-count 5

elif [ "$1" = "map" ]; then
python create_map_file.py --train_file "$TRAIN_FILE" --embeding_file "$EMBEDDING_FILE" --map_file "$VOCAB_FILE" \
   --size_file "$SIZE_FILE"

elif [ "$1" = "data" ]; then
python text_to_tfrecords.py --train_file "$TRAIN_FILE" --test_file "$TEST_FILE" --vocab_file\
   "$VOCAB_FILE" --size_file "$SIZE_FILE" --out_dir "$DATA_OUT_DIR" --max_length "$MAX_LENGTH"

elif [ "$1" = "train" ]; then
python train.py --model "$MODEL" --train_file "$TRAIN_TF" --test_file "$TEST_TF" --embedding_file "$EMBEDDING_FILE" \
--out_dir "$CHECKPOINT_DIR" --vocab_file "$VOCAB_FILE" --size_file "$SIZE_FILE"
elif [ "$1" = "export" ]; then
python export_model.py --checkpoint_dir "$CHECKPOINT_DIR" --out_dir "$MODEL_OUT_DIR/$MODEL"
else

echo "param error action need"
fi;
