NMT_DIR=..
python3 ${NMT_DIR}/joint_train.py \
    -model_type JNT \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/douban/vocab_src \
    -tgt_vocab ../data/douban/vocab_tgt \
    -train_file ../data/douban/train \
    -valid_file ../data/douban/dev_dev
