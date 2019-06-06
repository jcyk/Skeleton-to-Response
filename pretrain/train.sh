NMT_DIR=..
python3 ${NMT_DIR}/train.py \
    -model_type rg \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/golden/vocab_src \
    -tgt_vocab ../data/golden/vocab_tgt \
    -train_file ../data/golden/train \
    -valid_file ../data/golden/dev

