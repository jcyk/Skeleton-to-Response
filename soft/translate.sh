NMT_DIR=..
python3 ${NMT_DIR}/translate.py \
    -test_file ../data/douban/human_dev \
    -model_type soft \
    -tgt_out ./soft_out \
    -model ./checkpoint_epoch5.pkl \
    -src_vocab ../data/douban/vocab_src \
    -tgt_vocab ../data/douban/vocab_tgt
