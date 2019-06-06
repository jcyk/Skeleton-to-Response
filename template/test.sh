NMT_DIR=..
python3 ${NMT_DIR}/template.py \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/golden/vocab_src \
    -tgt_vocab ../data/golden/vocab_tgt \
    -model ./out/out_dir/checkpoint_epoch19.pkl \
    -test_file ../data/golden/train \
    -out_file ./TMP \
    -stop_words ../data/golden/stop_words \
    -mode test
python clean.py TMP > final_output