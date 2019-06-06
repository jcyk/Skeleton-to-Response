python final_step.py
NMT_DIR=..
python3 ${NMT_DIR}/template.py \
      -config ../template/config.yml \
      -nmt_dir ${NMT_DIR} \
      -src_vocab ../data/douban/vocab_src \
      -tgt_vocab ../data/douban/vocab_tgt \
      -model ../template/checkpoint_epoch8.pkl \
      -test_file ./in \
      -out_file ./in_tem \
      -stop_words ../data/douban/stop_words \
      -mode test

python3 ../read.py douban in_tem > output_skeleton

python3 ${NMT_DIR}/translate.py \
    -test_file ./in_tem \
    -model_type rg \
    -tgt_out ./output_case \
    -model ./hard.pkl \
    -src_vocab ../data/douban/vocab_src \
    -tgt_vocab ../data/douban/vocab_tgt
