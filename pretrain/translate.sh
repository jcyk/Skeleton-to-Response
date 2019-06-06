NMT_DIR=..
python3 ${NMT_DIR}/translate.py \
        -test_file  ../data/golden/ske\
        -model_type rg \
        -tgt_out ablation_response \
        -model ./this.is.the.out/out_dir/checkpoint_epoch19.pkl  \
        -src_vocab ../data/golden/vocab_src \
        -tgt_vocab ../data/golden/vocab_tgt


