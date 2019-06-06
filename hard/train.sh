NMT_DIR=..
python3 ${NMT_DIR}/joint_train.py \
    -model_type CAS \
    -config config.yml \
    -nmt_dir ${NMT_DIR} \
    -src_vocab ../data/douban/vocab_src \
    -tgt_vocab ../data/douban/vocab_tgt \
    -train_file ../data/douban/train \
    -valid_file ../data/douban/dev_dev \
    -rg_model ../douban_pretrain/checkpoint_epoch19.pkl.clean \
    -tg_model ../douban_template/checkpoint_epoch19.pkl \
    -critic_model ./checkpoint_epoch_critic2.pkl
