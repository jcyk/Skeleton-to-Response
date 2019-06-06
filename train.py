import nmt.utils.misc_utils as utils
import argparse
import codecs
import os
import shutil
import re
import torch
import torch.nn as nn
from torch import cuda
import nmt
import random
from data import Vocab, Data_Loader
import numpy

use_cuda = True


def report_func(opt, global_step, epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.steps_per_stats == -1 % opt.steps_per_stats:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        report_stats = nmt.Statistics()

    return report_stats

def build_or_load_model(args, model_opt, fields):
    if args.model_type == "base":
        model = nmt.model_helper.create_base_model(model_opt, fields)
    if args.model_type == "bibase":
        model = nmt.model_helper.create_bibase_model(model_opt, fields)
    if args.model_type == "ref":
        model = nmt.model_helper.create_ref_model(model_opt, fields)
    if args.model_type == "ev":
        model = nmt.model_helper.create_ev_model(model_opt, fields)
    if args.model_type == "rg":
        model = nmt.model_helper.create_response_generator(model_opt, fields)
    if args.model_type == "joint":
        model = nmt.model_helper.create_joint_model(model_opt, fields)
        model.response_generator.load_checkpoint(args.rg_model)
        model.template_generator.load_checkpoint(args.tg_model)
    latest_ckpt = nmt.misc_utils.latest_checkpoint(model_opt.out_dir)
    start_epoch_at = 0
    if model_opt.start_epoch_at is not None:
        ckpt = 'checkpoint_epoch%d.pkl'%(model_opt.start_epoch_at)
        ckpt = os.path.join(model_opt.out_dir,ckpt)
    else:
        ckpt = latest_ckpt

    if ckpt:
        print('Loding model from %s...'%(ckpt))
        start_epoch_at = model.load_checkpoint(ckpt)
    else:
        print('Building model...')
    print(model)
    #model.save_checkpoint(0, model_opt, os.path.join(model_opt.out_dir,"checkpoint_epoch0.pkl"))
    return model, start_epoch_at


def build_optim(model, optim_opt):
    optim = nmt.Optim(optim_opt.optim_method,
                  optim_opt.learning_rate,
                  optim_opt.max_grad_norm,
                  optim_opt.learning_rate_decay,
                  optim_opt.weight_decay,
                  optim_opt.start_decay_at)

    optim.set_parameters(model.parameters())
    return optim

def build_lr_scheduler(optimizer, opt):

    lr_lambda = lambda epoch: opt.learning_rate_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=[lr_lambda])
    return scheduler

def check_save_model_path(args, opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    print('saving config file to %s ...'%(opt.out_dir))
    shutil.copy(args.config, os.path.join(opt.out_dir,'config.yml'))


def train_model(opt, model, train_iter, valid_iter, fields, optim, lr_scheduler, start_epoch_at):
    train_loss = nmt.NMTLossCompute(model.generator,fields['tgt'].vocab)
    valid_loss = nmt.NMTLossCompute(model.generator,fields['tgt'].vocab)

    if use_cuda:
        train_loss = train_loss.cuda()
        valid_loss = valid_loss.cuda()

    shard_size = opt.train_shard_size
    trainer = nmt.Trainer(opt, model,
                        train_iter,
                        valid_iter,
                        train_loss,
                        valid_loss,
                        optim,
                        shard_size)

    num_train_epochs = opt.num_train_epochs
    print('start training...')
    for step_epoch in range(start_epoch_at+1, num_train_epochs):

        if step_epoch >= opt.start_decay_at:
            lr_scheduler.step()
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())

        #2. Validate on the validation set.
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())

        trainer.epoch_step(step_epoch, out_dir=opt.out_dir)

        model.train()

class vocab_wrapper(object):
    def __init__(self, vocab):
        self.vocab = vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str)
    parser.add_argument("-nmt_dir", type=str)
    parser.add_argument("-model_type", type=str)
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int)
    parser.add_argument("-valid_file", type=str)
    parser.add_argument("-train_file", type=str)
    parser.add_argument("-train_score", type=str, default= None)
    parser.add_argument("-src_vocab", type = str)
    parser.add_argument("-tgt_vocab", type = str)
    parser.add_argument("-rg_model", type = str, default = None)
    parser.add_argument("-tg_model", type = str, default = None)
    args = parser.parse_args()
    opt = utils.load_hparams(args.config)
    cuda.set_device(args.gpuid[0])
    if opt.random_seed > 0:
        random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        numpy.random.seed(opt.random_seed)

    fields = dict()
    vocab_src =  Vocab(args.src_vocab, noST = True)
    vocab_tgt = Vocab(args.tgt_vocab)

    fields['src'] = vocab_wrapper(vocab_src)
    fields['tgt'] = vocab_wrapper(vocab_tgt)

    mask_end = (args.model_type == "ev") or (args.model_type == "joint")
    train = Data_Loader(args.train_file, opt.train_batch_size, score = args.train_score, mask_end = mask_end)
    valid = Data_Loader(args.valid_file, opt.train_batch_size, mask_end = mask_end)

    # Build model.

    model, start_epoch_at = build_or_load_model(args, opt, fields)
    check_save_model_path(args, opt)

    # Build optimizer.
    optim = build_optim(model, opt)
    lr_scheduler = build_lr_scheduler(optim.optimizer, opt)

    if use_cuda:
        model = model.cuda()

    # Do training.

    train_model(opt, model, train, valid, fields, optim, lr_scheduler, start_epoch_at)
    x = 1
    while True:
        x = (x+1)%5
if __name__ == '__main__':
    main()
