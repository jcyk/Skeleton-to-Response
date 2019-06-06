import nmt.utils.misc_utils as utils
import argparse
import codecs
import os, sys
import shutil
import re
import torch
import torch.nn as nn
from torch import cuda
import nmt
import random
from data import Vocab, Data_Loader
from nmt.utils.data_utils import sequence_mask
import torch.nn.functional as F
use_cuda = True

def build_or_load_model(args, model_opt, fields):
    model = nmt.model_helper.create_template_generator(model_opt, fields)

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

def save_per_epoch(model, epoch, opt):
    f = open(os.path.join(opt.out_dir,'checkpoint'),'w')
    f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
    f.close()
    model.save_checkpoint(epoch, opt, os.path.join(opt.out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

def train_model(opt, model, train_iter, valid_iter, fields, optim, lr_scheduler, start_epoch_at):
    sys.stdout.flush()
    for step_epoch in range(start_epoch_at+1, opt.num_train_epochs):
        for batch in train_iter:
            model.zero_grad()
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            target, _ = batch.mask
            ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
            preds = model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
            preds = preds.squeeze(2)
            mask = sequence_mask(ref_tgt_lengths).transpose(0, 1)
            tot = mask.float().sum()

            reserved = target.float().sum()
            w1 = (0.5 * tot / reserved).data[0]
            w2 = (0.5 * tot / (tot - reserved)).data[0]
            #w1, w2 = 1., 1.
            weight = torch.FloatTensor(mask.size()).zero_().cuda()
            weight.masked_fill_(mask, w2)
            weight.masked_fill_(torch.eq(target, 1).data, w1)

            loss = F.binary_cross_entropy(preds, target.float(), weight)
            loss.backward()
            optim.step()

        loss = 0.
        acc = 0.
        ntokens = 0.
        reserved, targeted, received = 0., 0., 0.
        model.eval()
        for batch in valid_iter:
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            target, _ = batch.mask
            target= target.float()
            ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
            preds = model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
            preds = preds.squeeze(2)
            mask = sequence_mask(ref_tgt_lengths).transpose(0, 1).float()
            loss += F.binary_cross_entropy(preds, target, mask, size_average = False).data[0]
            ans = torch.ge(preds, 0.5).float()
            acc += (torch.eq(ans, target).float().data * mask ).sum()
            received += (ans.data * target.data * mask).sum()
            reserved += (ans.data * mask).sum()
            targeted += (target.data * mask).sum()
            ntokens += mask.sum()
        print ("epoch: ", step_epoch, "valid_loss: ", loss/ ntokens, "valid_acc: ", acc/ntokens, "precision: ", received/reserved, "recall: ", received/targeted)

        if step_epoch >= opt.start_decay_at:
            lr_scheduler.step()
        model.train()
        save_per_epoch(model, step_epoch, opt)
        sys.stdout.flush()

class vocab_wrapper(object):
    def __init__(self, vocab):
        self.vocab = vocab

def get_sentence(idx, vocab):
    return ' '.join([vocab.itos[x] for x in idx])

def Tensor2List(x, xlen, tgt = False):
    y = x.transpose(0, 1).data.tolist()
    return [ z[1:l-1] if tgt else z[:l] for z,l in zip(y, xlen) ]

def output_results(ans, batch, fo, vocab_tgt, for_train = True):
    src = Tensor2List(batch.src[0], batch.src[1])
    tgt = Tensor2List(batch.tgt[0], batch.tgt[1], tgt = True)
    ref_src = Tensor2List(batch.ref_src[0], batch.ref_src[1])
    #batch.ref_tgt[0].data.masked_fill_( torch.lt((batch.mask[0]).float(), 1.).data , 0)
    batch.ref_tgt[0].data.masked_fill_( torch.lt(ans, 1.).data , 0)
    ref_tgt = Tensor2List(batch.ref_tgt[0], batch.ref_tgt[1])

    if not for_train:
        for x in ref_tgt:
            fo.write(get_sentence(x, vocab_tgt)+'\n')
        return

    for x, y, z , w in zip(src, tgt, ref_src, ref_tgt):
        a = ' '.join( [str(t) for t in x ])
        b = ' '.join( [str(t) for t in y ])
        c = ' '.join( [str(t) for t in z ])
        d = ' '.join( [str(t) for t in w ])
        fo.write('|'.join([a,b,c,d])+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str)
    parser.add_argument("-nmt_dir", type=str)
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int)
    parser.add_argument("-valid_file", type=str)
    parser.add_argument("-train_file", type=str)
    parser.add_argument("-test_file", type = str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-src_vocab", type = str)
    parser.add_argument("-tgt_vocab", type = str)
    parser.add_argument("-mode", type = str)
    parser.add_argument("-out_file", type = str)
    parser.add_argument("-stop_words", type = str, default = None)
    parser.add_argument("-for_train", type = bool, default = True)
    args = parser.parse_args()
    opt = utils.load_hparams(args.config)

    if opt.random_seed > 0:
        random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)

    fields = dict()
    vocab_src =  Vocab(args.src_vocab, noST = True)
    vocab_tgt = Vocab(args.tgt_vocab)
    fields['src'] = vocab_wrapper(vocab_src)
    fields['tgt'] = vocab_wrapper(vocab_tgt)

    if args.mode == "test":
        model = nmt.model_helper.create_template_generator(opt, fields)
        if use_cuda:
            model = model.cuda()
        model.load_checkpoint(args.model)
        model.eval()
        test =  Data_Loader(args.test_file, opt.train_batch_size, train = False, mask_end = True, stop_words = args.stop_words)
        fo = open(args.out_file, 'w')
        loss, acc, ntokens = 0., 0., 0.
        reserved, targeted, received = 0., 0., 0.
        for batch in test:
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            target, _ = batch.mask
            target= target.float()
            ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
            preds = model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
            preds = preds.squeeze(2)
            mask = sequence_mask(ref_tgt_lengths).transpose(0, 1).float()
            loss += F.binary_cross_entropy(preds, target, mask, size_average = False).data[0]
            ans = torch.ge(preds, 0.5).float()
            output_results(ans, batch, fo, vocab_tgt, args.for_train)
            acc += (torch.eq(ans, target).float().data * mask ).sum()
            received += (ans.data * target.data * mask).sum()
            reserved += (ans.data * mask).sum()
            targeted += (target.data * mask).sum()
            ntokens += mask.sum()
        print ("test_loss: ", loss/ ntokens, "test_acc: ", acc/ntokens, "precision:", received/reserved, "recall: ", received/targeted, "leave percentage", targeted/ntokens)
        fo.close()
        #x = 1
        #while True:
        #    x = (x+1)%5
        return

    train = Data_Loader(args.train_file, opt.train_batch_size, mask_end = True, stop_words = args.stop_words)
    valid = Data_Loader(args.valid_file, opt.train_batch_size, mask_end = True, stop_words = args.stop_words)

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
    print ("DONE")
if __name__ == '__main__':
    main()
