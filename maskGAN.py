import nmt.utils.misc_utils as utils
import argparse
import codecs
import os
import shutil
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
import nmt
import random
from data import Vocab, Data_Loader, ListsToTensor
from torch.autograd import Variable
import sys


use_cuda = True

class GAN(nn.Module):
    def __init__(self, generator, discriminator, critic):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'generator_dict': self.generator.state_dict(),
                    'discriminator_dict': self.discriminator.state_dict(),
                    'critic_dict': self.critic.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.generator.load_state_dict(ckpt['generator_dict'])
        self.discriminator.load_state_dict(ckpt['discriminator_dict'])
        self.critic.load_state_dict(ckpt['critic_dict'])
        epoch = ckpt['epoch']
        return epoch

def sample(model, src, ref_src, ref_tgt, src_lengths, ref_src_lengths, ref_tgt_lengths, max_len, show_sample = False):
    model_type = model.__class__.__name__
    if model_type =="refNMTModel":
        context, enc_states, context_keys, context_mask, src_context, src_mask = model.encode(src, ref_src, ref_tgt, src_lengths, ref_src_lengths, ref_tgt_lengths)
    if model_type == "responseGenerator":
        context, enc_states, context_mask, dist, src_context, src_mask = model.encode(src, ref_tgt, src_lengths, ref_tgt_lengths)
    dec_states = model.init_decoder_state(enc_states, context)

    vocab = model.fields['tgt'].vocab
    EOS_idx = vocab.stoi[vocab.EOS]
    PAD_idx = vocab.stoi[vocab.PAD]
    EOT_idx = vocab.stoi[vocab.EOT]
    batch_size = src.size(1)

    notyet = torch.ByteTensor(batch_size).fill_(1)
    inp = Variable(torch.LongTensor(batch_size).fill_(EOS_idx))

    pad_mask = torch.LongTensor([PAD_idx])
    if use_cuda:
        notyet = notyet.cuda()
        inp = inp.cuda()
        pad_mask = pad_mask.cuda()

    result = [inp]
    eps = 1e-12
    log_prob= []

    while notyet.any() and len(result)<= max_len:
        inp = inp.unsqueeze(0)
        if model_type =="refNMTModel":
            dec_out, dec_states, attn = model.decode(inp, context_keys, context, dec_states, context_mask, src_context, src_mask)
        if model_type =="responseGenerator":
            dec_out, dec_states, attn = model.decode(inp, context, dec_states, None, context_mask, src_context, src_mask)
        dec_out = dec_out.squeeze(0)
        cur_log_prob = model.generator(dec_out)
        cur_log_prob.data.index_fill_(1, pad_mask, -float('inf'))
        word_prob = torch.exp(cur_log_prob + eps)
        #inp = torch.multinomial(word_prob, 1).squeeze(-1)
        _, inp = torch.max(cur_log_prob, -1)
        cur_log_prob = torch.gather(cur_log_prob, -1, inp.view(-1, 1)).squeeze(-1)
        cur_log_prob.data.masked_fill_(1-notyet, 0.)
        log_prob.append(cur_log_prob)
        inp.data.masked_fill_( 1-notyet, PAD_idx) # batch_size 
        result.append(inp)

        endding = torch.eq(inp, EOT_idx)
        notyet.masked_fill_(endding.data, 0)

    result = torch.stack(result, 0)
    log_prob = torch.stack(log_prob, 0)

    x = result.t().data.tolist()
    new_x = []
    for t in x:
        new_t = []
        for tt in t:
            if tt != PAD_idx:
                new_t.append(tt)
        new_x.append(new_t)
    x = new_x

    if show_sample:
        for t in x:
            print (' '.join([vocab.itos[tt] for tt in t]))
    return ListsToTensor(x, tgt = False), log_prob

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
    if args.model_type == "ref":
        generator, discriminator, critic = nmt.model_helper.create_GAN_model(model_opt, fields)
        model = GAN(generator, discriminator, critic)
        if args.start_point is None:
            generator.load_checkpoint("init_point")
            discriminator.base_model.load_checkpoint('init_point')
            critic.base_model.load_checkpoint('init_point')
        else:
            model.load_checkpoint(args.start_point)

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

def build_optims_and_lr_schedulers(model, opt):
    optimG = nmt.Optim(opt.optim_method,
                  opt.learning_rate,
                  opt.max_grad_norm,
                  opt.learning_rate_decay,
                  opt.weight_decay,
                  opt.start_decay_at)

    optimG.set_parameters(model.generator.parameters())

    lr_lambda = lambda epoch: opt.learning_rate_decay ** epoch
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizer=optimG.optimizer, lr_lambda=[lr_lambda])
    optimD = nmt.Optim(opt.optim_method,
                    opt.learning_rate_D,
                    opt.max_grad_norm,
                    opt.learning_rate_decay,
                    opt.weight_decay,
                    opt.start_decay_at)
    optimD.set_parameters( [ x for x in model.discriminator.parameters() ] + [ y for y in model.critic.parameters()] )
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizer=optimD.optimizer, lr_lambda=[lr_lambda])
    return optimG, schedulerG, optimD, schedulerD

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

def pretrain_discriminators(opt, model, train_iter, valid_iter, fields, optim, lr_scheduler, start_epoch_at):
    for step_epoch in range(start_epoch_at+1, opt.num_train_epochs):
        for batch in train_iter:
            model.zero_grad()
            src_inputs, src_lengths = batch.src
            tgt_inputs = batch.tgt[0]
            ref_src_inputs, ref_src_lengths = batch.ref_src
            ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
            (fake_tgt_inputs, _), fake_log_prob = sample(model.generator, src_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths, opt.max_sample_len)
            real_output = model.discriminator(src_inputs, tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
            fake_output  = model.discriminator(src_inputs, fake_tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
            real_output = real_output[1:]
            fake_output = fake_output[1:]

            target = torch.ones_like(real_output)
            loss_real = F.binary_cross_entropy_with_logits(real_output, target, torch.ne(tgt_inputs[1:], 0).float(), size_average = False)
            target = torch.zeros_like(fake_output)
            loss_fake = F.binary_cross_entropy_with_logits(fake_output, target, torch.ne(fake_tgt_inputs[1:], 0).float(), size_average = False)

            loss = (loss_real + loss_fake)/ (2 * batch.batch_size)
            loss.backward()
            optim.step()
        save_per_epoch(model, step_epoch, opt)
        sys.stdout.flush()

def G_turn(model, batch, optim, opt):
    model.zero_grad()
    advantages, log_probs, mask = D_turn(model, batch, None, opt, forG = True)
    loss = -(advantages * log_probs) * mask.float()
    loss = torch.sum(loss)/ batch.batch_size
    loss.backward()
    optim.step()

def D_turn(model, batch, optim, opt, forG = False, show_sample = False):
    if not forG:
        model.zero_grad()
    src_inputs, src_lengths = batch.src
    tgt_inputs = batch.tgt[0]
    ref_src_inputs, ref_src_lengths = batch.ref_src
    ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt

    if show_sample:
        sample(model.generator, src_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths, opt.max_sample_len, show_sample = True)
        return
    (fake_tgt_inputs, _), fake_log_prob = sample(model.generator, src_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths, opt.max_sample_len)

    real_output = model.discriminator(src_inputs, tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
    fake_output = model.discriminator(src_inputs, fake_tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
    real_output = real_output[1:]
    fake_output = fake_output[1:]

    target = torch.ones_like(real_output)
    loss_real = F.binary_cross_entropy_with_logits(real_output, target, torch.ne(tgt_inputs[1:], 0).float(), size_average = False)
    target = torch.zeros_like(fake_output)
    fake_tgt_mask = torch.ne(fake_tgt_inputs[1:], 0)
    loss_fake = F.binary_cross_entropy_with_logits(fake_output, target, fake_tgt_mask.float(), size_average = False)

    loss = (loss_real + loss_fake)/ (2 * batch.batch_size)
    eps = 1e-12

    estimated_rewards = model.critic(src_inputs, fake_tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
    estimated_rewards = estimated_rewards[:-1]

    rewards = torch.log(F.sigmoid(fake_output) + eps)
    rewards.data.masked_fill_(1 - fake_tgt_mask.data, 0.)
    split_rewards = torch.split(rewards, 1, dim = 0)

    sum_rewards = []
    cur = 0.
    for r in split_rewards[::-1]:
        cur = cur * opt.gamma + r
        sum_rewards.append(cur)
    sum_rewards = torch.cat(sum_rewards[::-1], 0)

    if forG:
        return (sum_rewards - estimated_rewards).detach(), fake_log_prob, fake_tgt_mask
    critic_loss = (sum_rewards - estimated_rewards)**2
    critic_loss.data.masked_fill_(1 - fake_tgt_mask.data, 0.)
    critic_loss = torch.sum(critic_loss)/ batch.batch_size
    loss = loss + critic_loss
    loss.backward()
    optim.step()

def train_model(opt, model, train_iter, valid_iter, fields, optimG, lr_schedulerG, optimD, lr_schedulerD, start_epoch_at):
    num_train_epochs = opt.num_train_epochs
    num_updates = 0
    print('start training...')
    valid_loss = nmt.NMTLossCompute(model.generator.generator,fields['tgt'].vocab)
    if use_cuda:
        valid_loss = valid_loss.cuda()
    shard_size = opt.train_shard_size
    trainer = nmt.Trainer(opt, model.generator, train_iter, valid_iter, valid_loss, valid_loss, optimG, lr_schedulerG, shard_size, train_loss_b = None)

    for step_epoch in range(start_epoch_at+1, num_train_epochs):
        for batch in train_iter:
            if num_updates  % (opt.D_turns+1) == -1 % (opt.D_turns+1):
                G_turn(model, batch, optimG, opt)
            else:
                D_turn(model, batch, optimD, opt)
            if num_updates % (opt.show_sample_every) == -1 %(opt.show_sample_every):
                D_turn(model, batch, optimD, opt, show_sample = True)
            num_updates += 1
            sys.stdout.flush()
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        sys.stdout.flush()
        if step_epoch >= opt.start_decay_at:
            lr_schedulerD.step()
            lr_schedulerG.step()
        save_per_epoch(model, step_epoch, opt)
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
    parser.add_argument("-start_point", type = str, default = None)

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

    train = Data_Loader(args.train_file, opt.train_batch_size, score = args.train_score, mask_end = (args.model_type == "ev"))
    valid = Data_Loader(args.valid_file, opt.train_batch_size, mask_end = (args.model_type == "ev"))

    # Build model.

    model, start_epoch_at = build_or_load_model(args, opt, fields)
    check_save_model_path(args, opt)

    optimG, schedulerG, optimD, schedulerD = build_optims_and_lr_schedulers(model, opt)

    if use_cuda:
        model = model.cuda()

    # Do training.
    #pretrain_discriminators(opt, model, train, valid, fields, optimD, schedulerD, start_epoch_at)
    train_model(opt, model, train, valid, fields, optimG, schedulerG, optimD, schedulerD, start_epoch_at)
    print("DONE")
    x = 0
    while True:
        x = (x +1)%5
if __name__ == '__main__':
    main()
