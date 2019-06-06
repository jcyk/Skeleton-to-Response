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
import numpy as np
from maskGAN import sample
from nmt.Trainer import Statistics
from torch.autograd import Variable
use_cuda = True

def report_func(opt, global_step, epoch, batch, num_batches,
                start_time, lr, report_stats):
    if batch % opt.steps_per_stats == -1 % opt.steps_per_stats:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        report_stats = nmt.Statistics()

    return report_stats

def build_or_load_model(args, model_opt, fields):
    if args.model_type == "CAS":
        model = nmt.model_helper.create_joint_model(model_opt, fields)
        if args.rg_model is not None:
            model.response_generator.load_checkpoint(args.rg_model)
        if args.tg_model is not None:
            model.template_generator.load_checkpoint(args.tg_model)
    if args.model_type == "JNT":
        model = nmt.model_helper.create_joint_template_response_model(model_opt, fields)
    if model_opt.use_critic:
        critic = nmt.model_helper.create_critic_model(model_opt, fields)
        if args.critic_model is not None:
            critic.load_checkpoint(args.critic_model)
    else:
        critic = None
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
    #model.save_checkpoint(0, model_opt, os.path.join(model_opt.out_dir,"checkpoint_epoch0.noskeleton.pkl"))
    return model, critic, start_epoch_at


def build_optims_and_schedulers(model, critic, opt):
    if model.__class__.__name__ == "jointTemplateResponseGenerator":
        optimR = nmt.Optim(opt.optim_method,
                           opt.learning_rate_R,
                           opt.max_grad_norm,
                           opt.learning_rate_decay,
                           opt.weight_decay,
                           opt.start_decay_at)
        optimR.set_parameters(model.parameters())
        lr_lambda = lambda epoch: opt.learning_rate_decay ** epoch
        schedulerR = torch.optim.lr_scheduler.LambdaLR(optimizer=optimR.optimizer,
                                                       lr_lambda=[lr_lambda])
        return optimR, schedulerR, None, None, None, None

    optimR = nmt.Optim(opt.optim_method,
                  opt.learning_rate_R,
                  opt.max_grad_norm,
                  opt.learning_rate_decay,
                  opt.weight_decay,
                  opt.start_decay_at)

    optimR.set_parameters(model.response_generator.parameters())

    lr_lambda = lambda epoch: opt.learning_rate_decay ** epoch
    schedulerR = torch.optim.lr_scheduler.LambdaLR(optimizer=optimR.optimizer,
                                                  lr_lambda=[lr_lambda])
    optimT = nmt.Optim(opt.optim_method,
                       opt.learning_rate_T,
                       opt.max_grad_norm,
                       opt.learning_rate_decay,
                       opt.weight_decay,
                       opt.start_decay_at)
    optimT.set_parameters(model.template_generator.parameters())
    schedulerT = torch.optim.lr_scheduler.LambdaLR(optimizer=optimT.optimizer,
                                                   lr_lambda=[lr_lambda])

    if critic is not None:
        optimC = nmt.Optim(opt.optim_method,
                       opt.learning_rate_C,
                       opt.max_grad_norm,
                       opt.learning_rate_decay,
                       opt.weight_decay,
                       opt.start_decay_at)
        optimC.set_parameters(critic.parameters())
        schedulerC = torch.optim.lr_scheduler.LambdaLR(optimizer=optimC.optimizer,
                                                       lr_lambda=[lr_lambda])
    else:
        optimC, schedulerC = None, None
    return optimR, schedulerR, optimT, schedulerT, optimC, schedulerC

def check_save_model_path(args, opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    print('saving config file to %s ...'%(opt.out_dir))
    shutil.copy(args.config, os.path.join(opt.out_dir,'config.yml'))


def train_model(opt, model, critic, train_iter, valid_iter, fields, optimR, lr_schedulerR, optimT, lr_schedulerT, optimC, lr_schedulerC, start_epoch_at):
    train_loss = nmt.NMTLossCompute(model.generator, fields['tgt'].vocab)
    valid_loss = nmt.NMTLossCompute(model.generator, fields['tgt'].vocab)

    if use_cuda:
        train_loss = train_loss.cuda()
        valid_loss = valid_loss.cuda()

    shard_size = opt.train_shard_size
    trainer = nmt.Trainer(opt, model,
                        train_iter,
                        valid_iter,
                        train_loss,
                        valid_loss,
                        optimR,
                        shard_size)

    scorer = nmt.Scorer(model, fields['tgt'].vocab, fields['src'].vocab, train_loss, opt)
    num_train_epochs = opt.num_train_epochs
    print('start training...')
    global_step = 0
    for step_epoch in range(start_epoch_at+1, num_train_epochs):

        if step_epoch >= opt.start_decay_at:
            lr_schedulerR.step()
            if lr_schedulerT is not None:
                lr_schedulerT.step()
            if lr_schedulerC is not None:
                lr_schedulerC.step()

        total_stats = Statistics()
        report_stats = Statistics()
        for step_batch, batch in enumerate(train_iter):
            global_step += 1
            if global_step % 6 == -1 % global_step:
                T_turn = False
                C_turn = False
                R_turn = True
            else:
                T_turn = False
                C_turn = False
                R_turn = True

            if C_turn:
                model.template_generator.eval()
                model.response_generator.eval()
                critic.train()
                optimC.optimizer.zero_grad()
                src_inputs, src_lengths = batch.src
                tgt_inputs, tgt_lengths = batch.tgt
                ref_src_inputs, ref_src_lengths = batch.ref_src
                ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
                I_word, I_word_length = batch.I
                D_word, D_word_length = batch.D
                preds, ev = model.template_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, return_ev = True)
                preds = preds.squeeze(2)
                template, template_lengths = model.template_generator.do_mask_and_clean(preds, ref_tgt_inputs, ref_tgt_lengths)

                #x = template.t().data.tolist()
                #vocab = fields['tgt'].vocab 
                #for t in x:
                #    print ("---", ' '.join([vocab.itos[tt] for tt in t]))
                (response, response_length), logp = sample(model.response_generator, src_inputs, None, template, src_lengths, None, template_lengths, max_len = 20)

                enc_embedding = model.response_generator.enc_embedding
                dec_embedding = model.response_generator.dec_embedding
                inds = np.arange(len(tgt_lengths))
                np.random.shuffle(inds)
                inds_tensor = Variable(torch.LongTensor(inds).cuda())
                random_tgt = tgt_inputs.index_select(1, inds_tensor)
                random_tgt_len = [tgt_lengths[i] for i in inds]

                #vocab = fields['tgt'].vocab
                #vocab_src = fields['src'].vocab
                #w = src_inputs.t().data.tolist()
                #x = tgt_inputs.t().data.tolist()
                #y = response.t().data.tolist()
                #z = random_tgt.t().data.tolist()
                #for tw, tx, ty, tz in zip(w, x, y, z):
                #    print (' '.join([vocab_src.itos[tt] for tt in tw]), '|||||', ' '.join([vocab.itos[tt] for tt in tx]), '|||||', ' '.join([vocab.itos[tt] for tt in ty]), '|||||',' '.join([vocab.itos[tt] for tt in tz]))

                x, y, z = critic(enc_embedding(src_inputs),src_lengths,
                                 dec_embedding(tgt_inputs), tgt_lengths,
                                 dec_embedding(response), response_length,
                                 dec_embedding(random_tgt), random_tgt_len
                                 )
                loss = torch.mean(-x)
                #print (loss.data[0])
                loss.backward()
                optimC.step()
                stats = Statistics()
            elif T_turn:
                model.template_generator.train()
                model.response_generator.eval()
                critic.eval()
                stats = scorer.update(batch, optimT, 'T', sample, critic)
            elif R_turn:
                if not ( model.__class__.__name__ == "jointTemplateResponseGenerator"):
                    model.template_generator.eval()
                    model.response_generator.train()
                    critic.eval()
                    if global_step % 2 ==0:
                        stats = trainer.update(batch)
                    else:
                        stats = scorer.update(batch, optimR, 'R', sample, critic)
                else:
                    stats = trainer.update(batch)
            report_stats.update(stats)
            total_stats.update(stats)
            report_func(opt, global_step, step_epoch, step_batch, len(train_iter), total_stats.start_time, optimR.lr, report_stats)

        if critic is not None:
            critic.save_checkpoint(step_epoch, opt, os.path.join(opt.out_dir,"checkpoint_epoch_critic%d.pkl"%step_epoch))
        print('Train perplexity: %g' % total_stats.ppl())

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
    parser.add_argument("-critic_model", type = str, default = None)

    args = parser.parse_args()
    opt = utils.load_hparams(args.config)
    cuda.set_device(args.gpuid[0])
    if opt.random_seed > 0:
        random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        np.random.seed(opt.random_seed)

    fields = dict()
    vocab_src =  Vocab(args.src_vocab, noST = True)
    vocab_tgt = Vocab(args.tgt_vocab)

    fields['src'] = vocab_wrapper(vocab_src)
    fields['tgt'] = vocab_wrapper(vocab_tgt)

    mask_end = True
    train = Data_Loader(args.train_file, opt.train_batch_size, score = args.train_score, mask_end = mask_end)
    valid = Data_Loader(args.valid_file, opt.train_batch_size, mask_end = mask_end)

    # Build model.
    model, critic, start_epoch_at = build_or_load_model(args, opt, fields)
    check_save_model_path(args, opt)

    # Build optimizer.
    optimR, lr_schedulerR, optimT, lr_schedulerT, optimC, lr_schedulerC = build_optims_and_schedulers(model, critic, opt)

    if use_cuda:
        model = model.cuda()
        if opt.use_critic:
            critic = critic.cuda()
    
    # Do training.
    train_model(opt, model, critic, train, valid, fields, optimR, lr_schedulerR, optimT, lr_schedulerT, optimC, lr_schedulerC, start_epoch_at)

if __name__ == '__main__':
    main()
