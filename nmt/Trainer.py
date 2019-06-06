import time
import nmt.utils.misc_utils as utils
import torch
from torch.autograd import Variable
import os
import sys
import math
import torch.nn as nn
import torch.nn.functional as F
from nmt.utils.data_utils import sequence_mask
import numpy as np

class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=1e-12, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return utils.safe_exp(self.loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()

        out_info = ("Epoch %2d, %5d/%5d| acc: %6.2f| ppl: %6.2f| " + \
               "%3.0f tgt tok/s| %4.0f s elapsed") % \
              (epoch, batch, n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_words / (t + 1e-5),
               time.time() - self.start_time)

        print(out_info)
        sys.stdout.flush()

class Scorer(object):
    def __init__(self, model, tgt_vocab, src_vocab, train_loss, opt):
        self.model = model
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        padding_idx = tgt_vocab.stoi[tgt_vocab.PAD]
        weight = torch.ones(len(tgt_vocab))
        weight[padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, reduce=False).cuda()
        self.global_step = 0
        self.train_loss = train_loss
        self.opt = opt

    def score_batch(self, src, tgt, ref_src, ref_tgt, src_lengths, tgt_lengths, ref_src_lengths, ref_tgt_lengths, normalization = False):
        # src : seq_len x batch_size
        self.model.eval()
        model_type = self.model.__class__.__name__
        if model_type == "vanillaNMTModel":
            outputs, attn = self.model(src, tgt[:-1], src_lengths)
        if model_type == "tem_resNMTModel":
            outputs, attn = self.model.response_generator(src, tgt[:-1], ref_tgt, src_lengths, ref_tgt_lengths)


        log_probs = self.model.generator(outputs)

        tgt_out = tgt[1:].view(-1)

        batch_size = src.size(1)


        log_probs =  log_probs.view(-1, log_probs.size(2))

        log_probs = self.criterion(log_probs, tgt_out).view(-1, batch_size).data
        logp = (-torch.sum(log_probs, 0)).tolist()


        if not normalization:
            return outputs, logp


        ret = []
        for lp, l in zip(logp, tgt_lengths):
            ret.append( lp / (l+ 1e-12) )
        return outputs, ret

    def update(self, batch, optim, update_what, sample_func = None, critic = None):
        optim.optimizer.zero_grad()
        src_inputs, src_lengths = batch.src
        tgt_inputs, tgt_lengths = batch.tgt
        ref_src_inputs, ref_src_lengths = batch.ref_src
        ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt
        I_word, I_word_length = batch.I
        D_word, D_word_length = batch.D
        preds, ev = self.model.template_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, return_ev = True)
        preds = preds.squeeze(2)
        template, template_lengths = self.model.template_generator.do_mask_and_clean(preds, ref_tgt_inputs, ref_tgt_lengths)

        if sample_func is None:
            outputs, scores = self.score_batch(src_inputs, tgt_inputs, None, template, src_lengths, tgt_lengths, None, template_lengths, normalization = True)
            avg  = sum(scores)/len(scores)
            scores = [ t-avg for t in scores]
        else:
            (response, response_length), logp = sample_func(self.model.response_generator, src_inputs, None, template, src_lengths, None, template_lengths, max_len = 20, show_sample = False)
            enc_embedding = self.model.response_generator.enc_embedding
            dec_embedding = self.model.response_generator.dec_embedding
            inds = np.arange(len(tgt_lengths))
            np.random.shuffle(inds)
            inds_tensor = Variable(torch.LongTensor(inds).cuda())
            random_tgt = tgt_inputs.index_select(1, inds_tensor)
            random_tgt_len = [tgt_lengths[i] for i in inds]

            vocab = self.tgt_vocab
            vocab_src = self.src_vocab
            w = src_inputs.t().data.tolist()
            x = tgt_inputs.t().data.tolist()
            y = response.t().data.tolist()
            z = random_tgt.t().data.tolist()
            for tw, tx, ty, tz, ww, xx, yy, zz in zip(w, x, y, z, src_lengths, tgt_lengths, response_length, random_tgt_len):
                print (' '.join([vocab_src.itos[tt] for tt in tw[:ww]]), '|||||', ' '.join([vocab.itos[tt] for tt in tx[1:xx-1]]), '|||||', ' '.join([vocab.itos[tt] for tt in ty[1:yy-1]]), '|||||',' '.join([vocab.itos[tt] for tt in tz[1:zz-1]]))

            x, y, z = critic(enc_embedding(src_inputs), src_lengths,
                   dec_embedding(tgt_inputs), tgt_lengths,
                   dec_embedding(response), response_length,
                   dec_embedding(random_tgt), random_tgt_len
                   )
            scores = y.data.tolist()

        if update_what == "R":
            logp = logp.sum(0)
            scores = torch.FloatTensor(scores)
            scores = torch.exp(Variable(scores.cuda()))
            #print (logp, scores)
            loss = -(logp * scores).mean()
            print (loss.data[0])
            loss.backward()
            optim.step()
            stats = Statistics()
            return stats

        ans = torch.ge(preds, 0.5)
        mask = sequence_mask(ref_tgt_lengths).transpose(0, 1)
        weight = torch.FloatTensor(mask.size()).zero_().cuda()
        weight.masked_fill_(mask, 1.)


        for i,x  in enumerate(scores):
            weight[:,i] *= x

        loss = F.binary_cross_entropy(preds, Variable(ans.float().data), weight)

        stats = Statistics() #self.train_loss.monolithic_compute_loss(batch, outputs)
        loss.backward()
        optim.step()
        return stats

    def train(self, epoch, train_iter, optim, report_func):
        total_stats = Statistics()
        report_stats = Statistics()
        for step_batch, batch in enumerate(train_iter):
            self.global_step += 1
            stats = self.update(batch, optim)
            report_stats.update(stats)
            total_stats.update(stats)
            if report_func is not None:
                report_stats = report_func(self.opt, self.global_step,
                                           epoch, step_batch, len(train_iter),
                                           total_stats.start_time, optim.lr, report_stats)
        return total_stats

class Trainer(object):
    def __init__(self, opt, model, train_iter, valid_iter,
                 train_loss, valid_loss, optim, shard_size=32):

        self.opt = opt
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim

        self.shard_size = shard_size

        # Set model in training mode.
        self.model.train()

        self.global_step = 0
        self.step_epoch = 0

    def update(self, batch):
        self.model.zero_grad()
        src_inputs, src_lengths = batch.src
        tgt_inputs = batch.tgt[0][:-1]

        ref_src_inputs, ref_src_lengths = batch.ref_src
        ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt

        model_type = self.model.__class__.__name__
        if model_type == "vanillaNMTModel":
            outputs, attn = self.model(src_inputs, tgt_inputs, src_lengths)
        if model_type == "bivanillaNMTModel":
            outputs, attn = self.model(src_inputs, tgt_inputs, ref_tgt_inputs, src_lengths, ref_tgt_lengths)
        if model_type == "refNMTModel":
            outputs, attn, outputs_f = self.model(src_inputs, tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
        if model_type == "evNMTModel":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            outputs,  attn = self.model(src_inputs, tgt_inputs, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
        if model_type == "responseGenerator":
            outputs, attn = self.model(src_inputs, tgt_inputs, ref_tgt_inputs, src_lengths, ref_tgt_lengths)
        if model_type == "tem_resNMTModel":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            outputs, attn = self.model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths)
        if model_type == "jointTemplateResponseGenerator":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            target, _ = batch.mask

            outputs, attn, preds = self.model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths)
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
            loss.backward(retain_graph = True)
        if batch.score is not None:
            score = Variable(torch.FloatTensor(batch.score)).cuda()
        else:
            score = None

        stats = self.train_loss.sharded_compute_loss(batch, outputs, self.shard_size, weight = score )

        self.optim.step()
        return stats

    def train(self, epoch, report_func=None):
        """ Called for each epoch to train. """
        total_stats = Statistics()
        report_stats = Statistics()

        for step_batch, batch in enumerate(self.train_iter):
            self.global_step += 1
            stats = self.update(batch)

            report_stats.update(stats)
            total_stats.update(stats)

            if report_func is not None:
                report_stats = report_func(self.opt, self.global_step,
                        epoch, step_batch, len(self.train_iter),
                        total_stats.start_time, self.optim.lr, report_stats)


        return total_stats

    def validate(self):
        self.model.eval()
        valid_stats = Statistics()

        for batch in self.valid_iter:
            src_inputs, src_lengths = batch.src
            tgt_inputs = batch.tgt[0][:-1]

            ref_src_inputs, ref_src_lengths = batch.ref_src
            ref_tgt_inputs, ref_tgt_lengths = batch.ref_tgt

            model_type = self.model.__class__.__name__
            if model_type == "vanillaNMTModel":
                outputs, attn = self.model(src_inputs, tgt_inputs, src_lengths)
            if model_type == "bivanillaNMTModel":
                outputs, attn = self.model(src_inputs, tgt_inputs, ref_tgt_input, src_lengths, ref_tgt_lengths)
            if model_type == "refNMTModel":
                outputs, attn, outputs_f = self.model(src_inputs, tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths)
            if model_type == "evNMTModel":
                I_word, I_word_length = batch.I
                D_word, D_word_length = batch.D
                outputs, attn = self.model(src_inputs, tgt_inputs, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
            if model_type == "responseGenerator":
                outputs, attn = self.model(src_inputs, tgt_inputs, ref_tgt_inputs, src_lengths, ref_tgt_lengths)
            if model_type == "tem_resNMTModel":
                I_word, I_word_length = batch.I
                D_word, D_word_length = batch.D
                outputs, attn = self.model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths)
            if model_type == "jointTemplateResponseGenerator":
                I_word, I_word_length = batch.I
                D_word, D_word_length = batch.D
                outputs, attn, preds = self.model(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths)
            stats = self.valid_loss.monolithic_compute_loss(batch, outputs)
            valid_stats.update(stats)
        # Set model back to training mode.
        self.model.train()
        return valid_stats

    def save_per_epoch(self, epoch, out_dir):
        f = open(os.path.join(out_dir,'checkpoint'),'w')
        f.write('latest_checkpoint:checkpoint_epoch%d.pkl'%(epoch))
        f.close()
        self.model.save_checkpoint(epoch, self.opt,
                    os.path.join(out_dir,"checkpoint_epoch%d.pkl"%(epoch)))

    def epoch_step(self, epoch, out_dir):
        """ save ckpt """
        self.save_per_epoch(epoch, out_dir)
