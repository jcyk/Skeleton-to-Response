# -*- coding: UTF-8 -*-
import math
import torch
from torch.autograd import Variable
import random
PAD, EOS, EOT, UNK = '<BLANK>', '<eos>', '<eot>', '<unk>'
PAD_idx, EOS_idx, EOT_idx, UNK_idx = 0, 1, 2, 3

class Vocab(object):
    def __init__(self, vocab_file, noST = False):
        with open(vocab_file, encoding="utf-8") as f:
            if noST:
                self.itos = [PAD] + [ token.strip() for token in f.readlines() ]
            else:
                self.itos = [PAD, EOS, EOT] + [ token.strip() for token in f.readlines() ]
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self.PAD = PAD
        #assert (self.itos[3] == UNK), (self.itos[3], "first word must be <unk>")
        self.UNK = UNK
        self.EOS = EOS
        self.EOT = EOT

    def i2s(self, idx):
        return self.itos[idx]

    def __len__(self):
        return len(self.itos)

def ListsToTensor(xs, tgt = False):
    batch_size = len(xs)
    lens = [ len(x)+(2 if tgt else 0) for x in xs]
    mx_len = max( max(lens),1)
    ys = []
    for i, x in enumerate(xs):
        y = ([EOS_idx] if tgt else [] )+ x + ([EOT_idx] if tgt else []) + ([PAD_idx]*(mx_len - lens[i]))
        ys.append(y)

    lens = [ max(1, x) for x in lens]
    data = Variable(torch.LongTensor(ys).t_())

    data = data.cuda()

    return (data, lens)

def LCS_mask(src, tgt, stop_words):
    m = len(src)
    n = len(tgt)
    if stop_words is None:
        stop_words = set()
    mat = [[0] * (n+1) for row in range(m+1)]
    for row in range(1, m+1):
        for col in range(1, n+1):
            if src[row - 1] == tgt[col - 1] and (src[row-1] not in stop_words):
                mat[row][col] = mat[row - 1][col - 1] + 1
            else:
                mat[row][col] = max(mat[row][col - 1], mat[row - 1][col])
    x,y = m,n
    mask = []
    while y >0 and x >0:
        if mat[x][y] == mat[x-1][y-1] + 1:
            x -=1
            y -=1
            mask.append(1)
        elif mat[x][y] == mat[x][y-1]:
            y -= 1
            mask.append(0)
        else:
            x -= 1
    while y>0:
        y -= 1
        mask.append(0)
    return mask[::-1]

class Batch(object):
    def __init__(self, qs, rs, rqs, rrs, ss, mask_end = False, stop_words = None):
        if mask_end:
            I, D, ref_tgt, mask,ref_src = [], [], [], [], []
            for q ,r,rq,rr in zip(qs, rs, rqs, rrs):
                rq = rq[0]
                rr = rr[0]
                q_set = set(q)
                rq_set = set(rq)
                ins = q_set&rq_set
                I.append( list(q_set - ins))
                D.append( list(rq_set - ins))
                ref_tgt.append(rr)
                ref_src.append(rq)
                mask.append(LCS_mask(r, rr, stop_words))
            self.I = ListsToTensor(I)
            self.D = ListsToTensor(D)
            self.ref_tgt = ListsToTensor(ref_tgt)
            self.mask = ListsToTensor(mask)
            self.src = ListsToTensor(qs)
            self.tgt = ListsToTensor(rs, tgt=True)
            self.ref_src = ListsToTensor(ref_src)
            self.batch_size = len(qs)
            self.score = ss
            return
        self.src = ListsToTensor(qs)
        self.tgt = ListsToTensor(rs, tgt=True)
        num_rq = max( 1, max(len(rq) for rq in rqs))
        num_rr = max( 1, max(len(rr) for rr in rrs))
        ref_src =[ ListsToTensor([
                                        rq[i] if i < len(rq) else [PAD_idx]
                                        for rq in rqs])
                       for i in range(num_rq)]
        ref_tgt =[ ListsToTensor([
                                        rr[i] if i < len(rr) else [PAD_idx]
                                        for rr in rrs], tgt = True)
                       for i in range(num_rr)]
        self.ref_src = ([x[0] for x in ref_src[:5]], [x[1] for x in ref_src[:5]])
        self.ref_tgt = ([x[0] for x in ref_tgt[:5]], [x[1] for x in ref_tgt[:5]])
        self.score = ss
        self.batch_size = len(qs)

class Data_Loader(object):
    def __init__(self, fname, batch_size = 32, train = True, score = None, mask_end = False, stop_words = None):
        all_q = []
        all_r = []
        all_rq = []
        all_rr = []
        with open(fname) as f:
            for line in f.readlines():
                x = line.split('|')
                q, r = x[:2]
                q = [int(x) for x in q.split()]
                r = [int(x) for x in r.split()]
                if train and (len(q) <= 2 or len(r)<=2):
                    continue
                q = q[:30]
                r = r[:30]
                all_q.append(q)
                all_r.append(r)
                rq, rr  = [], []
                for q,r in zip(x[2::2], x[3::2]):
                    q = [int(x) for x in q.split()]
                    r = [int(x) for x in r.split()]
                    q = q[:30]
                    r = r[:30]
                    rq.append(q)
                    rr.append(r)
                all_rq.append(rq)
                all_rr.append(rr)

        self.all_s = None
        if score:
            self.all_s = [float(x.strip()) for x in open(score).readlines()]
        self.batch_size = batch_size
        self.all_q = all_q
        self.all_r = all_r
        self.all_rq = all_rq
        self.all_rr= all_rr
        self.train = train
        self.mask_end = mask_end

        if stop_words is not None:
            stop_words = set( [ int(x.strip()) for x in open(stop_words).readlines()])
            self.stop_words = stop_words
        else:
            self.stop_words = None
    def __len__(self):
        return math.ceil(len(self.all_q)/self.batch_size)

    def __iter__(self):
        idx = list(range(len(self.all_q)))
        if self.train:
            random.shuffle(idx)
        cur = 0
        while cur < len(idx):
            batch = idx[cur:cur + self.batch_size]
            cur_q = [self.all_q[x] for x in batch]
            cur_r = [self.all_r[x] for x in batch]
            cur_rq = [self.all_rq[x] for x in batch]
            cur_rr = [self.all_rr[x] for x in batch]
            if self.all_s is not None:
                cur_s = [self.all_s[x] for x in batch]
            else:
                cur_s = None
            yield Batch(cur_q, cur_r, cur_rq, cur_rr, cur_s, self.mask_end, self.stop_words)
            cur += self.batch_size
        raise StopIteration
