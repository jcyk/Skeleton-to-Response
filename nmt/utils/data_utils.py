import nmt
from torch.autograd import Variable
import torch


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    lengths = torch.LongTensor(lengths).cuda()
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

# Pad a with the PAD symbol

def pad_seq(seq, max_length, padding_idx):
    seq += [padding_idx for i in range(max_length - len(seq))]
    return seq


def seq2indices(seq, word2index, max_len=None):
    seq_idx = []
    words_in = seq.split(' ')
    if max_len is not None:
        words_in = words_in[:max_len]
    for w in words_in:
        seq_idx.append(word2index[w])

    return seq_idx


def indices2words(idxs, index2word):
    words_list = [index2word[idx] for idx in idxs]
    return words_list
