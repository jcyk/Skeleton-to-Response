import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np

class EncoderBase(nn.Module):
    """
    EncoderBase class for sharing code among various encoder.
    """

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                    encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        raise NotImplementedError


class EncoderRNN(EncoderBase):
    """ The standard RNN encoder. """
    def __init__(self, rnn_type, input_size, 
                hidden_size, num_layers=1, 
                dropout=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional)

    def forward(self, input, lengths, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        emb = input
        is_sorted = lambda a: np.all(np.array(a[:-1]) >= np.array(a[1:]))
        packed_emb = emb
        changedorder = False
        if lengths is not None:
            # Lengths data is wrapped inside a Variable.
            if not is_sorted(lengths):
                inds = (np.argsort(lengths)[::-1]).copy()
                inds_tensor = Variable(torch.LongTensor(inds).cuda())
                emb = emb.index_select(1, inds_tensor)
                len_sub = list(np.array(lengths)[inds])
                packed_emb = pack(emb, len_sub)
                changedorder = True
            else:
                packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None:
            outputs = unpack(outputs)[0]

        if self.bidirectional:
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
            if self.rnn_type != 'LSTM':
                hidden_t = torch.cat([hidden_t[0:hidden_t.size(0):2], hidden_t[1:hidden_t.size(0):2]], 2)
            else:
                h_n, c_n = hidden_t
                h_n = torch.cat([h_n[0:h_n.size(0):2], h_n[1:h_n.size(0):2]], 2)
                c_n = torch.cat([c_n[0:c_n.size(0):2], c_n[1:c_n.size(0):2]], 2)
                hidden_t = (h_n, c_n)
        if changedorder:
            rinds = np.argsort(inds)
            rinds_tensor = Variable(torch.LongTensor(rinds).cuda())
            outputs = outputs.index_select(1, rinds_tensor)
            if self.rnn_type == 'LSTM':
                h_n, c_n = hidden_t
                hidden_t = (h_n.index_select(1, rinds_tensor), c_n.index_select(1, rinds_tensor))
            else:
                hidden_t = hidden_t.index_select(1, rinds_tensor)
        return outputs, hidden_t
