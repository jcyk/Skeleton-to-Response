import torch
import torch.nn as nn
from nmt.modules.Attention import GlobalAttention
# from nmt.modules.SRU import SRU
from nmt.modules.StackedRNN import StackedGRU, StackedLSTM
import torch.nn.functional as F
from torch.autograd import Variable
import math

class DecoderBase(nn.Module):
    def forward(self, input, context, state):
        """
        Forward through the decoder.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
        Returns:
            outputs (FloatTensor): a Tensor sequence of output from the decoder
                                   of shape (len x batch x hidden_size).
            state (FloatTensor): final hidden state from the decoder.
            attns (dict of (str, FloatTensor)): a dictionary of different
                                type of attention Tensor from the decoder
                                of shape (src_len x batch).
        """
        raise NotImplementedError

class KVAttnDecoderRNN(DecoderBase):
    def __init__(self, rnn_type, attn_type, input_size,
                hidden_size, num_layers=1, dropout=0.1, src_attention = False, mem_gate = False, gate_vector = False, return_original = False):
        super(KVAttnDecoderRNN, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout)

        self.src_attention = src_attention
        if src_attention:
            self.src_attn = GlobalAttention(hidden_size, attn_type)
        self.mem_attn = GlobalAttention(hidden_size, "general", context_gate= mem_gate, gate_vector = gate_vector)
        self.return_original = return_original

    def forward(self, input, context_keys, context_values, state, mem_mask =None, src_context = None, src_mask = None):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)

        if self.src_attention:
            originals, src_score = self.src_attn(
                rnn_outputs.transpose(0, 1).contiguous(),
                src_context.transpose(0, 1),
                mask = src_mask
            )
        else:
            orginals = rnn_outputs
        original = self.dropout(originals)

        if not self.return_original:
            rnn_outputs = originals
        # Calculate the attention.
        attn_outputs, attn_scores = self.mem_attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context_keys.transpose(0, 1),
                context_values.transpose(0, 1),                  # (contxt_len, batch, d)
                mask = mem_mask
        )
        outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)
        if self.return_original:
            return outputs, hidden, attn_scores, originals
        return outputs , hidden, attn_scores


class AttnDecoderRNN(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attn_type, input_size, 
                hidden_size, num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout)

        if self.attn_type != 'none':
            self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, input, context, state, attn_mask = None):
        emb = input
        rnn_outputs, hidden = self.rnn(emb, state)

        if self.attn_type != 'none':
            # Calculate the attention.
            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1),                   # (contxt_len, batch, d)
                mask = attn_mask
            )

            outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)
            attn = attn_scores
        else:
            outputs  = self.dropout(rnn_outputs)
            attn = None

        return outputs , hidden, attn

class AuxDecoderRNN(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attn_type, input_size,
                hidden_size, num_layers=1, dropout=0.1):
        super(AuxDecoderRNN, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout)

        if self.attn_type != 'none':
            self.attn = GlobalAttention(hidden_size, attn_type)

    def forward(self, input, context, state, aux, attn_mask = None):
        emb = input
        aux_input = aux.unsqueeze(0).repeat(emb.size()[0], 1, 1)

        emb = torch.cat([emb, aux_input], 2)
        rnn_outputs, hidden = self.rnn(emb, state)

        if self.attn_type != 'none':
            # Calculate the attention.
            attn_outputs, attn_scores = self.attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                context.transpose(0, 1),                   # (contxt_len, batch, d)
                mask = attn_mask
            )

            outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)
            attn = attn_outputs
        else:
            outputs  = self.dropout(rnn_outputs)
            attn = None
        return outputs , hidden, attn

class AuxMemDecoderRNN(DecoderBase):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, rnn_type, attn_type, input_size,
                 hidden_size, num_layers=1, dropout=0.1, src_attention = False, mem_gate = False, gate_vector = False):
        super(AuxMemDecoderRNN, self).__init__()
        # Basic attributes.
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout)

        self.src_attention = src_attention
        if src_attention:
            self.src_attn = GlobalAttention(hidden_size, attn_type)
        self.mem_attn = GlobalAttention(hidden_size, "general", context_gate= mem_gate, gate_vector = gate_vector)

    def forward(self, input, mem_context, state, aux, mem_mask =None, src_context = None, src_mask = None):
        emb = input
        aux_input = aux.unsqueeze(0).repeat(emb.size()[0], 1, 1)

        emb = torch.cat([emb, aux_input], 2)

        rnn_outputs, hidden = self.rnn(emb, state)

        if self.src_attention:
            originals, src_score = self.src_attn(
                rnn_outputs.transpose(0, 1).contiguous(),
                src_context.transpose(0, 1),
                mask = src_mask
            )
        else:
            orginals = rnn_outputs
        rnn_outputs = self.dropout(originals)

        # Calculate the attention.
        attn_outputs, attn_scores = self.mem_attn(
                rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
                mem_context.transpose(0, 1),
                mask = mem_mask
        )
        outputs  = self.dropout(attn_outputs)    # (input_len, batch, d)
        return outputs , hidden, attn_scores
