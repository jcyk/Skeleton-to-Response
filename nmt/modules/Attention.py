import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    def __init__(self, dim, attn_type="dot", context_gate = False, gate_vector = False):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")
        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        if self.attn_type == "mlp":
            self.s2s = nn.Linear(dim, dim, bias = False)
            self.t2s = nn.Linear(dim, dim)
            self.v =  nn.Linear(dim, 1, bias = False)

        self.context_gate = context_gate
        self.gate_vector = gate_vector
        if context_gate:
            self.gate_linear = nn.Linear(dim*2, (dim if gate_vector else 1))
        self.linear_out = nn.Linear(dim*2, dim, bias = False)
        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        elif self.attn_type == "mlp":
            _t = self.t2s(h_t.view(tgt_batch*tgt_len, tgt_dim))
            _t = _t.view(tgt_batch, tgt_len, 1, tgt_dim)
            _t = _t.expand(tgt_batch, tgt_len, src_len, tgt_dim)

            _s = self.s2s(h_s.view(src_batch*src_len, src_dim))
            _s = _s.view(src_batch, 1, src_len, src_dim)
            _s = _s.expand(src_batch, tgt_len, src_len, src_dim)

            #print (_t.size(), _s.size())
            return self.v(self.tanh(_t+_s).view(-1, tgt_dim)).view(tgt_batch, tgt_len, src_len)


    def forward(self, input, context, context_values = None, mask = None):

        # print (input.size(), context.size(), mask.size())
        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False


        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        #print (sourceL, targetL)
        # compute attention scores, as in Luong et al.
        align = self.score(input, context)
        if mask is not None:
            mask = mask.unsqueeze(1)
            align.data.masked_fill_(1 - mask, -float('inf'))
        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
        #print (align_vectors)
        # each context vector c_t is the weighted average
        # over all the source hidden states
        if context_values is not None:
            c = torch.bmm(align_vectors, context_values)
        else:
            c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        if self.context_gate:
            gates = F.sigmoid(self.gate_linear(concat_c).view(batch, targetL, (dim if self.gate_vector else 1)))
            attn_h = (1-gates) * c + gates * input
            attn_h = attn_h.view(batch, targetL, dim)
        else:
            attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
        return attn_h, align_vectors
