import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nmt.utils.data_utils import sequence_mask

def ListsToTensor(xs):
    batch_size = len(xs)
    lens = [ len(x)  for x in xs]
    mx_len = max( max(lens),1)
    ys = []
    for i, x in enumerate(xs):
        y =  x + ([0]*(mx_len - lens[i]))
        ys.append(y)

    lens = [ max(1, x) for x in lens]
    data = Variable(torch.LongTensor(ys).t_())

    data = data.cuda()

    return (data, lens)

class editVectorGenerator(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder_ref, attention_src, attention_ref, dropout):
        super(editVectorGenerator, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder_ref = encoder_ref
        self.attention_src = attention_src
        self.attention_ref = attention_ref
        self.dropout = dropout

    def forward(self, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths):

        enc_outputs, enc_hidden = self.encoder_ref(self.dec_embedding(ref_tgt_inputs), ref_tgt_lengths, None)
        I_context = self.enc_embedding(I_word)
        D_context = self.enc_embedding(D_word)
        enc_hidden = enc_hidden.squeeze(0)

        I_context = self.dropout(I_context)
        D_context = self.dropout(D_context)
        enc_hidden = self.dropout(enc_hidden)

        I_context = I_context.transpose(0, 1).contiguous()
        D_context = D_context.transpose(0, 1).contiguous()

        I, _ = self.attention_src(enc_hidden, I_context, mask = sequence_mask(I_word_length))
        D, _ = self.attention_ref(enc_hidden, D_context, mask = sequence_mask(D_word_length))

        return torch.cat([I,D], 1), enc_outputs

class jointTemplateResponseGenerator(nn.Module):
    def __init__(self, ev_generator, masker, masker_dropout, enc_embedding, dec_embedding, encoder_src, decoder, generator, bridge, fields):
        super(jointTemplateResponseGenerator, self).__init__()
        self.ev_generator = ev_generator
        self.masker = masker
        self.masker_dropout = masker_dropout
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder_src = encoder_src
        self.decoder = decoder
        self.generator = generator
        self.bridge = bridge
        self.fields = fields


    def forward(self, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths):
        ref_contexts, enc_hidden, ref_mask, dist, src_contexts, src_mask, preds = self.encode(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, src_lengths)

        dec_init_hidden = self.init_decoder_state(enc_hidden, ref_contexts)

        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs,  ref_contexts, dec_init_hidden, dist, ref_mask, src_contexts, src_mask
            )
        return dec_outputs, attn, preds


    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def encode(self, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, src_lengths):
        ev, enc_outputs = self.ev_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
        ev = self.masker_dropout(ev)
        ev_for_return = ev
        enc_outputs = self.masker_dropout(enc_outputs)
        _, _dim = ev.size()
        _len, _batch, _ = enc_outputs.size()

        if self.bridge is not None:
            dist = self.bridge(ev)
        else:
            dist = None

        ev = ev.unsqueeze(0)
        ev = ev.expand(_len, _batch, _dim)
        preds = self.masker(torch.cat([ev, enc_outputs], 2))
        preds = preds.squeeze(2)

        emb_src = self.enc_embedding(src_inputs)
        src_contexts, enc_hidden = self.encoder_src(emb_src, src_lengths, None)

        ref_mask = sequence_mask(ref_tgt_lengths)
        src_mask = sequence_mask(src_lengths)
        return enc_outputs, enc_hidden, ref_mask, dist, src_contexts, src_mask, preds

    def decode(self, input, context, state, dist, context_mask, src_context, src_context_mask):
        emb = self.dec_embedding(input)
        if dist is not None:
            dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state, dist, context_mask, src_context, src_context_mask)
        else:
            dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, context, state, context_mask, src_context, src_context_mask)

        return dec_outputs, dec_hiddens, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({ 'ev_generator_dict': self.ev_generator.state_dict(),
                    'masker_dict': self.masker.state_dict(),
                    'masker_dropout_dict': self.masker_dropout.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'encoder_src_dict': self.encoder_src.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'epoch': epoch,
                    'opt': opt,
        #            'bridge_dict': self.bridge.state_dict()
                    }, filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.ev_generator.load_state_dict(ckpt['ev_generator_dict'])
        self.masker.load_state_dict(ckpt['masker_dict'])
        self.masker_dropout.load_state_dict(ckpt['masker_dropout_dict'])
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder_src.load_state_dict(ckpt['encoder_src_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        if self.bridge is not None:
            self.bridge.load_state_dict(ckpt['bridge_dict'])
        epoch = ckpt['epoch']
        return epoch

class templateGenerator(nn.Module):
    def __init__(self, ev_generator, masker, dropout):
        super(templateGenerator, self).__init__()
        self.ev_generator = ev_generator
        self.masker = masker
        self.dropout = dropout

    def forward(self, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, return_ev = False):
        ev, enc_outputs = self.ev_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
        ev = self.dropout(ev)
        ev_for_return = ev
        enc_outputs = self.dropout(enc_outputs)
        _, _dim = ev.size()
        _len, _batch, _ = enc_outputs.size()
        ev = ev.unsqueeze(0)
        ev = ev.expand(_len, _batch, _dim)
        preds = self.masker(torch.cat([ev, enc_outputs], 2))
        if return_ev:
            return preds, ev_for_return
        return preds

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'ev_generator_dict': self.ev_generator.state_dict(),
                    'masker_dict': self.masker.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.ev_generator.load_state_dict(ckpt['ev_generator_dict'])
        self.masker.load_state_dict(ckpt['masker_dict'])
        epoch = ckpt['epoch']
        return epoch

    def do_mask_and_clean(self, preds, ref_tgt_inputs, ref_tgt_lengths):
        mask = sequence_mask(ref_tgt_lengths).transpose(0, 1).float()
        ans = torch.ge(preds, 0.5)
        ref_tgt_inputs.data.masked_fill_(1-ans.data, 0)
        y = ref_tgt_inputs.transpose(0, 1).data.tolist()
        data = [ z[:l] for z,l in zip(y, ref_tgt_lengths) ]
        new_data = []
        for z in data:
            new_z = []
            iszero = False
            for w in z:
                if iszero and w == 0:
                    continue
                else:
                    new_z.append(w)
                iszero = (w==0)
            new_data.append([1] + new_z+ [2])
        return ListsToTensor(new_data)

class responseGenerator(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder_src, decoder, generator, encoder_ref, bridge, fields):
        super(responseGenerator, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder_src = encoder_src
        self.decoder = decoder
        self.generator = generator
        self.encoder_ref = encoder_ref
        self.bridge = bridge
        self.fields = fields

    def forward(self, src_inputs, tgt_inputs, template_inputs, src_lengths, template_lengths, ev = None):
        # Run words through encoder 
        ref_contexts, enc_hidden, ref_mask, dist, src_contexts, src_mask = self.encode(src_inputs, template_inputs, src_lengths, template_lengths, ev)
        dec_init_hidden = self.init_decoder_state(enc_hidden, ref_contexts)
        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs,  ref_contexts, dec_init_hidden, dist, ref_mask, src_contexts, src_mask
            )
        return dec_outputs, attn

    def encode(self, src_inputs, template_inputs, src_lengths, template_lengths, ev = None):
        emb_src = self.enc_embedding(src_inputs)
        src_contexts, enc_hidden = self.encoder_src(emb_src, src_lengths, None)
        if ev is not None and self.bridge is not None:
            dist = self.bridge(ev)
        else:
            dist = None

        ref_contexts, ref_mask = [], []
        for template_input, template_length in zip(template_inputs, template_lengths):
            emb_ref = self.dec_embedding(template_input)
            ref_context, _ = self.encoder_ref(emb_ref, template_length)
            ref_mask_ = sequence_mask(template_length)
            ref_contexts.append(ref_context)
            ref_mask.append(ref_mask_)
        ref_contexts = torch.cat(ref_contexts, 0)
        ref_mask = torch.cat(ref_mask, 1)
        src_mask = sequence_mask(src_lengths)
        return ref_contexts, enc_hidden, ref_mask, dist, src_contexts, src_mask

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context, state, dist, context_mask, src_context, src_context_mask):
        emb = self.dec_embedding(input)
        if dist is not None:
            dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state, dist, context_mask, src_context, src_context_mask)
        else:
            dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, context, state, context_mask, src_context, src_context_mask)

        return dec_outputs, dec_hiddens, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_src_dict': self.encoder_src.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'encoder_ref_dict': self.encoder_ref.state_dict(),
                    #'bridge_dict': self.bridge.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                    filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.encoder_src.load_state_dict(ckpt['encoder_src_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        self.encoder_ref.load_state_dict(ckpt['encoder_ref_dict'])
        #self.bridge.load_state_dict(ckpt['bridge_dict'])
        epoch = ckpt['epoch']
        return epoch

class tem_resNMTModel(nn.Module):

    def __init__(self, template_generator, response_generator, use_ev):
        super(tem_resNMTModel, self).__init__()
        self.template_generator = template_generator
        self.response_generator = response_generator
        self.generator = self.response_generator.generator
        self.use_ev = use_ev

    def forward(self, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, src_inputs, tgt_inputs, src_lengths):
        preds, ev = self.template_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths, return_ev = True)
        preds = preds.squeeze(2)
        ev = ev.detach()
        template_inputs, template_lengths = self.template_generator.do_mask_and_clean(preds, ref_tgt_inputs, ref_tgt_lengths)

        dec_outputs, attn = self.response_generator(src_inputs, tgt_inputs, template_inputs, src_lengths, template_lengths, ev = ( None if not self.use_ev else ev) )
        return dec_outputs, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'template_generator_dict': self.template_generator.state_dict(),
                    'response_generator_dict': self.response_generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.template_generator.load_state_dict(ckpt['template_generator_dict'])
        self.response_generator.load_state_dict(ckpt['response_generator_dict'])
        epoch = ckpt['epoch']
        return epoch

class vanillaNMTModel(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder, decoder, generator):
        super(vanillaNMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src_inputs, tgt_inputs, src_lengths):

        # Run wrods through encoder

        enc_outputs, enc_hidden, enc_mask = self.encode(src_inputs, src_lengths, None)

        dec_init_hidden = self.init_decoder_state(enc_hidden, enc_outputs)

        dec_outputs, dec_hiddens, attn = self.decode(
                tgt_inputs, enc_outputs, dec_init_hidden, enc_mask
            )

        return dec_outputs, attn

    def encode(self, input, lengths=None, hidden=None):
        emb = self.enc_embedding(input)
        enc_outputs, enc_hidden = self.encoder(emb, lengths, None)
        enc_mask = sequence_mask(lengths)
        return enc_outputs, enc_hidden, enc_mask

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context, state, mask):
        emb = self.dec_embedding(input)

        dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state, mask
            )

        return dec_outputs, dec_hiddens, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'decoder_rnn_dict' : self.decoder.rnn.state_dict(),
                    'decoder_attn_dict': self.decoder.attn.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder.load_state_dict(ckpt['encoder_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        epoch = ckpt['epoch']
        return epoch

class bivanillaNMTModel(nn.Module):
    def __init__(self, enc_embedding, enc_embedding_ref, dec_embedding, encoder, encoder_ref, decoder, generator):
        super(bivanillaNMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.enc_embedding_ref = enc_embedding_ref
        self.encoder = encoder
        self.encoder_ref = encoder_ref
        self.decoder = decoder
        self.generator = generator
        self.bridge_h = nn.Linear(2*encoder.hidden_size + 2*encoder_ref.hidden_size, 2*encoder.hidden_size)
        self.bridge_c = nn.Linear(2*encoder.hidden_size + 2*encoder_ref.hidden_size, 2*encoder.hidden_size)

    def forward(self, src_inputs, tgt_inputs, ref_tgt_inputs, src_lengths, ref_tgt_lengths):

        # Run words through encoder

        enc_outputs, enc_hidden = self.encode(src_inputs, ref_tgt_inputs, src_lengths, ref_tgt_lengths, None)

        dec_init_hidden = self.init_decoder_state(enc_hidden, enc_outputs)

        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs, enc_outputs, dec_init_hidden
            )

        return dec_outputs, attn



    def encode(self, src, ref_tgt, src_lengths=None, ref_tgt_lengths = None, hidden=None):
        emb = self.enc_embedding(src)
        emb_ref = self.enc_embedding_ref(ref_tgt)
        enc_outputs, enc_hidden = self.encoder(emb, src_lengths, None)
        enc_outputs_x, enc_hidden_x = self.encoder_ref(emb_ref, ref_tgt_lengths, None)

        h = torch.cat([enc_hidden[0], enc_hidden_x[0]], -1)
        c = torch.cat([enc_hidden[1], enc_hidden_x[1]], -1)
        h = self.bridge_h(h)
        c = self.bridge_c(c)
        return enc_outputs, (h, c)

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context, state):
        emb = self.dec_embedding(input)
        dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context, state
            )

        return dec_outputs, dec_hiddens, attn
    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_dict': self.encoder.state_dict(),
                    'encoder_ref_dict': self.encoder_ref.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    "enc_embedding_ref":self.enc_embedding_ref.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'bridge_h_dict': self.bridge_h.state_dict(),
                    'bridge_c_dict': self.bridge_c.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.enc_embedding_ref.load_state_dict(ckpt['enc_embedding_ref'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder.load_state_dict(ckpt['encoder_dict'])
        self.encoder_ref.load_state_dict(ckpt['encoder_ref_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        self.bridge_h.load_state_dict(ckpt['bridge_h_dict'])
        self.bridge_c.load_state_dict(ckpt['bridge_c_dict'])
        epoch = ckpt['epoch']
        return epoch

class refNMTModel(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder_src, encoder_ref, decoder_ref, decoder, generator, fields):
        super(refNMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder_src = encoder_src
        self.encoder_ref = encoder_ref
        self.decoder_ref = decoder_ref
        self.decoder = decoder
        self.generator = generator
        self.fields = fields

    def forward(self, src_inputs, tgt_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths):

        # Run words through encoder
        ref_values, enc_hidden, ref_keys, ref_mask, src_context, src_mask = self.encode(src_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths, None)

        dec_init_hidden = self.init_decoder_state(enc_hidden, ref_values)

        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs, ref_keys, ref_values, dec_init_hidden, ref_mask, src_context, src_mask
            )
        return dec_outputs, attn, ref_keys

    def encode(self, src_inputs, ref_src_inputs, ref_tgt_inputs, src_lengths, ref_src_lengths, ref_tgt_lengths, hidden=None):
        emb_src = self.enc_embedding(src_inputs)
        embs_ref_src = [ self.enc_embedding(ref_src_input) for ref_src_input in ref_src_inputs ]
        embs_ref_tgt = [ self.dec_embedding(ref_tgt_input) for ref_tgt_input in ref_tgt_inputs ]

        ref_values, ref_keys, ref_mask = [], [], []
        for emb_ref_src, emb_ref_tgt, ref_src_length, ref_tgt_length in zip(embs_ref_src, embs_ref_tgt, ref_src_lengths, ref_tgt_lengths):
            ref_src_context, enc_ref_hidden = self.encoder_src(emb_ref_src, ref_src_length, None)
            ref_src_mask = sequence_mask(ref_src_length)
            ref_key, _, _ = self.decoder_ref(emb_ref_tgt, ref_src_context, enc_ref_hidden, ref_src_mask)
            ref_value, _ = self.encoder_ref(emb_ref_tgt, ref_tgt_length, None)
            ref_msk  = sequence_mask([ x-1 for x in ref_tgt_length])
            ref_values.append(ref_value[1:])
            ref_keys.append(ref_key[:-1])
            ref_mask.append(ref_msk)
        ref_values = torch.cat(ref_values, 0)
        ref_keys = torch.cat(ref_keys, 0)
        ref_mask =  torch.cat(ref_mask, 1)

        src_context, enc_hidden = self.encoder_src(emb_src, src_lengths, None)
        src_mask = sequence_mask(src_lengths)

        return ref_values, enc_hidden, ref_keys, ref_mask, src_context, src_mask

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input, context_key, context_value, state, context_mask, src_context, src_mask):
        emb = self.dec_embedding(input)
        dec_outputs , dec_hiddens, attn = self.decoder(
                emb, context_key, context_value, state, context_mask, src_context, src_mask
            )
        return dec_outputs, dec_hiddens, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_src_dict': self.encoder_src.state_dict(),
                    'encoder_ref_dict': self.encoder_ref.state_dict(),
                    "decoder_ref_dict": self.decoder_ref.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder_src.load_state_dict(ckpt['encoder_src_dict'])
        self.encoder_ref.load_state_dict(ckpt['encoder_ref_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.decoder_ref.load_state_dict(ckpt['decoder_ref_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        epoch = ckpt['epoch']
        return epoch

class evNMTModel(nn.Module):
    def __init__(self, enc_embedding, dec_embedding, encoder_src, decoder, generator, ev_generator, bridge, fields):
        super(evNMTModel, self).__init__()
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.encoder_src = encoder_src
        self.ev_generator = ev_generator
        self.decoder = decoder
        self.generator = generator
        self.bridge = bridge
        self.fields = fields

    def forward(self, src_inputs, tgt_inputs, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths):

        # Run words through encoder
        ref_contexts, enc_hidden, ref_mask, dist = self.encode(src_inputs, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)

        dec_init_hidden = self.init_decoder_state(enc_hidden, ref_contexts)
        dec_outputs , dec_hiddens, attn = self.decode(
                tgt_inputs, ref_contexts, dec_init_hidden, dist, ref_mask
        )
        return dec_outputs, attn

    def encode(self, src_inputs, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths):
        emb_src = self.enc_embedding(src_inputs)
        _, enc_hidden = self.encoder_src(emb_src, src_lengths, None)
        ev, ref_contexts = self.ev_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt_inputs, ref_tgt_lengths)
        dist = self.bridge(ev)
        ref_mask = sequence_mask(ref_tgt_lengths)
        return ref_contexts, enc_hidden, ref_mask, dist

    def init_decoder_state(self, enc_hidden, context):
        return enc_hidden

    def decode(self, input,context, state, dist, context_mask):
        emb = self.dec_embedding(input)
        dec_outputs , dec_hiddens, attn = self.decoder(
                emb,  context ,state, dist, context_mask
            )
        return dec_outputs, dec_hiddens, attn

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({'encoder_src_dict': self.encoder_src.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'enc_embedding_dict': self.enc_embedding.state_dict(),
                    'encoder_ref_dict': self.ev_generator.encoder_ref.state_dict(),
                    'attention_src_dict': self.ev_generator.attention_src.state_dict(),
                    'attention_ref_dict': self.ev_generator.attention_ref.state_dict(),
         #           'bridge_dict': self.bridge.state_dict(),
                    'dec_embedding_dict': self.dec_embedding.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    },
                   filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.enc_embedding.load_state_dict(ckpt['enc_embedding_dict'])
        self.dec_embedding.load_state_dict(ckpt['dec_embedding_dict'])
        self.encoder_src.load_state_dict(ckpt['encoder_src_dict'])
        self.decoder.load_state_dict(ckpt['decoder_dict'])
        self.generator.load_state_dict(ckpt['generator_dict'])
        #self.bridge.load_state_dict(ckpt['bridge_dict'])
        self.ev_generator.encoder_ref.load_state_dict(ckpt['encoder_ref_dict'])
        self.ev_generator.attention_src.load_state_dict(ckpt['attention_src_dict'])
        self.ev_generator.attention_ref.load_state_dict(ckpt['attention_ref_dict'])
        epoch = ckpt['epoch']
        return epoch

class Critic(nn.Module):
    def __init__(self, encoder_src, encoder_tgt, dropout):
        super(Critic, self).__init__()
        self.encoder_src = encoder_src
        self.encoder_tgt = encoder_tgt
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(encoder_src.hidden_size*2, encoder_tgt.hidden_size*2)
        self.log_softmax = nn.LogSoftmax(dim = -1)

    def forward(self, src_inputs_emb, src_lengths, *args):
        _, src = self.encoder_src(src_inputs_emb, src_lengths)
        assert len(args)%2==0
        ret = []
        src = torch.squeeze(src[-1], 0)
        src = self.linear_out(self.dropout(src))
        for input_emb, lengths in zip(args[::2], args[1::2]):
            _, tgti = self.encoder_tgt(input_emb, lengths)
            tgti = torch.squeeze(tgti[-1], 0)
            tgti = self.dropout(tgti)
            score = torch.sum(src * tgti, 1)
            ret.append(score)
        cat = torch.stack(ret, 1)
        logp = self.log_softmax(cat)


        _, max_i = torch.max(logp, 1)
        print ('-',torch.mean(torch.eq(max_i, 0).float()).data[0])
        print ('-',torch.mean(torch.eq(max_i, 1).float()).data[0])
        print ('-',torch.mean(torch.eq(max_i, 2).float()).data[0])
        x, y, z =  torch.split(logp, 1, -1)
        return torch.squeeze(x, 1), torch.squeeze(y, 1), torch.squeeze(z, 1)

    def save_checkpoint(self, epoch, opt, filename):
        torch.save({
            'encoder_src_dict': self.encoder_src.state_dict(),
            'encoder_tgt_dict': self.encoder_tgt.state_dict(),
            'linear_out_dict': self.linear_out.state_dict(),
            'opt': opt,
            'epoch': epoch
                    } ,filename)

    def load_checkpoint(self, filename):
        ckpt = torch.load(filename)
        self.encoder_src.load_state_dict(ckpt['encoder_src_dict'])
        self.encoder_tgt.load_state_dict(ckpt['encoder_tgt_dict'])
        self.linear_out.load_state_dict(ckpt['linear_out_dict'])
        epoch = ckpt['epoch']
        return epoch

class Discriminator(nn.Module):
    def __init__(self, base_model, adaptor):
        super(Discriminator, self).__init__()
        self.base_model = base_model
        self.adaptor = adaptor

    def forward(self, *args, **kwargs):
        outputs, _, _ = self.base_model.forward(*args, **kwargs)
        logits = self.adaptor(outputs)
        logits = logits.squeeze(-1)
        return logits

    def load_base_checkpoint(self, filename):
        self.base_model.load_checkpoint(filename)

