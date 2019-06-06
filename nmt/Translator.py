import torch
import torch.nn as nn
from torch.autograd import Variable
import nmt
from nmt.utils.data_utils import sequence_mask
import time

class Translator(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        # Set model in eval mode.
        self.model.eval()
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.beam_size = max(beam_size, n_best)
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": []}

    def translate_batch(self, src, ref_src, ref_tgt, src_lengths, ref_src_lengths, ref_tgt_lengths, batch = None):
        """
        Translate a batch of sentences.
        Mostly a wrapper around :obj:`Beam`.
        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        #last_time = time.time()
        beam_size = self.beam_size
        batch_size = len(src_lengths)
        vocab = self.fields["tgt"].vocab
        beam = [nmt.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[vocab.PAD],
                                    eos=vocab.stoi[vocab.EOT],
                                    bos=vocab.stoi[vocab.EOS],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        model_type = self.model.__class__.__name__
        if model_type == "refNMTModel":
            context, enc_states, context_keys, context_mask, src_context, src_mask = self.model.encode(src, ref_src, ref_tgt, src_lengths, ref_src_lengths, ref_tgt_lengths)
            dec_states = self.model.init_decoder_state(enc_states, context)
            context_mask = context_mask.repeat(beam_size, 1)
            context = rvar(context.data)
            context_keys = rvar(context_keys.data)
            src_context = rvar(src_context.data)
            src_mask = src_mask.repeat(beam_size, 1)
        if model_type == "vanillaNMTModel":
            context, enc_states, context_mask = self.model.encode(src, src_lengths)
            dec_states = self.model.init_decoder_state(enc_states, context)
            context = rvar(context.data)
            context_mask = context_mask.repeat(beam_size, 1)
        if model_type == "bivanillaNMTModel":
            context, enc_states = self.model.encode(src, ref_tgt, src_lengths, ref_tgt_lengths)
            dec_states = self.model.init_decoder_state(enc_states, context)
            context = rvar(context.data)
        if  model_type == "evNMTModel":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            context, enc_hidden, context_mask, dist = self.model.encode(src, src_lengths, I_word, I_word_length, D_word, D_word_length, ref_tgt, ref_tgt_lengths)
            dec_states = self.model.init_decoder_state(enc_hidden, context)
            dist = dist.repeat(beam_size, 1)
            context_mask = context_mask.repeat(beam_size, 1)
            context = rvar(context.data)
        if  model_type == "responseGenerator":
            context, enc_states, context_mask, dist, src_context, src_mask = self.model.encode(src, ref_tgt, src_lengths, ref_tgt_lengths)
            dec_states = self.model.init_decoder_state(enc_states, context)
            context_mask = context_mask.repeat(beam_size, 1)
            context = rvar(context.data)
            src_context = rvar(src_context.data)
            src_mask = src_mask.repeat(beam_size, 1)
        if model_type == "jointTemplateResponseGenerator":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            context, enc_states, context_mask, dist, src_context, src_mask, preds = self.model.encode(I_word, I_word_length, D_word, D_word_length, ref_tgt, ref_tgt_lengths, src, src_lengths)
            dec_states = self.model.init_decoder_state(enc_states, context)
            if dist is not None:
                dist = dist.repeat(beam_size, 1)
            context_mask = context_mask.repeat(beam_size, 1)
            context = rvar(context.data)
            src_context = rvar(src_context.data)
            src_mask = src_mask.repeat(beam_size, 1)
        if model_type == "tem_resNMTModel":
            I_word, I_word_length = batch.I
            D_word, D_word_length = batch.D
            preds, ev = self.model.template_generator(I_word, I_word_length, D_word, D_word_length, ref_tgt, ref_tgt_lengths, return_ev = True)
            preds = preds.squeeze(2)
            template, template_lengths = self.model.template_generator.do_mask_and_clean(preds, ref_tgt, ref_tgt_lengths)
            context, enc_states, context_mask, dist, src_context, src_mask = self.model.response_generator.encode(src, template, src_lengths, template_lengths, ev)
            dec_states = self.model.response_generator.init_decoder_state(enc_states, context)
            if dist is not None:
                dist = dist.repeat(beam_size, 1)
            context_mask = context_mask.repeat(beam_size, 1)
            context = rvar(context.data)
            src_context = rvar(src_context.data)
            src_mask = src_mask.repeat(beam_size, 1)
        # (2) Repeat src objects `beam_size` times. 
        if not isinstance(dec_states, tuple): # GRU
            dec_states = Variable(dec_states.data.repeat(1, beam_size, 1))
        else: # LSTM
            dec_states = (
                Variable(dec_states[0].data.repeat(1, beam_size, 1)),
                Variable(dec_states[1].data.repeat(1, beam_size, 1)),
                )

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))


            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            # inp = inp.unsqueeze(2)

            # Run one step.
            if model_type == "refNMTModel":
                dec_out, dec_states, attn = self.model.decode(inp, context_keys, context, dec_states, context_mask, src_context, src_mask)
            if model_type == "vanillaNMTModel":
                dec_out, dec_states, attn = self.model.decode(inp, context, dec_states, context_mask)
            if model_type == "bivanillaNMTModel":
                dec_out, dec_states, attn = self.model.decode(inp, context, dec_states)
            if model_type == "evNMTModel":
                dec_out, dec_states, attn = self.model.decode(inp, context, dec_states, dist, context_mask)
            if model_type == "responseGenerator":
                dec_out, dec_states, attn = self.model.decode(inp, context, dec_states, None, context_mask, src_context, src_mask)
            if model_type == "tem_resNMTModel":
                dec_out, dec_states, attn = self.model.response_generator.decode(inp, context, dec_states, dist, context_mask, src_context, src_mask)
            if model_type == "jointTemplateResponseGenerator":
                dec_out, dec_states, attn = self.model.decode(inp, context, dec_states, dist, context_mask, src_context, src_mask)
            dec_out = dec_out.squeeze(0)

            # (b) Compute a vector of batch*beam word scores.
            out = self.model.generator(dec_out).data
            out = unbottle(out)
            # beam x batch_size x tgt_vocab
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j])
                self.beam_update(j, b.get_current_origin(), beam_size, dec_states)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)


        return ret

    def beam_update(self, idx, positions, beam_size, states):
        if not isinstance(states, tuple):
            states = (states, )

        for e in states:
            sizes = e.size()
            br = sizes[1]
            sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2])[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": []}
        for b in beam:
            if self.beam_accum:
                self.beam_accum['predicted_ids'].append(torch.stack(b.next_ys[1:]).tolist())
                self.beam_accum['beam_parent_ids'].append(torch.stack(b.prev_ks).tolist())
                self.beam_accum['scores'].append(torch.stack(b.all_scores).tolist())

            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
        return ret
