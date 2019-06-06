from nmt.modules.Encoder import EncoderRNN
from nmt.modules.Decoder import AttnDecoderRNN, KVAttnDecoderRNN, AuxDecoderRNN, AuxMemDecoderRNN
from nmt.modules.Attention import GlobalAttention
from nmt.modules.Embedding import Embedding
from nmt.Model import vanillaNMTModel, refNMTModel, bivanillaNMTModel, editVectorGenerator, templateGenerator, evNMTModel, Discriminator, responseGenerator, tem_resNMTModel, Critic, jointTemplateResponseGenerator
import torch
import torch.nn as nn

def create_emb_for_encoder_and_decoder(src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       padding_idx):

    embedding_encoder = Embedding(src_vocab_size,src_embed_size,padding_idx)
    embedding_decoder = Embedding(tgt_vocab_size,tgt_embed_size,padding_idx)


    return embedding_encoder, embedding_decoder


def create_emb_for_encoders_and_decoder(src_vocab_size,
                                        ref_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       ref_embed_size,
                                       tgt_embed_size,
                                       padding_idx):

    embedding_encoder_src = Embedding(src_vocab_size, src_embed_size,padding_idx)
    embedding_encoder_ref = Embedding(ref_vocab_size, ref_embed_size,padding_idx)
    embedding_decoder = Embedding(tgt_vocab_size, tgt_embed_size,padding_idx)

    return embedding_encoder_src, embedding_encoder_ref, embedding_decoder

def create_encoder(opt):
    encoder = EncoderRNN(opt.rnn_type,
                        opt.embedding_size,
                        opt.hidden_size,
                        opt.num_layers,
                        opt.dropout,
                        opt.bidirectional)

    return encoder

def create_decoder(opt):
    if opt.decoder_type == 'AttnDecoderRNN':
        decoder = AttnDecoderRNN(opt.rnn_type,
                                opt.atten_model,
                                opt.embedding_size,
                                opt.hidden_size,
                                opt.num_layers,
                                opt.dropout)

    return decoder

def create_generator(input_size, output_size):
    generator = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LogSoftmax(dim=-1))
    return generator

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
    if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for layer in range(m.num_layers):
            nn.init.orthogonal(getattr(m,"weight_ih_l%d"%(layer)).data)
            nn.init.orthogonal(getattr(m,"weight_hh_l%d"%(layer)).data)

def create_ref_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = \
            create_emb_for_encoder_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                padding_idx)
    encoder_src = create_encoder(opt)
    encoder_ref = create_encoder(opt)
    decoder_ref = AttnDecoderRNN(opt.rnn_type,
                                 (opt.atten_model if opt.src_attention else "none" ),
                                opt.embedding_size,
                                opt.hidden_size,
                                opt.num_layers,
                                opt.dropout)
    decoder = KVAttnDecoderRNN(opt.rnn_type,
                                opt.atten_model,
                                opt.embedding_size,
                                opt.hidden_size,
                                opt.num_layers,
                                opt.dropout,
                                opt.src_attention,
                                opt.mem_gate,
                                opt.gate_vector)

    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    model = refNMTModel(enc_embedding,
                     dec_embedding,
                     encoder_src,
                     encoder_ref,
                     decoder_ref,
                     decoder,
                     generator,
                     fields)

    model.apply(weights_init)
    return model

def create_base_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = \
            create_emb_for_encoder_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                padding_idx)
    encoder = create_encoder(opt)
    decoder = create_decoder(opt)
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    model = vanillaNMTModel(enc_embedding,
                     dec_embedding,
                     encoder,
                     decoder,
                     generator)

    model.apply(weights_init)
    return model

def create_bibase_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, enc_embedding_ref, dec_embedding = \
            create_emb_for_encoders_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                tgt_vocab_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                padding_idx)
    encoder = create_encoder(opt)
    encoder_ref = create_encoder(opt)
    decoder = create_decoder(opt)
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    model = bivanillaNMTModel(enc_embedding,
                     enc_embedding_ref,
                     dec_embedding,
                     encoder,
                     encoder_ref,
                     decoder,
                     generator)

    model.apply(weights_init)
    return model

def create_template_generator(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = \
            create_emb_for_encoder_and_decoder(src_vocab_size,
                                                tgt_vocab_size,
                                                opt.embedding_size,
                                                opt.embedding_size,
                                                padding_idx)
    encoder_ref = EncoderRNN("GRU", opt.embedding_size, opt.hidden_size, 1, opt.dropout_ev, opt.bidirectional)
    attention_src = GlobalAttention(opt.embedding_size, attn_type="mlp")
    attention_ref = GlobalAttention(opt.embedding_size, attn_type="mlp")

    ev_generator = editVectorGenerator(enc_embedding, dec_embedding, encoder_ref, attention_src, attention_ref, nn.Dropout(opt.dropout_ev))
    masker = nn.Sequential(nn.Linear(3*opt.embedding_size, opt.embedding_size), nn.ReLU(), nn.Linear(opt.embedding_size, 1), nn.Sigmoid())
    model = templateGenerator(ev_generator, masker, nn.Dropout(opt.dropout_ev))
    model.apply(weights_init)
    return model

def create_ev_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = \
        create_emb_for_encoder_and_decoder(src_vocab_size,
                                           tgt_vocab_size,
                                           opt.embedding_size,
                                           opt.embedding_size,
                                           padding_idx)
    encoder_ref = EncoderRNN("GRU", opt.embedding_size, opt.hidden_size, 1, opt.dropout_ev, opt.bidirectional)
    attention_src = GlobalAttention(opt.embedding_size, attn_type="mlp")
    attention_ref = GlobalAttention(opt.embedding_size, attn_type="mlp")
    ev_generator = editVectorGenerator(enc_embedding, dec_embedding, encoder_ref, attention_src, attention_ref, nn.Dropout(opt.dropout_ev))
    bridge  = nn.Sequential(nn.Linear(2*opt.embedding_size, opt.aux_size), nn.ReLU())
    encoder_src = create_encoder(opt)
    decoder = AuxDecoderRNN(opt.rnn_type,
                                opt.atten_model,
                                opt.embedding_size + opt.aux_size,
                                opt.hidden_size,
                                opt.num_layers,
                                opt.dropout)

    generator = create_generator(opt.hidden_size, tgt_vocab_size)

    model = evNMTModel(enc_embedding,
                     dec_embedding,
                     encoder_src,
                     decoder,
                     generator,
                     ev_generator,
                     bridge,
                     fields)

    model.apply(weights_init)
    return model

def create_response_generator(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = create_emb_for_encoder_and_decoder(src_vocab_size, tgt_vocab_size,
                                            opt.embedding_size,
                                            opt.embedding_size,
                                            padding_idx)
    encoder_src = create_encoder(opt)
    if opt.use_ev:
        decoder = AuxMemDecoderRNN(opt.rnn_type,
                                   opt.atten_model,
                                   opt.embedding_size + opt.aux_size,
                                   opt.hidden_size,
                                   opt.num_layers,
                                   opt.dropout,
                                   opt.src_attention,
                                   opt.mem_gate,
                                   opt.gate_vector)

        bridge = nn.Sequential(nn.Linear(2*opt.embedding_size, opt.aux_size), nn.ReLU())
    else:
        decoder = KVAttnDecoderRNN(opt.rnn_type,
                                  opt.atten_model,
                                  opt.embedding_size,
                                  opt.hidden_size,
                                  opt.num_layers,
                                  opt.dropout,
                                  opt.src_attention,
                                  opt.mem_gate,
                                  opt.gate_vector)
        bridge = None
    encoder_ref = create_encoder(opt)
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    model = responseGenerator(enc_embedding, dec_embedding, encoder_src, decoder, generator, encoder_ref, bridge, fields)
    model.apply(weights_init)
    return model

def create_joint_model(opt, fields):
    template_generator = create_template_generator(opt, fields)
    response_generator = create_response_generator(opt, fields)
    model = tem_resNMTModel(template_generator, response_generator, opt.use_ev)
    return model

def create_joint_template_response_model(opt, fields):
    src_vocab_size = len(fields['src'].vocab)
    tgt_vocab_size = len(fields['tgt'].vocab)
    padding_idx = fields['src'].vocab.stoi[fields['src'].vocab.PAD]
    enc_embedding, dec_embedding = create_emb_for_encoder_and_decoder(src_vocab_size, tgt_vocab_size,
                                            opt.embedding_size,
                                            opt.embedding_size,
                                            padding_idx)
    encoder_src = create_encoder(opt)
    if opt.use_ev:
        decoder = AuxMemDecoderRNN(opt.rnn_type,
                                   opt.atten_model,
                                   opt.embedding_size + opt.aux_size,
                                   opt.hidden_size,
                                   opt.num_layers,
                                   opt.dropout,
                                   opt.src_attention,
                                   opt.mem_gate,
                                   opt.gate_vector)

        bridge = nn.Sequential(nn.Linear(2*opt.embedding_size, opt.aux_size), nn.ReLU())
    else:
        decoder = KVAttnDecoderRNN(opt.rnn_type,
                                  opt.atten_model,
                                  opt.embedding_size,
                                  opt.hidden_size,
                                  opt.num_layers,
                                  opt.dropout,
                                  opt.src_attention,
                                  opt.mem_gate,
                                  opt.gate_vector)
        bridge = None
    encoder_ref = EncoderRNN("GRU", opt.embedding_size, opt.hidden_size, 1, opt.dropout_ev, opt.bidirectional)
    attention_src = GlobalAttention(opt.embedding_size, attn_type="mlp")
    attention_ref = GlobalAttention(opt.embedding_size, attn_type="mlp")
    ev_generator = editVectorGenerator(enc_embedding, dec_embedding, encoder_ref, attention_src, attention_ref, nn.Dropout(opt.dropout_ev))
    generator = create_generator(opt.hidden_size, tgt_vocab_size)
    masker = nn.Sequential(nn.Linear(3*opt.embedding_size, opt.embedding_size), nn.ReLU(), nn.Linear(opt.embedding_size, 1), nn.Sigmoid())
    model = jointTemplateResponseGenerator(ev_generator, masker, nn.Dropout(opt.dropout_ev), enc_embedding, dec_embedding, encoder_src, decoder, generator, bridge, fields)
    model.apply(weights_init)
    return model

def create_critic_model(opt, fields):
    encoder_src = EncoderRNN("GRU", opt.embedding_size, opt.hidden_size, opt.num_layers, opt.dropout, opt.bidirectional)
    encoder_tgt = EncoderRNN("GRU", opt.embedding_size, opt.hidden_size, opt.num_layers, opt.dropout, opt.bidirectional)
    model = Critic(encoder_src, encoder_tgt, opt.dropout)
    model.apply(weights_init)
    return model

def create_GAN_model(opt, fields):
    # For now, we only consider the ev model for GAN training
    generator = create_ref_model(opt, fields)
    disc = create_ref_model(opt, fields)
    #critic = create_ref_model(opt, fields)

    discriminator = Discriminator(disc, nn.Linear(opt.hidden_size, 1))
    critic  = Discriminator(disc, nn.Linear(opt.hidden_size, 1))
    discriminator.adaptor.apply(weights_init)
    critic.adaptor.weight.data.zero_()
    critic.adaptor.bias.data.zero_()
    return generator, discriminator, critic
