import torch
import argparse
import codecs
import nmt
import json
from torch import cuda
import time
from train import vocab_wrapper
from data import Data_Loader, Vocab


def get_sentence(idx, vocab):
    return ' '.join([vocab.itos[x] for x in idx])

def translate_file(translator, test_iter, tgt_fout, fields, use_cuda):
    print('start translating ...')
    process_bar = nmt.misc_utils.ShowProcess(len(test_iter))
    with codecs.open(tgt_fout, 'w', 'utf8') as tgt_file:
        for batch in test_iter:
            process_bar.show_process()
            src, src_lengths = batch.src
            ref_src, ref_src_lengths = batch.ref_src
            ref_tgt, ref_tgt_lengths = batch.ref_tgt
            ret = translator.translate_batch(src, ref_src, ref_tgt, src_lengths, ref_src_lengths, ref_tgt_lengths, batch)
            for raw, s in zip(ret['predictions'], ret['scores']):
                for idx, hyp_inds in enumerate(raw):
                    sentence_out = get_sentence(hyp_inds, fields['tgt'].vocab)
                    tgt_file.write(sentence_out+'\n')
                    #ms = s[idx]/ len(hyp_inds)
                    #tgt_file.write(sentence_out+'\t'+str(ms)+'\n')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_type", type=str)
    parser.add_argument("-test_file", type=str)
    parser.add_argument("-tgt_out", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int)
    parser.add_argument('-beam_size', default = 5, type=int)
    parser.add_argument('-decode_max_length', default = 20, type = int)
    parser.add_argument('-n_best', default= 1, type = int)
    parser.add_argument('-src_vocab', type=str)
    parser.add_argument('-tgt_vocab', type=str)
    parser.add_argument('-gan', type = bool, default = False)
    args = parser.parse_args()
    opt = torch.load(args.model)['opt']


    fields = dict()
    vocab_src =  Vocab(args.src_vocab, noST = True)
    vocab_tgt = Vocab(args.tgt_vocab)
    fields['src'] = vocab_wrapper(vocab_src)
    fields['tgt'] = vocab_wrapper(vocab_tgt)

    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True

    if args.model_type == "base":
        model = nmt.model_helper.create_base_model(opt, fields)
    if args.model_type == "bibase":
        model = nmt.model_helper.create_bibase_model(opt, fields)
    if args.model_type == "ref":
        model = nmt.model_helper.create_ref_model(opt, fields)
    if args.model_type == "ev":
        model = nmt.model_helper.create_ev_model(opt, fields)
    if args.model_type == "rg":
        model = nmt.model_helper.create_response_generator(opt, fields)
    if args.model_type == "CAS":
        model = nmt.model_helper.create_joint_model(opt, fields)
    if args.model_type == "JNT":
        model = nmt.model_helper.create_joint_template_response_model(opt, fields)

    print('Loading parameters ...')

    if args.gan:
        ckpt = torch.load(args.model)
        model.load_state_dict(ckpt['generator_dict'])
    else:
        model.load_checkpoint(args.model)

    if use_cuda:
        model = model.cuda()

    translator = nmt.Translator(model=model,
                                fields=fields,
                                beam_size = args.beam_size,
                                n_best= args.n_best,
                                max_length= args.decode_max_length,
                                global_scorer=None,
                                cuda=use_cuda)
    mask_end = (args.model_type == 'ev') or (args.model_type == 'CAS') or (args.model_type == 'JNT')
    test_iter = Data_Loader(args.test_file, 1, train = False, mask_end = mask_end)
    translate_file(translator, test_iter, args.tgt_out, fields, use_cuda)

if __name__ == '__main__':
    main()
