with open('../data/douban/vocab_src') as f:
    words = [line.strip() for line in f.readlines()]
    vocab_src = dict(zip(words, range(0, len(words))))

with open('../data/douban/vocab_tgt') as f:
    words = [line.strip() for line in f.readlines()]
    vocab_tgt = dict(zip(words, range(0, len(words))))

def write(res, fo, SrcnoST = False):
    vocabs = [vocab_src, vocab_tgt]
    line = []
    if SrcnoST:
        tmp = [1, 3]
    else:
        tmp = [3, 3]
    for vid, s in enumerate(res):
        words = s.strip().split()
        line.append(' '.join([str(tmp[vid%2] + vocabs[vid%2].get(w, 0)) for w in words]))
    line = '|'.join(line)
    fo.write(line + '\n')

with open('input_case') as fsrc, \
    open('in', 'w') as fox:
    last_src, last_tgt = None, None
    result = []
    for line in fsrc.readlines():
        result = [ t.strip() for t in line.strip().split('|')]
        assert len(result) ==4
        write(result, fox, True)
