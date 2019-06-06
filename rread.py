import sys
from data import Vocab
if __name__ == "__main__":
    prefix = './data/'+sys.argv[1]+'/vocab_'
    vocabs = [Vocab(prefix+'src', noST = True),  Vocab(prefix+'tgt')]
    with open(sys.argv[2]) as f:
        for line in f.readlines():
            res = []
            sents = line.split('|')
            for idx, s in enumerate(sents):
                res.append(' '.join( [str(vocabs[idx%2].stoi.get(x, vocabs[idx%2].stoi[vocabs[idx%2].UNK])) for x in s.split()]  ))
            print ('|'.join(res))

