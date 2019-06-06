import sys
from data import Vocab
if __name__ == "__main__":
    prefix = '/home/jcykcai/JunNMT/data/'+sys.argv[1]+'/vocab_'
    vocabs = [Vocab(prefix+'src', noST = True),  Vocab(prefix+'tgt')]
    with open(sys.argv[2]) as f, open(sys.argv[3]) as f1:
        line1s = f1.readlines()
        cur = 0
        for line in f.readlines():
            ooo = line1s[cur].strip().split('|')
            cur += 1
            sent = line.strip().split('\t')
            #assert len(sent) == 2
            sent = sent[0]
            ooo[1] = ' '.join( [ str(vocabs[1].stoi.get(x, vocabs[1].stoi[vocabs[1].UNK])) for x in sent.split()[:-1]])
            res = ooo[:2]
            print ('|'.join(res))

