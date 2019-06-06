import sys
from data import Vocab

vocab_tgt = Vocab('../data/golden/vocab_tgt')

with open(sys.argv[1]) as f:
    for line in f.readlines():
        x = line.strip().split('|')
        y = x[-1]
        z = [ int(t) for t in y.split()]
        iszero = False
        new_z = []
        for w in z:
            if iszero and w == 0:
                continue
            else:
                new_z.append(w)
            iszero = (w==0)
        print (' '.join([vocab_tgt.i2s(w) for w in new_z]))

