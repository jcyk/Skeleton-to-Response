from data import Vocab
import sys

vocab = sys.argv[1]
stopwords = sys.argv[2]


v = Vocab(vocab)#, noST = True)

for line in open(stopwords, encoding = "gbk").readlines():
    w = line.strip()
    if w in v.stoi:
        print (v.stoi[line.strip()])
