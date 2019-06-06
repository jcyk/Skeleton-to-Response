import sys


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
        x[-1] = ' '.join([str(t) for t in new_z])
        print '|'.join(x)

