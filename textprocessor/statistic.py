import pandas as pd

def token_statistic(file, out_file=None, per_sent=False):
    freq = {}
    total = 0
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            toks = line.split(' ')
            if per_sent: 
                toks = set(toks)
                total += 1
            for tok in toks:
                if tok in freq:
                    freq[tok] += 1
                else:
                    freq[tok] = 1
                if not per_sent: total += 1
            line = f.readline()
    for k, v in freq.items():
        freq[k] = v / total
    l = [(k,v) for k,v in freq.items()]
    l = sorted(l, key=lambda x: x[1], reverse=True)
    if out_file:
        with open(out_file, 'w') as o:
            for k, v in l:
                o.write(f'{k} {v}\n')
    return freq, total, l