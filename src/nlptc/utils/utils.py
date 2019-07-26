

def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = max(lengths)
    mask = []
    for l in lengths:
        if l < maxlen:
            m = [1] * l
        else:
            m = [1] * maxlen
        m.extend([0] * (maxlen - l))
        mask.append(m)
    return mask