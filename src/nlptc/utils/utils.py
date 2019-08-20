from mxnet import nd

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


def log_sum_exp(vec):
    max_score = nd.max(vec, axis=-1).expand_dims(-1)
    return nd.log(nd.sum(nd.exp(vec - max_score)), axis=-1) + max_score.squeeze()

def to_int(x):
    return int(x.asscalar())