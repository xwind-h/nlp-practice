from mxnet import nd

from nlptc.utils.utils import log_sum_exp


class CRF:
    def __init__(self, tag_size, start_tag_idx=None, stop_tag_idx=None):
        self._tag_size = tag_size
        self._start_tag_idx = start_tag_idx if start_tag_idx is not None else 0
        self._stop_tag_idx = stop_tag_idx if stop_tag_idx is not None else (tag_size - 1)

        # transition score of tag, shape=(tag_size, tag_size)
        self._transitions = nd.random_normal(shape=(self._tag_size, self._tag_size))

    def _forward_alg(self, state_feats):
        # state_feats: state features of each tag, shape=(seq_len, tag_size)
        alpha = nd.full(self._tag_size, -10000.)
        alpha[self._start_tag_idx] = 0.
        for feat in state_feats:
            score = feat + self._transitions
            alpha = log_sum_exp(alpha + score.transpose())
        alpha = log_sum_exp(alpha + self._transitions[:, self._stop_tag_idx]).asscalar()
        return alpha




