from mxnet import nd
from mxnet.gluon import nn

from nlptc.utils.utils import log_sum_exp, to_int


class CRF(nn.Block):
    def __init__(self, tag_size, start_tag_idx=None, stop_tag_idx=None):
        self._tag_size = tag_size
        self._start_tag_idx = start_tag_idx if start_tag_idx is not None else 0
        self._stop_tag_idx = stop_tag_idx if stop_tag_idx is not None else (tag_size - 1)

        # transition score of tag, shape=(tag_size, tag_size)
        self._transitions = self.params.get('transition', shape=(self._tag_size, self._tag_size))

    def _log_norm(self, state_feats):
        # state_feats: state features of each tag, shape=(seq_len, tag_size)
        alpha = nd.full(self._tag_size, -10000.)
        alpha[self._start_tag_idx] = 0.
        for feat in state_feats:
            score = feat + self._transitions.data()
            alpha = log_sum_exp(alpha + score.transpose())
        alpha = log_sum_exp(alpha + self._transitions.data()[:, self._stop_tag_idx])
        return alpha

    def _seq_score(self, state_feats, tag_seq):
        score = nd.array([0])
        tag_seq = nd.concat(nd.array([self._start_tag_idx]), *tag_seq, dim=0)
        for idx, feat in enumerate(state_feats):
            score += feat[to_int(tag_seq[idx + 1])] + self._transitions.data()[
                to_int(tag_seq[idx]), to_int(tag_seq[idx + 1])]

        score += self._transitions.data()[to_int(tag_seq[-1]), self._stop_tag_idx]
        return score

    def neg_log_likehood(self, state_feats, tag_seq):
        seq_score = self._seq_score(state_feats, tag_seq)
        log_norm = self._log_norm(state_feats)
        return log_norm - seq_score

    def viterbi_decode(self, state_feats):
        backpointers = []

        max_score = nd.full(self._tag_size, -10000.)
        max_score[self._start_tag_idx] = 0

        for feat in state_feats:
            next_tag_score = max_score + (feat + self._transitions.data()).transpose()
            backpointers.append(nd.argmax(next_tag_score, axis=-1))
            max_score = nd.max(next_tag_score, axis=-1)

        max_score += self._transitions.data()[:, self._stop_tag_idx]
        best_tag = to_int(nd.argmax(max_score, axis=-1))
        path_score = max_score[best_tag]

        best_path = [best_tag]
        for bp in reversed(backpointers):
            best_path.append(to_int(bp[best_path[-1]]))
        start = best_path.pop()
        assert start == self._start_tag_idx

        best_path.reverse()
        return path_score, best_path



