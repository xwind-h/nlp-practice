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
        """
        :param state_feats: shape=(batch_size, seq_len, tag_size)
        :return: shape=(batch_size,)
        """
        state_feats_tmp = state_feats.transpose((1, 0, 2))
        alpha = nd.full((state_feats.shape[0], self._tag_size), -10000.)
        alpha[:, self._start_tag_idx] = 0.

        for feat in state_feats_tmp:
            score = feat.expand_dims(1) + self._transitions.data()
            alpha = log_sum_exp(alpha.expand_dims(1) + score.transpose((0, 2, 1)))
        alpha = log_sum_exp(alpha + self._transitions.data()[:, self._stop_tag_idx])
        return alpha

    def _seq_score(self, state_feats, tag_seq):
        """
        :param state_feats: shape=(batch_size, seq_len, tag_size)
        :param tag_seq: shape=(batch_size, seq_len)
        :return: shape=(batch_size,)
        """
        score = nd.zeros(shape=state_feats.shape[0])
        tag_seq = nd.concat(nd.array([self._start_tag_idx]), *tag_seq, dim=0)

        # state_feats_tmp: shape=(seq_len, batch_size, tag_size)
        state_feats_tmp = state_feats.transpose(1, 0, 2)
        # tag_seq_tmp: shape=(seq_len, batch_size)
        tag_seq_tmp = tag_seq.transpose()
        pad = nd.full((1, tag_seq_tmp.shape[-1]), self._start_tag_idx)
        tag_seq_tmp = nd.concat(pad, tag_seq_tmp, dim=0)

        for idx, feat in enumerate(state_feats_tmp):
            score += nd.pick(feat, tag_seq_tmp[idx + 1], axis=1) + \
                     nd.pick(self._transitions.data()[tag_seq_tmp[idx]], tag_seq_tmp[idx + 1], axis=1)

        score += self._transitions.data()[tag_seq_tmp[-1], self._stop_tag_idx]
        return score

    def neg_log_likehood(self, x, tag_seq):
        """
        :param x: shape=(batch_size, seq_len, tag_size)
        :param tag_seq: shape=(batch_size, seq_len)
        :return: (batch_size,)
        """
        seq_score = self._seq_score(x, tag_seq)
        log_norm = self._log_norm(x)
        return log_norm - seq_score

    def viterbi_decode(self, state_feats):
        """
        :param state_feats: shape=(batch_size, seq_len, tag_size)
        :return:
            path_score: shape=(batch_size,)
            best_path: shape=(batch_size, seq_len)
        """
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

    def forward(self, x):
        """
        :param x: shape=(batch_size, seq_len, tag_size)
        :return:
            score: shape=(batch_size,)
            path: shape=(batch_size, seq_len)
        """
        score, path = self.viterbi_decode(x)
        return score, path
