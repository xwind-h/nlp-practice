from nlptc.dataset import SegmentData
from nlptc.dataset.basic import Vocab
from nlptc.model import CRF
from nlptc.model.cnnwe import CNNWE

from mxnet import nd


def train():
    data = SegmentData('sighan2005-msr', tagger='bmes')
    vocab = Vocab(data.train_data[0], min_freq=5)
    tag_vocab = Vocab(data.train_data[1])

    train_data = [vocab.to_indices(seq) for seq in data.train_data[0]]
    train_tag = [tag_vocab.to_indices(tags) for tags in data.train_data[1]]

    vocab_size = len(vocab.idx_to_word)
    tag_size = len(tag_vocab.idx_to_word)
    embed_size = 200
    kernel_size = 3
    hidden_layers = 5
    channels = [200] * 5
    drop_prob = 0.2

    cnn_model = CNNWE(vocab_size=vocab_size, embed_size=embed_size, kernel_size=kernel_size,
                      hidden_layers=hidden_layers, channels=channels, num_outputs=tag_size, drop_prob=drop_prob)

    crf_model = CRF()
