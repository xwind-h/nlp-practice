import mxnet as mx
import time
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn

from nlptc.dataset import Embedding, SegmentData, Vocab
from nlptc.model import CRF
from nlptc.model.cnnwe import CNNWE


def train():
    data = SegmentData('sighan2005-msr', tagger='bmes')
    vocab = Vocab(data.train_data[0], min_freq=5, unk='<unk>')
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
    crf_model = CRF(tag_size=tag_size)

    model = nn.Sequential()
    model.add(cnn_model)
    model.add(crf_model)
    model.initialize(mx.init.Xavier())
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

    embedding = Embedding('news_tensite.200d', vocab)
    cnn_model.embedding[0].weight.set_data(embedding.idx_to_vec)
    cnn_model.embedding[0].collect_params().setattr('grad_req', 'null')

    epoch = 1
    for e in range(epoch):
        likehood = 0
        iter = 0
        start = time.time()
        for st, tags in zip(train_data, train_tag):
            if len(st) < 2:
                continue
            with autograd.record():
                x = nd.array(st).reshape(1, -1)
                y = nd.array(tags).reshape(1, -1)
                output = cnn_model(x)
                loss = crf_model.neg_log_likehood(output, y).sum()
                loss.backward()
            optimizer.step(1)

            likehood += loss.asscalar()
            iter += 1
            if iter % 100 == 0:
                t = time.time() - start
                print("Epoch %i, iter %i, neg likehood %.4f, process time: %.4f" % (e, iter, likehood / 100, t))
                start = time.time()
                likehood = 0


if __name__ == '__main__':
    train()
