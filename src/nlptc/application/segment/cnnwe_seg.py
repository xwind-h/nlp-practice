import random

import mxnet as mx
import time

import os
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn

from nlptc.dataset import Embedding, SegmentData, Vocab
from nlptc.model import CRF
from nlptc.model.cnnwe import CNNWE
from nlptc.utils import try_gpu


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

    ctx = try_gpu()
    model.initialize(mx.init.Xavier(), ctx=ctx)
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

    embedding = Embedding('news_tensite.200d', vocab).idx_to_vec.copyto(ctx)

    cnn_model.embedding[0].weight.set_data(embedding)
    cnn_model.embedding[0].collect_params().setattr('grad_req', 'null')

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    model_out = os.path.join(project_path, 'data', 'model', 'cnnwe_seg_model.params')

    if os.path.exists(model_out):
        model.load_parameters(model_out, ctx=ctx)

    epoch = 10
    random_idx = list(range(len(train_data)))
    random.shuffle(random_idx)
    val_size = len(data.val_data[0])
    for e in range(epoch):
        likehood = 0
        iter = 0
        start = time.time()
        for i in random_idx:
            st = train_data[i]
            tags = train_tag[i]
            if len(st) < 2:
                continue
            with autograd.record():
                x = nd.array(st, ctx=ctx).reshape(1, -1)
                y = nd.array(tags, ctx=ctx).reshape(1, -1)
                output = cnn_model(x)
                loss = crf_model.neg_log_likehood(output, y).sum()
                loss.backward()
            optimizer.step(1)

            likehood += loss.asscalar()
            iter += 1
            if iter % 10000 == 0:
                t = time.time() - start
                print("Epoch %i, iter %i, neg likehood %.4f, process time: %.4f sec" % (e, iter, likehood / 100, t))

                j = random.randint(0, val_size)
                val_st, val_tag = data.val_data[0][j], data.val_data[1][j]
                x = nd.array(vocab.to_indices(val_st), ctx=ctx).reshape(1, -1)
                pred_tag = crf_model.viterbi_decode(cnn_model(x))[1].reshape(-1,).asnumpy().astype('int').tolist()
                pred_tag = tag_vocab.to_words(pred_tag)
                assert len(pred_tag) == len(val_st)
                st = data.tagger.rebuild(val_st, val_tag)
                rst = data.tagger.rebuild(val_st, pred_tag)
                print('gold: %s' % ' '.join(st))
                print('predict: %s' % ' '.join(rst))
                model.save_parameters(model_out)
                start = time.time()
                likehood = 0


if __name__ == '__main__':
    train()
