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
    ctx = try_gpu()

    data = SegmentData('sighan2005-msr', tagger='bmes')
    vocab = Vocab(data.train_data[0], min_freq=5, unk='<unk>')
    tag_vocab = Vocab(data.train_data[1])

    train_data = []
    train_tags = []
    train_size = min(len(data.train_data[0]), len(data.train_data[1]))
    for i in range(train_size):
        st = data.train_data[0][i]
        tags = data.train_data[1][i]
        if len(st) < 2 or len(tags) < 2:
            continue
        train_data.append(nd.array(vocab.to_indices(st), ctx=ctx).reshape(1, -1))
        train_tags.append(nd.array(tag_vocab.to_indices(tags), ctx=ctx).reshape(1, -1))

    val_seqs = []
    val_tags = []
    val_array = []
    val_size = min(len(data.val_data[0]), len(data.val_data[1]))
    for i in range(val_size):
        st = data.val_data[0][i]
        tags = data.val_data[1][i]
        if len(st) < 2 or len(tags) < 2:
            continue
        val_seqs.append(st)
        val_tags.append(tags)
        val_array.append(nd.array(vocab.to_indices(st), ctx=ctx).reshape(1, -1))

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
    for e in range(epoch):
        likehood = 0
        iter = 0
        start = time.time()
        for i in random_idx:
            st = train_data[i]
            tags = train_tags[i]
            if st.shape[1] < 2:
                continue
            with autograd.record():
                output = cnn_model(st)
                loss = crf_model.neg_log_likehood(output, tags).sum()
                loss.backward()
            optimizer.step(1)

            likehood += loss.asscalar()
            iter += 1
            if iter % 100 == 0:
                t = time.time() - start
                print("Epoch %i, iter %i, neg likehood %.4f, process time: %.4f sec" % (e, iter, likehood / 100, t))

                j = random.randint(0, len(val_array))
                val_st = val_array[j]
                pred_tag = crf_model.viterbi_decode(cnn_model(val_st))[1].reshape(-1,).asnumpy().astype('int').tolist()
                pred_tag = tag_vocab.to_words(pred_tag)
                assert len(pred_tag) == val_st.shape[1]
                st = data.tagger.rebuild(val_seqs[j],val_tags[j])
                rst = data.tagger.rebuild(val_seqs[j], pred_tag)
                print('gold: %s' % ' '.join(st))
                print('predict: %s' % ' '.join(rst))
                model.save_parameters(model_out)
                start = time.time()
                likehood = 0


if __name__ == '__main__':
    train()
