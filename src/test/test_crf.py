from nlptc import model
from mxnet import nd, autograd, gluon


tag_size = 5
crf = model.CRF(tag_size=tag_size, start_tag_idx=0, stop_tag_idx=4)
crf.initialize()

x = nd.random_normal(shape=(5, 4, 5))
tag_seq = nd.random_randint(low=1, high=4, shape=(5, 4)).astype('float32')

optimizer = gluon.Trainer(crf.collect_params(), 'sgd', {'learning_rate': 0.01})

for i in range(5):
    with autograd.record():
        loss = crf.neg_log_likehood(x, tag_seq).sum()
    loss.backward()
    optimizer.step(5)
    tr = crf.params.get('transition').data()
    print("epoch %i " % i)
    print(tr)

y = crf(x)
print(y)

