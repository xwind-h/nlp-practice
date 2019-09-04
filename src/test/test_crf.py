from nlptc import model
from mxnet import nd, autograd, gluon

tag_size = 5
crf = model.CRF(tag_size=tag_size)
crf.initialize()

x = nd.random_normal(shape=(2, 1, 5))
tag_seq = nd.random_randint(low=1, high=4, shape=(2, 1)).astype('float32')
print('x',x)
loss = crf.neg_log_likehood(x, tag_seq)
print('loss', loss)
y = crf(x)
print(y)
