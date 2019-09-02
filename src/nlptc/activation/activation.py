from mxnet.gluon import nn
from mxnet import nd


class GLU(nn.Block):
    def __init__(self, w, v):
        self.w = w
        self.v = v

    def forward(self, x):
        return self.w(x) * nd.sigmoid(self.v(x))
