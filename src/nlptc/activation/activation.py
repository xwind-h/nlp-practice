from mxnet.gluon import nn


class GLU(nn.HybridBlock):
    def __init__(self, w, v, **kwargs):
        super().__init__(**kwargs)
        self.w = w
        self.v = v

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.w(x) * F.sigmoid(self.v(x))
