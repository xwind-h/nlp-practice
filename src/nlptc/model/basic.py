from mxnet.gluon import nn


class SPConv1D(nn.HybridBlock):
    def __init__(self, kernel_size, strides=1, padding=None, **kwargs):
        super().__init__()
        self.right_padding = (kernel_size - 1) // 2
        self.left_padding = (kernel_size - 1) - self.right_padding
        strides = 1
        self.cnn = nn.Conv1D(kernel_size=kernel_size, strides=strides, **kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.pad(x.expand_dims(0), pad_width=(0, 0, 0, 0, 0, 0, self.left_padding, self.right_padding),
                   mode='constant').squeeze(axis=0)
        return self.cnn(x)