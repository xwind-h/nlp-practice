from mxnet import nd
from mxnet.gluon import nn


class SPConv1D(nn.Block):
    def __init__(self, kernel_size, strides=1, padding=None, **kwargs):
        super().__init__()
        self.right_padding = (kernel_size - 1) // 2
        self.left_padding = (kernel_size - 1) - self.right_padding
        strides = 1
        self.cnn = nn.Conv1D(kernel_size=kernel_size, strides=strides, **kwargs)

    def forward(self, x):
        x = nd.pad(x.expand_dims(0), pad_width=(0, 0, 0, 0, 0, 0, self.left_padding, self.right_padding),
                   mode='constant').squeeze()
        return self.cnn(x)