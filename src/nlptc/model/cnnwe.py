from mxnet.gluon import nn
from mxnet import nd

from nlptc.model.basic import SPConv1D
from ..activation import GLU
from ..utils import sequence_mask


class CNNWE(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_size, hidden_layers, channels, num_outputs, drop_prob, **kwargs):
        super().__init__(**kwargs)

        self.embedding = nn.Sequential()
        self.embedding.add(nn.Embedding(vocab_size, embed_size))
        self.embedding.add(nn.Dropout(drop_prob))

        self.conv_net = nn.Sequential()
        for i in range(hidden_layers):
            w = SPConv1D(channels=channels[i], kernel_size=kernel_size)
            v = SPConv1D(channels=channels[i], kernel_size=kernel_size)
            self.conv_net.add(GLU(w=w, v=v))
            self.conv_net.add(nn.Dropout(drop_prob))

        self.output_layer = nn.Dense(num_outputs, flatten=False)


    def forward(self, x):
        output = self.embedding(x)
        output = nd.swapaxes(output, 1, 2)
        output = self.conv_net(output)
        output = nd.swapaxes(output, 1, 2)
        output = self.output_layer(output)
        return output


