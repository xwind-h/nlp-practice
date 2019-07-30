from mxnet.gluon import nn
from mxnet import nd

from ..activation import GLU
from ..utils import sequence_mask


class CNNWE(nn.Block):
    def __init__(self, seq_lens, vocab_size, embed_size, kernel_size, hidden_layers, channels, num_outputs, drop_prob, **kwargs):
        super.__init__(**kwargs)

        self.mask = nd.swapaxes(nd.expand_dims(nd.array(sequence_mask(seq_lens)), -1), 1, 2)
        self.embedding = nn.Sequential()
        self.embedding.add(nn.Embedding(vocab_size, embed_size))
        self.embedding.add(nn.Dropout(drop_prob))

        right_padding = (kernel_size - 1) // 2
        left_padding = (kernel_size - 1) - right_padding
        padding = (left_padding, right_padding)
        self.conv_nets = []
        for i in range(hidden_layers):
            conv_net = nn.Sequential()
            w = nn.Conv1D(channels=channels[i], kernel_size=kernel_size, padding=padding)
            v = nn.Conv1D(channels=channels[i], kernel_size=kernel_size, padding=padding)
            conv_net.add(GLU(w=w, v=v))
            conv_net.add(nn.Dropout(drop_prob))

            self.conv_nets.append(conv_net)

        self.output_layer = nn.Dense(num_outputs, flatten=False)


    def forward(self, x):
        output = self.embedding(x)
        output = nd.swapaxes(output, 1, 2)
        for conv in self.conv_nets:
            output = conv(output) * self.mask
        output = nd.swapaxes(output, 1, 2)
        output = self.output_layer(output)
        return output


