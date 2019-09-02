from nlptc import model
from mxnet import nd

vocab_size = 10
tag_size = 4
embed_size = 20
kernel_size = 3
hidden_layers = 3
channels = [20] * 5
drop_prob = 0.2

model = model.CNNWE(vocab_size=vocab_size, embed_size=embed_size, kernel_size=kernel_size,
                    hidden_layers=hidden_layers, channels=channels, num_outputs=tag_size, drop_prob=drop_prob)

x = nd.random_randint(low=0, high=10, shape=(5, 4))
y = model(x)
print(y)

nd.Convolution()
