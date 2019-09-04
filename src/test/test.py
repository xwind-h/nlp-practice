from nlptc import model
from mxnet import nd

vocab_size = 10
tag_size = 7
embed_size = 20
kernel_size = 3
hidden_layers = 3
channels = [20] * 5
drop_prob = 0.2

model = model.SPConv1D(channels=20, kernel_size=kernel_size)
model.initialize()

x = nd.random_uniform(shape=(1, 4, 20))
y = model(x)

print(y)