from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

mx.random.seed(1)

gpus = 4
ctx = [mx.gpu(_) for _ in range(gpus)]


def transformer(data, label):
    # data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label


print("prepare data")
batch_size = 256 * gpus
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False, last_batch='discard')

print("prepare model")
# alex_net = gluon.nn.Sequential()
# with alex_net.name_scope():
#     #  First convolutional layer
#     alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4, 4), activation='relu'))
#     alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
#     #  Second convolutional layer
#     alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
#     alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2, 2)))
#     # Third convolutional layer
#     alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
#     # Fourth convolutional layer
#     alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
#     # Fifth convolutional layer
#     alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
#     alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
#     # Flatten and apply fullly connected layers
#     alex_net.add(gluon.nn.Flatten())
#     alex_net.add(gluon.nn.Dense(4096, activation="relu"))
#     alex_net.add(gluon.nn.Dense(4096, activation="relu"))
#     alex_net.add(gluon.nn.Dense(10))

from models import net as alex_net

alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': .001 * gpus})
criterion = gluon.loss.SoftmaxCrossEntropyLoss()


def valid_batch(batch, net):
    current_ctx = ctx[0]
    data = batch[0].as_in_context(current_ctx)
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch[1].as_in_context(current_ctx)).asscalar()


def evaluate_accuracy(data_iterator, net):
    current_ctx = ctx[0]
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(current_ctx)
        label = l.as_in_context(current_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 20
smoothing_constant = .01

print("prepare training")
for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = gluon.utils.split_and_load(d, ctx)
        label = gluon.utils.split_and_load(l, ctx)
        # data = d.as_in_context(ctx)
        # label = l.as_in_context(ctx)
        batch_size = d.shape[0]
        with autograd.record():
            losses = [criterion(alex_net(X), Y) for X, Y in zip(data, label)]
            # output = alex_net(data)
            # loss = criterion(output, label)
        # loss.backward()
        total_loss = 0.0
        for l in losses:
            l.backward()
            total_loss += l.sum().as_in_context(mx.cpu()).asscalar()
        trainer.step(batch_size)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        # curr_loss = nd.mean(loss).asscalar()
        curr_loss = total_loss
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        if i % 20 == 0:
            print("Train:%s Iter:%d\tLoss: %.6f, " % (e, i, moving_loss))

    test_accuracy = evaluate_accuracy(test_data, alex_net)
    train_accuracy = evaluate_accuracy(train_data, alex_net)
    print("Epoch %s. Loss: %.4f, Train_acc %.4f, Test_acc %.4f" % (e, moving_loss, train_accuracy, test_accuracy))
