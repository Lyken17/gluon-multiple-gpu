from time import time

import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.data import DataLoader

from mxnet.io import NDArrayIter, DataIter
from models import net

from mxnet.test_utils import get_mnist, get_cifar10

loss = gluon.loss.SoftmaxCrossEntropyLoss()


def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()


def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch[0], ctx)
    label = gluon.utils.split_and_load(batch[1], ctx)
    # compute gradient

    # forward_backward(net, data, label)
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]

    total_loss = 0.0
    for l in losses:
        l.backward()
        total_loss += l.sum().as_in_context(mx.cpu()).asscalar()

    # update parameters
    trainer.step(batch[0].shape[0])
    return total_loss


def valid_batch(batch, ctx, net):
    current_ctx = ctx[0]
    data = batch[0].as_in_context(current_ctx)
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred == batch[1].as_in_context(current_ctx)).asscalar()


def main(num_gpus, batch_size, lr):
    # the list of GPUs will be used
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))

    # data iterator
    # mnist = get_mnist()
    # train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    # valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
    def preprocess(data, label):
        # data = mx.image.imresize(data, 224, 224)
        data = mx.nd.transpose(data, (2, 0, 1))
        data = data.astype("float32")
        # data = data.astype(np.float32)
        label = label.astype("float32")
        return data, label

    batch_size = 64
    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True, transform=preprocess),
        batch_size=batch_size, shuffle=True, last_batch='discard')

    valid_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False, transform=preprocess),
        batch_size=batch_size, shuffle=False, last_batch='discard')

    print('Batch size is {}'.format(batch_size))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
    for epoch in range(5):
        # train
        start = time()
        for iter, batch in enumerate(train_data):
            loss = train_batch(batch, ctx, net, trainer)
            if iter % 100 == 0:
                nd.waitall()
                print("%d\t%.4f" % (iter, loss))
        nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec' % (epoch, time() - start))

        # validating
        correct, num = 0.0, 0.0
        for batch in valid_data:
            correct += valid_batch(batch, ctx, net)
            num += batch[0].shape[0]
        print('         validation accuracy = %.4f' % (correct / num))


# global environments
GPU_COUNT = 4  # increase if you have more

main(1, 64, .1)
# main(GPU_COUNT, 64 * GPU_COUNT, .1)
