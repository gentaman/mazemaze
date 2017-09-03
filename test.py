from tensorflow.examples.tutorials.mnist import input_data
import chainer
from chainer.training import extensions
import numpy as np
import argparse


class MyFeatureExtrator(chainer.Chain):
    def __init__(self):
        super(MyFeatureExtrator, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(in_channels=1, out_channels=64, ksize=5, stride=1)
            self.conv2 = chainer.links.Convolution2D(in_channels=None, out_channels=128, ksize=3, stride=1)

    def __call__(self, x):
        h = chainer.functions.max_pooling_2d(chainer.functions.relu(self.conv1(x)), ksize=2)
        h = chainer.functions.max_pooling_2d(chainer.functions.relu(self.conv2(h)), ksize=2)
        return h


class MyDiscriminator(chainer.Chain):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        with self.init_scope():
            self.linear = chainer.links.Linear(None, 10)

    def __call__(self, x):
        return chainer.functions.relu(self.linear(x))


class MyNet():
    def __init__(self, extr, disc, gpu=-1):
        self.extr = extr
        self.disc = disc
        self.mode = 'pre_train'
        if gpu >= 0:
            self.extr.to_gpu()
            self.disc.to_gpu()

    def __call__(self, x):
        z = self.extr(x)
        if self.mode == 'fin_train':
            z.creator = None
        o = self.disc(z)
        return o

    def chmod(self, mode):
        self.mode = mode


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = chainer.functions.softmax_cross_entropy(y, t)
        accuracy = chainer.functions.accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MazeMaze train')
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--epoch', '-e', type=int, default=20)

    parser.add_argument('--pre_dataset', '-pd', default='fashion-mnist')
    parser.add_argument('--fin_dataset', '-fd', default='mnist')

    args = parser.parse_args()
    def make_dataset(dataset):
        if dataset == 'mnist':
            train, test = chainer.datasets.get_mnist(ndim=3)
        elif dataset == 'cifar-10':
            pass
            # TODO: resize image size
            # train, test = chainer.datasets.get_cifar10(ndim=3)
        elif dataset == 'fashion-mnist':
            data = input_data.read_data_sets('../fashion-mnist/data/fashion')
            train = [(x.reshape(28, 28)[np.newaxis, :], y.astype(np.int32)) for x, y in zip(data.train.images, data.train.labels)]
            test = [(x.reshape(28, 28)[np.newaxis, :], y.astype(np.int32)) for x, y in zip(data.test.images, data.test.labels)]
        else:
            raise
        return train, test

    train, test = make_dataset(args.pre_dataset)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batch_size, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batch_size, shuffle=True, repeat=False)
    net = MyNet(MyFeatureExtrator(), MyDiscriminator(), gpu=args.gpu)
    model = Classifier(net)
    optimizer = chainer.optimizers.AdaGrad()
    optimizer.setup(model)
    updater = chainer.training.StandardUpdater(train_iter, optimizer=optimizer, device=args.gpu)
    pre_trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out='pre_result')
    pre_trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    pre_trainer.extend(extensions.LogReport())
    pre_trainer.extend(extensions.PlotReport(['epoch', 'main/accuray', 'validation/main/accuracy']))
    pre_trainer.extend(extensions.ProgressBar())
    pre_trainer.run()

    train, test = make_dataset(args.fin_dataset)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batch_size, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batch_size, shuffle=True, repeat=False)

    net.chmod('fin_train')
    model = Classifier(net)
    optimizer = chainer.optimizers.AdaGrad()
    optimizer.setup(model)
    updater = chainer.training.StandardUpdater(train_iter, optimizer=optimizer, device=args.gpu)
    fin_trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out='fin_result')
    fin_trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    fin_trainer.extend(extensions.LogReport())
    fin_trainer.extend(extensions.PlotReport(['epoch', 'main/accuray', 'validation/main/accuracy']))
    fin_trainer.extend(extensions.ProgressBar())
    fin_trainer.run()
