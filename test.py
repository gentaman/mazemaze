from tensorflow.examples.tutorials.mnist import input_data
import chainer
from chainer.training import extensions
import numpy as np
import argparse
from PIL import Image
import net
import os
from lrp import LRP, RetainOutputHook
import matplotlib.pyplot as plt


class MyFeatureExtrator(chainer.Chain):
    def __init__(self):
        super(MyFeatureExtrator, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(
                        in_channels=1, out_channels=16, ksize=5, stride=1)
            self.conv2 = chainer.links.Convolution2D(
                        in_channels=None, out_channels=32, ksize=3, stride=1)

    def __call__(self, x):
        h = chainer.functions.max_pooling_2d(
            chainer.functions.relu(self.conv1(x)), ksize=2)
        h = chainer.functions.max_pooling_2d(
            chainer.functions.relu(self.conv2(h)), ksize=2)
        return h


class MyDiscriminator(chainer.Chain):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        with self.init_scope():
            self.linear = chainer.links.Linear(None, 10)

    def __call__(self, x):
        return self.linear(x)


class MyNet(chainer.Chain):
    def __init__(self, extr, disc, gpu=-1):
        super(MyNet, self).__init__()
        with self.init_scope():
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
    parser.add_argument('--epsilon', '-epsilon', type=float, default=100)

    parser.add_argument('--pre_dataset', '-pd', default='cifar-10')
    parser.add_argument('--fin_dataset', '-fd', default='mnist')

    args = parser.parse_args()
    def make_dataset(dataset):
        if dataset == 'mnist':
            train, test = chainer.datasets.get_mnist(ndim=3)
        elif dataset == 'cifar-10':
            # resize image size to 28x28
            train, test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=255.)
            tmp_array = []
            for array, label in train:
                image = np.asarray(
                        Image.fromarray(np.uint8(array.transpose(1, 2, 0))).resize((28, 28)).convert('L')
                        ).reshape(1, 28, 28).astype(np.float32)/255.
                tmp_array.append((image, label))
            train = tmp_array
            tmp_array = []
            for array, label in test:
                image = np.asarray(
                        Image.fromarray(np.uint8(array.transpose(1, 2, 0))).resize((28, 28)).convert('L')
                        ).reshape(1, 28, 28).astype(np.float32)/255.
                tmp_array.append((image, label))
            test = tmp_array
            del tmp_array

        elif dataset == 'fashion-mnist':
            data = input_data.read_data_sets('../fashion-mnist/data/fashion')
            train = [(x.reshape(28, 28)[np.newaxis, :], y.astype(np.int32)) for x, y in zip(data.train.images, data.train.labels)]
            test = [(x.reshape(28, 28)[np.newaxis, :], y.astype(np.int32)) for x, y in zip(data.test.images, data.test.labels)]
        else:
            raise
        return train, test

    def do_training(network, train, test, optimizer=chainer.optimizers.AdaGrad(), **kwargs):
        train_iter = chainer.iterators.SerialIterator(train, batch_size=kwargs["batch_size"], shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, batch_size=kwargs["batch_size"], shuffle=True, repeat=False)
        model = Classifier(network)
        optimizer.setup(model)
        updater = chainer.training.StandardUpdater(train_iter, optimizer=optimizer, device=kwargs["gpu"])
        trainer = chainer.training.Trainer(updater, (kwargs["epoch"], 'epoch'), out= kwargs["out"])
        trainer.extend(extensions.Evaluator(test_iter, model, device=kwargs["gpu"]))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PlotReport(['main/accuray', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

    def view_lrp(network, inputs, epsilon=0, out='result'):
        with RetainOutputHook():
            outputs = network(inputs)
        results = LRP(outputs, epsilon)

        fig, axs = plt.subplots(
            results.shape[0], 2, figsize=(2*2, results.shape[0]*2), subplot_kw={'xticks': [], 'yticks': []})
        for i in range(results.shape[0]):
            im = inputs[i][0]
            axs[i, 0].imshow(im, vmin=im.min(), vmax=im.max(), cmap='gray')
            im = results[i][0]
            axs[i, 1].imshow(im, vmin=im.min(), vmax=im.max(), cmap='plasma')
        plt.savefig(os.path.join(out, 'visualize_byLRP.png'))

    def mazemaze(args, i=1):
        train, test = make_dataset(args.pre_dataset)
        network = MyNet(net.VarNet(i), MyDiscriminator(), gpu=args.gpu)
        do_training(network, train, test,
                    out=os.path.join('pre_result_' + args.pre_dataset, str(i)),
                    batch_size=args.batch_size,
                    gpu=args.gpu,
                    epoch=args.epoch
                    )
        view_lrp(network.to_cpu(),
                 np.array(map(lambda x: x[0], train[10:20])),
                 epsilon=args.epsilon,
                 out=os.path.join('pre_result_' + args.pre_dataset, str(i)))

        train, test = make_dataset(args.fin_dataset)
        network.chmod('fin_train')
        do_training(network, train, test,
                    out=os.path.join('pre_result_' + args.pre_dataset + '_fin_result_' + args.fin_dataset, str(i)),
                    batch_size=args.batch_size,
                    gpu=args.gpu,
                    epoch=args.epoch
                    )
        network.chmod(None)
        view_lrp(network.to_cpu(),
                 np.array(map(lambda x: x[0], train[10:20])),
                 epsilon=args.epsilon,
                 out=os.path.join('pre_result_' + args.pre_dataset + '_fin_result_' + args.fin_dataset, str(i)))

    for i in range(1, 4):
        mazemaze(args, i)
