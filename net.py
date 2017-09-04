import chainer


class VarNet(chainer.ChainList):
    def __init__(self, n_conv, channels=None):
        super(VarNet, self).__init__()
        if channels is None:
            channels = [16 + 3*i for i in range(n_conv)]

        if len(channels) != n_conv:
            raise ValueError("channels {} are not equal to n_conv {}".format(len(channels), n_conv))

        for i in range(n_conv):
            self.append(chainer.links.Convolution2D(None, channels[i], ksize=3))

    def __call__(self, x):
        h = x
        #print(self.links())
        #exit()
        for link in self.children():
            h = chainer.functions.max_pooling_2d(
                chainer.functions.relu(link(h)), ksize=2)
        return h
