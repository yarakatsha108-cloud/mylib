from Layers import *
import numpy as np

import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None

        self.params = [self.gamma, self.beta]
        self.grads = [None, None]

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            _, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 1e-7)

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum(dxn * self.xc, axis=0) / (self.std ** 2)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar

        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.grads[0] = dgamma
        self.grads[1] = dbeta

        return dx
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask