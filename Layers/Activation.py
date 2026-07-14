from Layers.Base import Base
import numpy as np

class Linear(Base):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x

    def backward(self, doubt):
        return doubt
    
class Relu(Base):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        #return np.maximum(0, x)
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, doubt ):
        doubt[self.mask] = 0
        dx = doubt
        return dx

class Sigmoid(Base):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, doubt):
        dx = doubt * (1.0 - self.out) * self.out
        return dx

class Tanh(Base):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, doubt):
        dx = doubt * (1.0 - self.out ** 2)
        return dx
    
class Softmax(Base):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        out = exp_x / sum_exp_x
        self.out = out
        return out
#غالبا ما احتاج له لان عندي SoftmaxWithLoss
    def backward(self, doubt):
        batch_size = doubt.shape[0]
        dx = np.empty_like(doubt)

        for i in range(batch_size):
            y = self.out[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
            dx[i] = np.dot(jacobian_matrix, doubt[i])

        return dx
