from Layers.Activation import Softmax
from Layers.Base import Base
import numpy as np

class MeanSquaredError(Base):
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.batch_size = None

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_true.shape[0]
        loss = np.sum((y_true - y_pred) ** 2) / self.batch_size
        return loss

    def backward(self, dout=1):
        dx = (-2 / self.batch_size) * (self.y_true - self.y_pred) * dout
        return dx
    
class SoftmaxWithLoss(Base):
    def __init__ (self):
        super().__init__()
        self.loss = None
        self.y = None
        self.t = None 

    def forward(self, x, t):
        self.t = t
        self.y = Softmax().forward(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, doubt=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size