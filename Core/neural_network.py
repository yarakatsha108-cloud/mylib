from collections import OrderedDict
import numpy as np
from Layers import *

class NeuralNetwork :
    def __init__(self, layers):
        #استخدمت ال OrderedDict عشان أحافظ على ترتيب الطبقات بشكل ديناميكي 
        self.layers = OrderedDict()
        # self.params = []
        # self.grads = []
        last_Layer = len(layers) - 1
        #هون عم انشاء الطبقات بشكل يناميكي
        for idx, layer in enumerate(layers):
            self.layers[f'Layer_{idx}'] = layer
            # if hasattr(layer, 'params'):
            #     for param in layer.params:
            #         self.params.append(param)
            #     for grad in layer.grads:
            #         self.grads.append(grad)
            if idx == last_Layer:
                self.last_Layer = layer

    def predict(self, x):
        for layer in list(self.layers.values())[:-1]:
            x = layer.forward(x)
        return x
    def loss(self , x , t ):
        y = self.predict(x)
        return self.last_Layer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    def gradient(self, x, t):
        self.loss(x, t)
        # doubt = 1
        doubt = self.last_Layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            doubt = layer.backward(doubt)
        grads = {}
        for key, layer in self.layers.items():
            if hasattr(layer, 'dW'):
                grads[f'{key}_param_0'] = layer.dW
                if layer.b is not None:
                    grads[f'{key}_param_1'] = layer.db
        return grads