from Optimization import *
from Layers import *
from Core.neural_network import NeuralNetwork
from Loss import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class Trainer:
    def __init__(self, model: NeuralNetwork,
                 x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01}):

        self.model = model
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.train_size = x_train.shape[0]

        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)

        self.optimizer = globals()[optimizer](**optimizer_param)

        self.current_iter = 0
        self.current_epoch = 0

    def fit_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.model.gradient(x_batch, t_batch)

        params = {}
        grads_to_update = {}
        for key, layer in self.model.layers.items():
            if hasattr(layer, 'params') and layer.params is not None:
                for i, param in enumerate(layer.params):
                    param_key = f'{key}_param_{i}'
                    params[param_key] = layer.params[i]
                    if hasattr(layer, 'grads') and layer.grads[i] is not None:
                        grads_to_update[param_key] = layer.grads[i]

        self.optimizer.update(params, grads_to_update)

        for key, layer in self.model.layers.items():
            if hasattr(layer, 'params') and layer.params is not None:
                for i in range(len(layer.params)):
                    layer.params[i] = params[f'{key}_param_{i}']

        if self.current_iter % self.iter_per_epoch == 0:
            train_loss = self.model.loss(self.x_train, self.t_train)
            test_loss = self.model.loss(self.x_test, self.t_test)
            print(
                f'Epoch {self.current_epoch}: '
                f'Train Loss = {train_loss:.4f}, '
                f'Test Loss = {test_loss:.4f}'
            )
            self.current_epoch += 1

        self.current_iter += 1

    def fit(self):
        total_iters = self.epochs * self.iter_per_epoch
        for _ in range(total_iters):
            self.fit_step()
