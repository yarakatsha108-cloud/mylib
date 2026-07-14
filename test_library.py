# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from Core.neural_network import NeuralNetwork
# from Core.trainer import Trainer
# from Layers.Dense import Dense_Affine
# from Layers.Activation import Sigmoid, Relu
# from Layers.Accelerate import BatchNormalization
# from Loss.Loss_classes import SoftmaxWithLoss

# # def he_init(size_in, size_out):
# #     return np.random.randn(size_in, size_out) * np.sqrt(2.0 / size_in)

# # def glorot_init(size_in, size_out):
# #     return np.random.randn(size_in, size_out) * np.sqrt(2.0 / (size_in + size_out))

# def load_data():
#     iris = load_iris()
#     x = iris.data
#     t = iris.target

#     encoder = OneHotEncoder(sparse_output=False)
#     t_onehot = encoder.fit_transform(t.reshape(-1, 1))

#     x_train, x_test, t_train, t_test = train_test_split(
#         x, t_onehot, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)

#     return x_train, t_train, x_test, t_test

# def test_neural_network():

#     Layers = [
#     Dense_Affine(4, 10, weight_init="he"),
#     BatchNormalization(gamma=np.ones(10), beta=np.zeros(10)),
#     Relu(),
#     Dense_Affine(10, 3, weight_init="glorot"),
#     SoftmaxWithLoss()
#     ]
    
#     model = NeuralNetwork(Layers)

#     x_dummy = np.random.rand(5, 4)
#     t_dummy = np.array([[1, 0, 0],
#                         [0, 1, 0],
#                         [0, 0, 1],
#                         [1, 0, 0],
#                         [0, 1, 0]])
#     loss = model.loss(x_dummy, t_dummy)
#     assert loss > 0, "Loss should be positive"
#     y_pred = model.predict(x_dummy)
#     print("Predictions:", y_pred)
#     x_train, t_train, x_test, t_test = load_data()
#     trainer = Trainer(
#         model=model,
#         x_train=x_train,
#         t_train=t_train,
#         x_test=x_test,
#         t_test=t_test,
#         epochs=100,
#         mini_batch_size=16,
#         optimizer='SGD',
#         optimizer_param={'lr': 0.01},
#         # learning_rate=0.01, 
#         # verbose=False
#     )
#     trainer.fit()
#     test_accuracy = model.accuracy(x_test, t_test)
#     print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# if __name__ == "__main__":
#     test_neural_network()

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Core.neural_network import NeuralNetwork
from Core.trainer import Trainer
from Core.hyperparameter import HyperparameterTuning
from Layers.Dense import Dense_Affine
# from Layers.Activation import Sigmoid, Relu , Softmax
from Layers.Activation import *
from Layers.Accelerate import BatchNormalization
from Layers.Accelerate import Dropout
from Loss.Loss_classes import SoftmaxWithLoss

def load_data():
    iris = load_iris()
    x = iris.data
    t = iris.target

    encoder = OneHotEncoder(sparse_output=False)
    t_onehot = encoder.fit_transform(t.reshape(-1, 1))

    x_train, x_test, t_train, t_test = train_test_split(
        x, t_onehot, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, t_train, x_test, t_test

def build_model(config):
    Layers = [
        # Dense_Affine(4, 10, weight_init="he"),
        # Sigmoid(),
        # Relu(),
        # BatchNormalization(gamma=np.ones(10), beta=np.zeros(10)),
        # Dropout(dropout_ratio=0.8),
        # Sigmoid(),
        # Dropout(dropout_ratio=0.5),
        # Dense_Affine(10, 3, weight_init="glorot"),
        # SoftmaxWithLoss()
        # Softmax()
        Dense_Affine(4, 10, weight_init="he"),
        Sigmoid(),
        BatchNormalization(gamma=np.ones(10), beta=np.zeros(10)),
        Relu(),
        Dense_Affine(10, 3, weight_init="glorot"),
        SoftmaxWithLoss()
    ]
    model = NeuralNetwork(Layers)
    return model

def test_neural_network_with_hyper():
    x_train, t_train, x_test, t_test = load_data()

    search_space = {
        "lr": [0.01, 0.001, 0.005],
        "batch_size": [16, 32],
        "epochs": [50, 100]
    }

    tuner = HyperparameterTuning(build_model, Trainer, search_space)

    
    best_config, best_score = tuner.tune(x_train, t_train, x_test, t_test)

    print("Best config:", best_config)
    print(f"Best accuracy: {best_score * 100:.2f}%")

    model = build_model(best_config)
    trainer = Trainer(
        model=model,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=best_config.get("epochs", 100),
        mini_batch_size=best_config.get("batch_size", 16),
        optimizer='SGD',
        optimizer_param={'lr': best_config.get("lr", 0.01)}
    )
    trainer.fit()
    test_accuracy = model.accuracy(x_test, t_test)
    print(f"Test Accuracy with best config: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_neural_network_with_hyper()
