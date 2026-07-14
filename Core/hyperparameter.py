from Optimization import *
from Layers import *
from Core.neural_network import *
from Loss import *
# import numpy as np
class HyperparameterTuning:
    def __init__(self, build_model_fn, trainer_class, search_space):
        self.build_model_fn = build_model_fn
        self.trainer_class = trainer_class
        self.search_space = search_space
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')

    def tune(self, x_train, t_train, x_test, t_test):
        import itertools

        keys, values = zip(*self.search_space.items())
        for v in itertools.product(*values):
            config = dict(zip(keys, v))

            model = self.build_model_fn(config)

            trainer = self.trainer_class(
                model,
                x_train, t_train,
                x_test, t_test,
                epochs=config.get("epochs", 20),
                mini_batch_size=config.get("batch_size", 100),
                optimizer=config.get("optimizer", "SGD"),
                optimizer_param={"lr": config.get("lr", 0.01)}
            )

            trainer.fit()
            acc = model.accuracy(x_test, t_test)

            self.results.append((config, acc))

            if acc > self.best_score:
                self.best_score = acc
                self.best_config = config

        return self.best_config, self.best_score

    def get_all_results(self):
        return self.results
