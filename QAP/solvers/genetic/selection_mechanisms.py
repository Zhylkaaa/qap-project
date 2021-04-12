import numpy as np


class SelectionMechanism:

    def select(self, generation: np.ndarray):
        raise NotImplementedError('this is abstract class method and should be implemented by descendants')

    def __call__(self, *args, **kwargs):
        return self.select(*args, **kwargs)


class RouletteWheel(SelectionMechanism):
    def __init__(self, ):
        pass

    def select(self, generation: np.ndarray):
        pass
