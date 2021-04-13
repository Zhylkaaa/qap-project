import numpy as np
from .chromosome import Chromosome


class SelectionMechanism:

    def select(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.select(*args, **kwargs)


class RouletteWheel(SelectionMechanism):

    def __init__(self, selection_size: int = 100):
        self.selection_size = selection_size

    def select(self,
               population: np.ndarray) -> np.ndarray:
        probs = np.array([p.cost for p in population])
        probs = probs / np.sum(probs)
        return np.random.choice(population,
                                replace=False,
                                size=self.selection_size,
                                p=probs)
