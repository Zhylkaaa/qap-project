import numpy as np
import random
from typing import List
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

    def select(self, population: List[Chromosome]):
        probs = np.array([p.cost for p in population])
        probs = probs / np.sum(probs)
        return random.choices(population, k=self.selection_size, weights=probs)
