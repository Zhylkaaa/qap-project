import numpy as np
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

    def __init__(self, selection_size):
        self.selection_size = selection_size

    def select(self, population: List[Chromosome]):
        pass
