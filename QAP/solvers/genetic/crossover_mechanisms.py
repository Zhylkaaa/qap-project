import numpy as np
from typing import List
from .chromosome import Chromosome


class CrossoverMechanism:

    def crossover(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.crossover(*args, **kwargs)


class OrderedCrossover(CrossoverMechanism):

    def __init__(self, crossover_count: int = 100):
        self.count = crossover_count

    def crossover(self, population: List[Chromosome]):
        pass
