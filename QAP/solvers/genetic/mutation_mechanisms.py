import numpy as np
from typing import List
from .chromosome import Chromosome


class MutationMechanism:

    def mutate(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.mutate(*args, **kwargs)


class SwapMutation(MutationMechanism):

    def __init__(self, mutation_prob: float = 0.2):
        self.prob = mutation_prob

    def mutate(self, population: List[Chromosome]):
        pass
