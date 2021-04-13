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

    def mutate(self, population: List[Chromosome]) -> List[Chromosome]:
        for chromosome in population:
            i, j = np.random.choice(chromosome.permutation.shape[0],
                                    size=2,
                                    replace=True)
            chromosome.permutation[[i, j]] = chromosome.permutation[[j, i]]
        return population
