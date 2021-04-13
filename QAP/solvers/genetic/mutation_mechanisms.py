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

    def __init__(self, mutation_prob: float = 0.5):
        self.mutation_prob = mutation_prob

    def mutate(self, population: List[Chromosome]) -> List[Chromosome]:
        for chromosome in population:
            if np.random.sample() <= self.mutation_prob:
                i, j = np.random.choice(chromosome.permutation.shape[0],
                                        size=2,
                                        replace=False)
                chromosome.permutation[[i, j]] = chromosome.permutation[[j, i]]
                chromosome.calculate_cost()
        return population
