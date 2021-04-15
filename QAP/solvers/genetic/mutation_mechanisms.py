import numpy as np
from typing import List
from .chromosome import Chromosome


class MutationMechanism:
    """Abstract class for mutation mechanism.
    Each descendant must implement `mutate` function that gets called with `()` syntax
    """

    def mutate(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.mutate(*args, **kwargs)


class SwapMutation(MutationMechanism):
    """Random swap mutation.
    Swaps 2 uniformly selected genes in chromosome with certain probability

    Args:
         mutation_prob: (optional) probability of gen swap
    """
    def __init__(self, mutation_prob: float = 0.5):
        self.mutation_prob = mutation_prob

    def mutate(self,
               population: np.ndarray) -> np.ndarray:
        """Perform random swap mutation on population
        Args:
            population: list of genes that might be mutated
        Returns:
            mutated population.
        """
        for chromosome in population:
            if np.random.sample() <= self.mutation_prob:
                i, j = np.random.choice(chromosome.permutation.shape[0],
                                        size=2,
                                        replace=False)
                chromosome.permutation[[i, j]] = chromosome.permutation[[j, i]]
                chromosome.calculate_cost()
        return population
