import numpy as np
from .chromosome import Chromosome
from typing import Tuple, List, Set, Union
from QAP.utils.solver_utils import generate_random_solutions, get_liveness_score


class CrossoverMechanism:
    """Abstract class for crossover mechanism.
    Each descendant must implement `crossover` function that gets called with `()` syntax
    """
    def crossover(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.crossover(*args, **kwargs)


class OrderedCrossover(CrossoverMechanism):
    """Ordered crossover.
    Two partners are selected from population based on liveness factor.
    Method leaves continuous subset of genes copied from one partner and order of other genes from second partner.
    Each breading procedure generates 2 child
    Args:
        crossover_count: (optional) number of breading interactions (creates 2*crossover_count child)
        crossover_retry: (optional) number of retries to generate new chromosomes before falling to random chromosome generation # noqa
    """
    def __init__(self, crossover_count: int = 100, crossover_retry: int = 50):
        self.count = crossover_count
        self.retry_count = crossover_retry

    def _cross_genes(self, a: Chromosome, b: Chromosome) -> Chromosome:
        """Procedure of breading:
        1. Select span [i, j) that defines continuous subset of genes that are copied from parent a
        2. Select other not used genes from parent b with the same order
        Example: a: (3, 5, <2, 4,> 1)
                 b: (2, 1, 4, 5, 3)
                 i, j = 2, 4
                 ------------------
                 r: (1, 5, <2, 4,> 3)
        Args:
            a: first partner (for continuous subset)
            b: second partner (for order of genes)
        Returns:
            New chromosome (Note: we don't calculate cost because we can discard this element in next steps)
        """
        new_perm = np.zeros_like(a.permutation)
        start, finish = np.sort(np.random.choice(new_perm.shape[0], size=2, replace=False))
        parent_mask = np.zeros_like(new_perm, dtype=bool)
        parent_mask[start:finish] = True
        new_perm[parent_mask] = a.permutation[parent_mask]
        new_perm[~parent_mask] = b.permutation[np.isin(b.permutation, a.permutation[parent_mask], invert=True)]

        return Chromosome(new_perm, calculate_cost=False)

    def cross_genes(self, a: Chromosome, b: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Creates 2 child from partners.
        Args:
            a: first partner
            b: second partner
        Returns:
            pair of new chromosomes
        """
        return self._cross_genes(a, b), self._cross_genes(b, a)

    def crossover(self, population: np.ndarray) -> np.ndarray:
        """Perform `crossover_count` crossovers based on liveness scores.
        Args:
            population: current population of chromosomes
        Returns:
            New population created by previously described procedure.
        """
        descendants: Set[Chromosome] = set() # noqa
        costs = np.array([p.cost for p in population])
        probs = get_liveness_score(costs)

        for _ in range(self.count):
            child = self.cross_genes(
                *np.random.choice(population, replace=False, size=2, p=probs))
            c = 0
            while child[0] in descendants and child[1] in descendants and c < self.retry_count:
                child = self.cross_genes(
                    *np.random.choice(population, replace=False, size=2, p=probs))
                c += 1
            if c == self.retry_count:
                child = [
                    Chromosome(p, calculate_cost=False)
                    for p in generate_random_solutions(child[0].permutation.shape[0], 2)
                ]

            descendants.update(child)
        return np.array([d.calculate_cost() for d in descendants])  # don't forget to calculate cost
