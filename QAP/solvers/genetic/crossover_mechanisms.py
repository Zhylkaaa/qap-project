import numpy as np
import random
from typing import List
from .chromosome import Chromosome
from QAP.utils.solver_utils import generate_random_solutions


class CrossoverMechanism:

    def crossover(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.crossover(*args, **kwargs)


class OrderedCrossover(CrossoverMechanism):

    def __init__(self, crossover_count: int = 100, crossover_retry: int = 50):
        self.count = crossover_count
        self.retry_count = crossover_retry

    def _cross_genes(self, a: Chromosome, b: Chromosome):
        new_perm = np.zeros_like(a.permutation)
        start, finish = np.sort(
            np.random.choice(new_perm.shape[0], size=2, replace=False))
        parent_mask = np.zeros_like(new_perm, dtype=bool)
        parent_mask[start:finish] = True
        new_perm[parent_mask] = a.permutation[parent_mask]
        new_perm[~parent_mask] = b.permutation[np.isin(
            b.permutation, a.permutation[parent_mask], invert=True)]

        return Chromosome(new_perm, calculate_cost=False)

    def cross_genes(self, a: Chromosome, b: Chromosome):
        return [self._cross_genes(a, b), self._cross_genes(b, a)]

    def crossover(self, population: List[Chromosome]) -> List[Chromosome]:
        descendants = set()
        probs = np.array([p.cost for p in population])
        probs = probs / np.sum(probs)

        for _ in range(self.count):
            children = self.cross_genes(
                *random.choices(population, weights=probs, k=2))
            c = 0
            while children[0] in descendants and children[1] in descendants and c < self.retry_count:
                children = self.cross_genes(
                    *random.choices(population, weights=probs, k=2))
                c += 1
            if c == self.retry_count:
                children = [
                    Chromosome(p, calculate_cost=False)
                    for p in generate_random_solutions(
                        children[0].permutation.shape[0], 1)
                ]
            descendants.update(children)
        return [d.calculate_cost() for d in descendants]
