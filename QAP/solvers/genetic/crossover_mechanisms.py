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

    def cross_genes(self, a: Chromosome, b: Chromosome):
        new_perm = np.zeros_like(a.permutation)
        start, finish = np.sort(
            np.random.choice(new_perm.shape[0], size=2, replace=False))
        parent_mask = np.zeros_like(new_perm, dtype=bool)
        parent_mask[start:finish] = True
        new_perm[parent_mask] = a.permutation[parent_mask]
        new_perm[~parent_mask] = b.permutation[np.isin(
            b.permutation, a.permutation[parent_mask], invert=True)]
        return Chromosome(new_perm, calculate_cost=False)

    def crossover(self, population: List[Chromosome]) -> List[Chromosome]:
        descendants = set()
        probs = np.array([p.cost for p in population])
        probs = probs / np.sum(probs)

        for _ in range(self.count):
            child = self.cross_genes(
                *random.choices(population, weights=probs, k=2))
            c = 0
            while child in descendants and c < self.retry_count:
                child = self.cross_genes(
                    *random.choices(population, weights=probs, k=2))
                c += 1
            if c == self.retry_count:
                child = Chromosome(generate_random_solutions(
                    child.permutation.shape[0], 1).squeeze(),
                                   calculate_cost=False)
            descendants.add(child)
        return [d.calculate_cost() for d in descendants]
