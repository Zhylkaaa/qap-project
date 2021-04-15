import numpy as np
from typing import Callable


class Chromosome:
    """Represents solution for problem.
    Args:
        objective: static function that should be set to objective function of our choice
        permutation: permutation that represents solution
        cost: cost of the solution represented by permutation
        calculate_cost: (optional) weather to calculate cost function (we don't want to do it for elements that can be discarded immediately) # noqa
    """

    objective: Callable[[np.ndarray], int] = None

    def __init__(self, permutation: np.ndarray, calculate_cost=True):
        self.permutation = permutation
        self.cost = Chromosome.objective(permutation) if calculate_cost else -1

    def calculate_cost(self):
        """Calculates objective function for permutation.
        Returns:
            Chromosome object after setting cost
        """
        self.cost = Chromosome.objective(self.permutation)
        return self

    # basic utility functions
    def __gt__(self, other):
        return self.cost < other.cost

    def __lt__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return np.all(self.permutation == other.permutation)

    def __hash__(self):
        return hash(tuple(self.permutation))

    def __str__(self):
        return f'result: {self.cost}, permutation: {self.permutation}'
