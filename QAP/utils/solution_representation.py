import numpy as np
from typing import Callable


class SolutionRepresentation:
    """Abstract solution representation for problem.
    Args:
        objective: static function that should be set to objective function of our choice
        permutation: permutation that represents solution
        cost: cost of the solution represented by permutation
    """
    objective: Callable[[np.ndarray], int] = None

    def __init__(self, permutation, calculate_cost=True):
        self.permutation = permutation
        self.cost = self.objective(permutation) if calculate_cost else -1

    def calculate_cost(self):
        """Calculates objective function for permutation.
        Returns:
            Chromosome object after setting cost
        """
        self.cost = self.objective(self.permutation)
        return self

    def __str__(self):
        return f'result: {self.cost}, permutation: {self.permutation}'
