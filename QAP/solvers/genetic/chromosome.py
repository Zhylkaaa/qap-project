import numpy as np
from typing import Callable
from QAP.utils.solution_representation import SolutionRepresentation


class Chromosome(SolutionRepresentation):
    """Represents solution for problem.
    Args:
        objective: static function that should be set to objective function of our choice
        permutation: permutation that represents solution
        cost: cost of the solution represented by permutation
        calculate_cost: (optional) weather to calculate cost function (we don't want to do it for elements that can be discarded immediately) # noqa
    """
    def __init__(self, permutation: np.ndarray, calculate_cost=True):
        super().__init__(permutation, calculate_cost)

    # basic utility functions
    def __gt__(self, other):
        return self.cost < other.cost

    def __lt__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return np.all(self.permutation == other.permutation)

    def __hash__(self):
        return hash(tuple(self.permutation))
