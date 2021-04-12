from typing import Callable
import numpy as np
from QAP.utils.solver_utils import generate_random_solutions
from .selection_mechanisms import SelectionMechanism, RouletteWheel
from QAP.objective import objective


def genetic_solver(n:int,
                   dist: np.ndarray,
                   cost: np.ndarray,
                   objective: Callable[[np.ndarray, np.ndarray, np.ndarray], int] = objective,
                   max_iterations: int = 100,
                   population_size: int = 100,
                   crossover_prob: float = 0.8,
                   mutation_prob: float = 0.1,
                   selection_mechanism: SelectionMechanism = RouletteWheel,
                   **kwargs) -> np.ndarray:
    """Genetic algorithm solver for QAP.
    Args:
        n: size of a problem
        dist: 2d distance matrix
        cost: 2d cost matrix
        objective: (optional) objective function that accepts dist, cost and permutation and returns calculated objective
        max_iterations: (optional) computational budget
        population_size: (optional) size of the population
        crossover_prob: (optional) probability of crossover between 2 genes
        mutation_prob: (optional) probability of random mutation in genes
        selection_mechanism: (optional) class that implements SelectionMechanism and wrowides selection mechanism
        kwargs: (optional) used to pass optional arguments to selection_mechanism
    Returns:
        Permutation that achieves the best objective score on the task
    """
    initial_population = generate_random_solutions(n, size=population_size)
    pass
