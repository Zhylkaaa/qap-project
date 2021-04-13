from typing import Callable
from functools import partial
import numpy as np
import re
from QAP.utils.solver_utils import generate_random_solutions
from .selection_mechanisms import SelectionMechanism, RouletteWheel
from .mutation_mechanisms import MutationMechanism, SwapMutation
from .crossover_mechanisms import CrossoverMechanism, OrderedCrossover
from .chromosome import Chromosome
from QAP.objective import objective
from tqdm import tqdm


def genetic_solver(n: int,
                   dist: np.ndarray,
                   cost: np.ndarray,
                   objective: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                       int] = objective,
                   max_iterations: int = 100,
                   population_size: int = 100,
                   crossover_mechanism: CrossoverMechanism = OrderedCrossover,
                   mutation_mechanism: MutationMechanism = SwapMutation,
                   selection_mechanism: SelectionMechanism = RouletteWheel,
                   bad_epoch_patience: int = 20,
                   **kwargs) -> np.ndarray:
    """Genetic algorithm solver for QAP.
    Args:
        n: size of a problem
        dist: 2d distance matrix
        cost: 2d cost matrix
        objective: (optional) objective function that accepts dist, cost and permutation and returns calculated objective
        max_iterations: (optional) computational budget
        population_size: (optional) size of the population
        crossover_mechanism: (optional) class that implements CrossoverMechanism and provides crossover mechanism
        mutation_mechanism: (optional) class that implements MutationMechanism and provides mutation mechanism
        selection_mechanism: (optional) class that implements SelectionMechanism and provides selection mechanism
        bad_epoch_patience: (optional) number of allowed bad epochs before perturbation
        kwargs: (optional) used to pass optional arguments to selection_mechanism
    Returns:
        Permutation that achieves the best objective score on the task
    """
    # generate initial population
    objective = partial(objective, dist, cost)
    Chromosome.objective = objective

    population = [
        Chromosome(solution)
        for solution in generate_random_solutions(n, size=population_size)
    ]

    # maybe cut prefix off?
    crossover_args = {
        key: value
        for key, value in kwargs.items()
        if re.match('crossover_*', key)
    }
    mutation_args = {
        key: value
        for key, value in kwargs.items()
        if re.match('mutation_*', key)
    }
    selection_args = {
        key: value
        for key, value in kwargs.items()
        if re.match('selection_*', key)
    }

    crossover = crossover_mechanism(**crossover_args)
    mutation = mutation_mechanism(**mutation_args)
    selection = selection_mechanism(**selection_args)

    best_solution = max(population)
    bad_epoch_counter = 0
    # TODO: probably use convergence criterion
    for _ in tqdm(range(max_iterations)):
        new_generation = mutation(crossover(population))
        population.extend(new_generation)

        population = selection(population)
        population_best = max(population)
        if best_solution.cost > population_best.cost:
            best_solution = population_best
        else:
            bad_epoch_counter += 1
            if bad_epoch_counter == bad_epoch_patience:
                population.extend([
                    Chromosome(solution) for solution in
                    generate_random_solutions(n, size=population_size)
                ])
                population_best = max(population)
                best_solution = max(best_solution, population_best)
                population = selection(population)
                bad_epoch_counter = 0

    return best_solution
