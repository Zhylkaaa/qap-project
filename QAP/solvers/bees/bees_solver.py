# papers:
# * https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note
# * https://link.springer.com/chapter/10.1007/978-3-319-23437-3_53
import numpy as np
from typing import Callable, Union, Type
from functools import partial
import numpy as np
import re
from QAP.objective import objective
from tqdm import tqdm
from ..selection_mechanisms import SelectionMechanism, BestFit
from ..mutation_mechanisms import MutationMechanism, UniformMutationScheduler, SwapMutation
from .location import Location
from QAP.utils.solver_utils import generate_random_solutions
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import time


def search_elite(elite, elite_search_size=1):
    return elite.search_neighbourhood_elite(elite_search_size)


def search_selected(selected, selected_search_size=1):
    return selected.search_neighbourhood(selected_search_size)


def init_fn():
    s = int(time.time() * 100000) % 1000
    np.random.seed(s)
    print(f'worker seed {np.random.get_state()[1][0]}')


def bees_solver(n: int,
                dist: np.ndarray,
                cost: np.ndarray,
                objective: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                    int] = objective,
                max_iterations: int = 100,
                population_size: int = 100,
                verbose: bool = True,
                print_every: int = 100,
                mutation_mechanism: Type[MutationMechanism] = UniformMutationScheduler,
                selection_mechanism: Type[SelectionMechanism] = BestFit,
                solution_lifetime: int = -1,
                elite_population: Union[float, int] = 0.01,
                selected_population: Union[float, int] = 0.49,
                elite_search_size: Union[float, int] = 0.02,
                selected_search_size: Union[float, int] = 0.01,
                bad_epoch_patience: int = 20,
                thread_pool_size: int = 6,
                **kwargs) -> np.ndarray:
    objective = partial(objective, dist, cost)
    Location.objective = objective

    # get args for each component
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

    def float_to_int(num: Union[float, int]) -> int:
        if isinstance(num, float):
            return int(population_size * num)
        return num

    elite_population = float_to_int(elite_population)
    selected_population = float_to_int(selected_population)
    elite_search_size = float_to_int(elite_search_size)
    selected_search_size = float_to_int(selected_search_size)

    mutation = mutation_mechanism(**mutation_args)
    selection = selection_mechanism(selected_population, **selection_args)

    Location.solution_lifetime = solution_lifetime

    population = np.array([
        Location(solution, mutation)
        for solution in generate_random_solutions(n, size=population_size)
    ])

    p = Pool(thread_pool_size, initializer=init_fn)
    #p = ThreadPoolExecutor(thread_pool_size, initializer=init_fn)

    bad_epoch_counter = 0
    best_solution = min(population)
    search_elite_fn = partial(search_elite, elite_search_size=elite_search_size)
    search_selected_fn = partial(search_selected, selected_search_size=selected_search_size)

    iterator = tqdm(range(max_iterations)) if verbose else range(max_iterations)
    for i in iterator:
        if verbose and i % print_every == 0:
            print(best_solution)

        population.sort()
        elite_locations, other_locations = population[:elite_population], population[elite_population:]

        selected_locations = selection(other_locations)

        elite_neighbourhood = p.map(search_elite_fn, elite_locations)

        selected_locations = p.map(search_selected_fn, selected_locations)

        random_locations = np.array([
            Location(solution, mutation)
            for solution in generate_random_solutions(
                n, size=max(0, population_size - elite_population - selected_population))
        ])

        population = np.concatenate([elite_locations, elite_neighbourhood, selected_locations, random_locations])
        population_best = min(population)
        if population_best.cost >= best_solution.cost:
            bad_epoch_counter += 1
            if bad_epoch_counter >= bad_epoch_patience:
                population = mutation(population)
                for location in population:
                    location.age = 0
                bad_epoch_counter = 0
        else:
            best_solution = population_best

    return best_solution
