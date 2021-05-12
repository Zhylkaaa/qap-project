from typing import Tuple

import numpy as np
import time
from QAP.objective import objective
from QAP.solvers.bees import bees_solver
from QAP.solvers.genetic import genetic_solver
from QAP.solvers.mutation_mechanisms import SwapMutation, ShiftMutation, UniformMutationScheduler
from QAP.utils import load_solution, load_example


def test_genetic(size: int, dists: np.ndarray, costs: np.ndarray, reruns_number: int = 3) -> Tuple[float, float]:
    mutation_mutations = (SwapMutation(mutation_prob=0.3), ShiftMutation())
    results = []
    times = []
    for _ in range(reruns_number):
        start = time.time()
        res = genetic_solver(
            size,
            dists,
            costs,
            objective,
            max_iterations=1000,
            population_size=100,
            verbose=True,
            print_every=100,
            selection_size=100,
            crossover_count=50,
            mutation_mechanism=UniformMutationScheduler,
            # mutation_prob=0.1,
            mutation_mutations=mutation_mutations,
        )
        end = time.time()
        results.append(res.cost)
        times.append(end - start)

    return sum(results) / len(results), sum(times) / len(times)


def test_bees(size: int, dists: np.ndarray, costs: np.ndarray, reruns_number: int = 3) -> Tuple[float, float]:
    mutation_mutations = (SwapMutation(mutation_prob=1), ShiftMutation())
    results = []
    times = []

    for _ in range(reruns_number):
        start = time.time()
        res = bees_solver(
            size,
            dists,
            costs,
            objective,
            max_iterations=1000,
            population_size=500,
            verbose=True,
            print_every=100,
            elite_population=20,
            selected_population=300,
            elite_search_size=20,
            selected_search_size=10,
            solution_lifetime=20,
            bad_epoch_patience=40,
            thread_pool_size=12,
            mutation_mechanism=UniformMutationScheduler,
            mutation_mutations=mutation_mutations,
        )
        end = time.time()
        results.append(res.cost)
        times.append(end - start)

    return sum(results) / len(results), sum(times) / len(times)


if __name__ == "__main__":
    data_folder = 'data/'
    problems_folder = data_folder + 'qapdata/'
    solutions_folder = data_folder + 'qapsoln/'
    results_file = 'results.csv'
    reruns_number = 3

    problems = [
        'lipa20b.dat',
        'lipa30b.dat',
        'lipa40b.dat',
        'lipa50b.dat',
        'lipa60b.dat',
        'lipa70b.dat',
        'lipa80b.dat',
        'lipa90b.dat',
    ]

    solutions = [
        'lipa20b.sln',
        'lipa30b.sln',
        'lipa40b.sln',
        'lipa50b.sln',
        'lipa60b.sln',
        'lipa70b.sln',
        'lipa80b.sln',
        'lipa90b.sln',
    ]

    results = []

    for problem, solution in zip(problems, solutions):
        size, dists, costs = load_example(problems_folder + problem, dist_first=True)
        _, opt, permutation = load_solution(solutions_folder + solution)

        bees_result, bees_time = test_bees(size, dists, costs, reruns_number)
        genetic_result, genetic_time = test_genetic(size, dists, costs, reruns_number)

        results.append((problem, size, opt, bees_result, bees_time, genetic_result, genetic_time))

    with open(data_folder + results_file, 'w') as file:
        file.write("problem_name,size,optimal_solution,bees_result,bees_time,genetic_result,genetic_time\n")
        for line in results:
            file.write(",".join(map(lambda x: str(x), line)) + '\n')

