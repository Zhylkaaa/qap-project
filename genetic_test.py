from QAP.utils import load_solution, load_example
from QAP.objective import objective
from QAP.solvers.genetic import genetic_solver
import numpy as np

if __name__ == '__main__':
    n, dists, costs = load_example('data/qapdata/lipa20a.dat', dist_first=False)
    _, opt, permutation = load_solution('data/qapsoln/lipa20a.sln')

    # TODO: hiperparameter tuning with WandB?
    res = genetic_solver(n,
                         dists,
                         costs,
                         objective,
                         max_iterations=1000,
                         population_size=1000,
                         selection_size=1000,
                         crossover_count=500,
                         mutation_prob=0.5)
    print(res.permutation, res.cost)
    print(f'optimal solution: {opt}')
