from QAP.utils import load_solution, load_example
from QAP.objective import objective
from QAP.solvers.genetic import genetic_solver
import numpy as np

if __name__ == '__main__':
    n, dists, costs = load_example('data/qapdata/kra30a.dat')
    _, opt, permutation = load_solution('data/qapsoln/kra30a.sln')

    res = genetic_solver(n,
                         dists,
                         costs,
                         objective,
                         max_iterations=1000,
                         population_size=1000,
                         selection_size=1000,
                         crossover_count=400)
    print(res.permutation, res.cost)
    print(f'optimal solution: {opt}')
