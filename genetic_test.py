from QAP.utils import load_solution, load_example
from QAP.objective import objective
from QAP.solvers.genetic import genetic_solver
import numpy as np

if __name__ == '__main__':
    n, dists, costs = load_example('data/qapdata/kra30a.dat', dist_first=True)
    _, opt, permutation = load_solution('data/qapsoln/kra30a.sln')

    assert opt == objective(dists, costs,
                            permutation), "something wrong with objective"

    # TODO: hiperparameter tuning with WandB?
    res = genetic_solver(n,
                         dists,
                         costs,
                         objective,
                         max_iterations=1000,
                         population_size=1000,
                         verbose=True,
                         print_every=100,
                         selection_size=1000,
                         crossover_count=500,
                         mutation_prob=0.5)

    print('=======================================')
    print(f'result permutation: {res.permutation}')
    print(f'result solution: {res.cost}')
    print(f'optimal solution: {opt}')
