from QAP.utils import load_solution, load_example
from QAP.objective import objective
from QAP.solvers.genetic import genetic_solver
from QAP.solvers.selection_mechanisms import BestFit
from QAP.solvers.mutation_mechanisms import ShiftMutation, SwapMutation, UniformMutationScheduler
import numpy as np

if __name__ == '__main__':
    n, dists, costs = load_example('data/qapdata/kra30a.dat', dist_first=True)
    _, opt, permutation = load_solution('data/qapsoln/kra30a.sln')

    assert opt == objective(dists, costs,
                            permutation), "something wrong with objective"

    # TODO: hiperparameter tuning with WandB?
    mutation_mutations = (SwapMutation(mutation_prob=0.3), ShiftMutation())
    res = genetic_solver(
        n,
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

    print('=======================================')
    print(f'result permutation: {res.permutation}')
    print(f'result solution: {res.cost}')
    print(f'optimal solution: {opt}')
    print(f'relative error: {(res.cost - opt) / opt}')
