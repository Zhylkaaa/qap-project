from QAP.utils import load_solution, load_example
from QAP.objective import objective
from QAP.solvers.bees import bees_solver
from QAP.solvers.selection_mechanisms import BestFit
from QAP.solvers.mutation_mechanisms import ShiftMutation, SwapMutation, UniformMutationScheduler
import numpy as np

if __name__ == '__main__':
    n, dists, costs = load_example('data/qapdata/kra30a.dat', dist_first=True)
    _, opt, permutation = load_solution('data/qapsoln/kra30a.sln')

    assert opt == objective(dists, costs,
                            permutation), "something wrong with objective"

    # TODO: hiperparameter tuning with WandB?
    mutation_mutations = (SwapMutation(mutation_prob=1), ShiftMutation())
    res = bees_solver(
        n,
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

    print('=======================================')
    print(f'result permutation: {res.permutation}')
    print(f'result solution: {res.cost}')
    print(f'optimal solution: {opt}')
