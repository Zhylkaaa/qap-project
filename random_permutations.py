from QAP.utils import load_solution, load_example
from QAP.objective import objective, naive_objective
import numpy as np

if __name__ == '__main__':

    n, dists, costs = load_example('data/qapdata/lipa20a.dat', dist_first=False)
    _, opt, permutation = load_solution('data/qapsoln/lipa20a.sln')

    assert opt == naive_objective(dists, costs,
                                  permutation), "something wrong with objective"

    np.random.seed(1234)
    diffs = []
    for _ in range(10000):
        diffs.append(objective(dists, costs, np.random.permutation(n)) - opt)

    print(
        f'mean difference for random permutations is {np.mean(diffs)}, min {np.min(diffs)}, max {np.max(diffs)}\n'
        f'optimal is {opt}')
