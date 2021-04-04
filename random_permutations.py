from QAP.utils import load_solution, load_example
from QAP.objective import objective
import numpy as np

if __name__ == '__main__':

    n, dists, costs = load_example('data/qapdata/kra30a.dat')
    _, opt, permutation = load_solution('data/qapsoln/kra30a.sln')

    assert opt == objective(permutation, dists, costs)

    np.random.seed(1234)
    diffs = []
    for _ in range(1000):
        diffs.append(
            objective(np.random.permutation(n), dists, costs) - opt
        )

    print(f'mean difference for random permutations is {np.mean(diffs)}, min {np.min(diffs)}, max {np.max(diffs)}\n'
          f'optimal is {opt}')
