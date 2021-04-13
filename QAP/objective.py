import numpy as np


def naive_objective(permutation: np.ndarray, dist: np.ndarray,
                    costs: np.ndarray) -> int:
    """Naive objective calculation.

    Args:
        permutation: permutation that represents current solution
        dist: matrix that represents distances between places
        costs: matrix that represents cost per unit of distance for facilities

    Returns:
        Objective value for given permutation
    """
    res = 0
    for i in range(permutation.shape[0]):
        for j in range(permutation.shape[0]):
            if i == j:
                continue

            res += costs[i, j] * dist[permutation[i], permutation[j]]
    return res


def objective(dist: np.ndarray, costs: np.ndarray,
              permutation: np.ndarray) -> int:
    """Vectorized objective calculation.

    Args:
        dist: matrix that represents distances between places
        costs: matrix that represents cost per unit of distance for facilities
        permutation: permutation that represents current solution

    Returns:
        Objective value for given permutation
    """
    P = np.zeros_like(dist)
    P[np.arange(dist.shape[0]), permutation] = 1  # create permutation matrix
    return np.sum(np.diagonal(costs @ P @ dist @ P.T))
