import numpy as np


def generate_random_solutions(n: int, size: int = 100) -> np.ndarray:
    """Generates random permutation that can be used as initial solutions.
    Args:
        n: size of a problem
        size: (optional) number of permutations
    Returns:
        ndarray permutations matrix of shape (size, n)
    """
    return np.stack([np.random.permutation(n) for _ in range(size)], axis=0) if size > 0 else np.array([], dtype=np.object_)


# TODO: check if naive implementation is faster
def get_liveness_score(costs: np.ndarray) -> np.ndarray:
    """Get list of chromosome liveness score (probability of surviving and breading).
    Args:
        costs: vector of solution costs
    Returns:
        liveness scores for each solution in current population
    """
    n = costs.shape[0]
    liveness = np.arange(1, n + 1) / (((1 + n) * n) // 2)
    idxs = np.argsort(-costs)  # there is no reverse to I should improvise :)
    P = np.zeros((n, n))
    P[np.arange(n), idxs] = 1  # create permutation matrix
    return liveness @ P
