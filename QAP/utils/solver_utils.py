import numpy as np


def generate_random_solutions(n: int, size: int = 100) -> np.ndarray:
    """Generates random permutation that can be used as initial solutions.
    Args:
        n: size of a problem
        size: (optional) number of permutations
    Returns:
        ndarray permutation matrix of shape (size, n)
    """
    return np.stack([np.random.permutation(n) for _ in range(size)], axis=0)
