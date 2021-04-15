import numpy as np
from QAP.utils.solver_utils import get_liveness_score


class SelectionMechanism:
    """Abstract class for selection mechanism.
    Each descendant must implement `select` function that gets called with `()` syntax
    """
    def select(self, generation: np.ndarray):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def __call__(self, *args, **kwargs):
        return self.select(*args, **kwargs)


class RouletteWheel(SelectionMechanism):
    """Randomly select sub-population from general population.
    Each sample is weighted by it's position in sorted
    by cost array (higher the cost, lover the probability of surviving)
    """
    def __init__(self, selection_size: int = 100):
        self.selection_size = selection_size

    def select(self,
               population: np.ndarray) -> np.ndarray:
        costs = np.array([p.cost for p in population])
        probs = get_liveness_score(costs)
        return np.random.choice(population,
                                replace=False,
                                size=self.selection_size,
                                p=probs)
