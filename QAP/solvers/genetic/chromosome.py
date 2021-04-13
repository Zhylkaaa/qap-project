import numpy as np


class Chromosome:
    objective = None

    def __init__(self, permutation: np.ndarray):
        self.permutation = permutation
        self.cost = Chromosome.objective(permutation)

    def __ge__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return self.cost == other.cost
