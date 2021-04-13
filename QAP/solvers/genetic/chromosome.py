import numpy as np


class Chromosome:
    objective = None

    def __init__(self, permutation: np.ndarray, calculate_cost=True):
        self.permutation = permutation
        self.cost = Chromosome.objective(permutation) if calculate_cost else -1

    def calculate_cost(self):
        self.cost = Chromosome.objective(self.permutation)
        return self

    def __gt__(self, other):
        return self.cost < other.cost

    def __lt__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __hash__(self):
        return hash(tuple(self.permutation))
