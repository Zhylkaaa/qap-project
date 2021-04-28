import numpy as np
from QAP.utils.solution_representation import SolutionRepresentation
from QAP.utils.solver_utils import generate_random_solutions
from copy import copy


class Location(SolutionRepresentation):
    lifetime: int = -1

    def __init__(self, permutation, mutation, calculate_cost=True):
        super(Location, self).__init__(permutation, calculate_cost)
        self.find_neighbors = mutation
        self.age = 0

    def increase_age(self):
        if Location.lifetime > 0:
            self.age += 1
            if self.age >= Location.lifetime:
                self.age = 0
                self.permutation = generate_random_solutions(self.permutation.shape[0], size=1)

    def search_neighbourhood_elite(self, neighbourhood_size):
        neighbourhood = self.find_neighbors([copy(self) for _ in range(neighbourhood_size)])
        best_neighbour = min(neighbourhood)

        self.increase_age()

        return best_neighbour

    def search_neighbourhood(self, neighbourhood_size):
        neighbourhood = self.find_neighbors([copy(self) for _ in range(neighbourhood_size)])
        best_neighbour = min(neighbourhood)
        if best_neighbour < self:
            self.age = 0
            self.permutation = best_neighbour.permutation
        else:
            self.increase_age()

    def __lt__(self, other):
        return self.cost < other.cost

    def __copy__(self):
        return Location(self.permutation.copy(), self.find_neighbors, calculate_cost=False)
