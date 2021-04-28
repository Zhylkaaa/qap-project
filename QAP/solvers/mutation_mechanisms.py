import numpy as np
from typing import List, Union, Iterable, Type
from QAP.utils.solution_representation import SolutionRepresentation


class MutationMechanism:
    """Abstract class for mutation mechanism.
    Each descendant must implement `single_mutation` function that gets called from `mutate`
    or overwrite `mutate` function to achive different potentially better performing behaviour
    """
    def __init__(self, *args, **kwargs):
        pass

    def single_mutation(self, representation):
        raise NotImplementedError(
            'this is abstract class method and should be implemented by descendants'
        )

    def mutate(self,
               population: Union[np.ndarray, Type[SolutionRepresentation]]) -> np.ndarray:
        """Perform mutation defined in single_mutation on population or single representation.
        Args:
            population: list of representations that might be mutated
        Returns:
            mutated population.
        """
        if isinstance(population, SolutionRepresentation):
            return self.single_mutation(population)

        for representation in population:
            self.single_mutation(representation)

        return population

    def __call__(self, *args, **kwargs):
        return self.mutate(*args, **kwargs)


class SwapMutation(MutationMechanism):
    """Random swap mutation.
    Swaps 2 uniformly selected genes in chromosome with certain probability

    Args:
         mutation_prob: (optional) probability of gen swap
    """

    def __init__(self, mutation_prob: float = 0.5):
        self.mutation_prob = mutation_prob

    def single_mutation(self, representation):
        """With `mutation_prob` swap 2 random elements with each other."""
        if np.random.sample() <= self.mutation_prob:
            i, j = np.random.choice(representation.permutation.shape[0],
                                    size=2,
                                    replace=False)
            representation.permutation[[i, j]] = representation.permutation[[j, i]]
        return representation.calculate_cost()


class ShiftMutation(MutationMechanism):
    def single_mutation(self, representation):
        i, j = np.random.choice(representation.permutation.shape[0], size=(2,), replace=False)
        if i < j:  # shift right
            tmp = representation.permutation[i]
            representation.permutation[i:j] = representation.permutation[i+1:j+1]
            representation.permutation[j] = tmp
        else:
            tmp = representation.permutation[i]
            representation.permutation[j+1:i+1] = representation.permutation[j:i]
            representation.permutation[j] = tmp
        return representation.calculate_cost()


class UniformMutationScheduler(MutationMechanism):
    def __init__(self, mutation_mutations: Iterable[Type[MutationMechanism]] = (SwapMutation(), ShiftMutation())):
        self.mutations = mutation_mutations

    def single_mutation(self, representation):
        mutation = np.random.choice(self.mutations)
        return mutation(representation)
