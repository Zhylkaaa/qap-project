# TODO: implement bees algorithm to solve QAP
# papers:
# * https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note
# * https://link.springer.com/chapter/10.1007/978-3-319-23437-3_53
import numpy as np
from typing import Callable
from functools import partial
import numpy as np
import re
from QAP.objective import objective
from tqdm import tqdm
from ..selection_mechanisms import SelectionMechanism, BestFit
from ..mutation_mechanisms import MutationMechanism, UniformMutationScheduler, SwapMutation


def bees_solver(n: int,
                dist: np.ndarray,
                cost: np.ndarray,
                objective: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                    int] = objective,
                max_iterations: int = 100,
                population_size: int = 100,
                verbose: bool = True,
                print_every: int = 100,
                mutation_mechanism: MutationMechanism = UniformMutationScheduler,
                selection_mechanism: SelectionMechanism = BestFit,
                solution_lifetime: int = 20,
                **kwargs) -> np.ndarray:
    pass
