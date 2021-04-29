import numpy as np
import os
import warnings
from typing import Tuple, List


def _parse_matrix(n: int,
                  rows: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Parses through rows accounting for different data formats in QAPLib.

    Args:
        n: expected size of matrix
        rows: rows containing data from both matrices

    Returns:
        resulting ndarray and rest of rows for next matrix in the file
    """
    result = []
    row = []
    row_len = 0

    for idx, r in enumerate(rows):
        row_len += len(r.strip().split())
        row.append(r.strip())
        if row_len == n:
            result.append(list(map(int, ('\t'.join(row)).split())))
            row = []
            row_len = 0
            if len(result) == n:
                return np.array(result), rows[idx+1:]
        if row_len > n:
            raise ValueError('something is wrong with data formatting')

    return np.array(result), []


def load_example(file_path: str,
                 check_prefix: bool = True,
                 dist_first: bool = True) -> Tuple[int, np.ndarray, np.ndarray]:
    """Reads specified .dat file from QAPLib and return problem size with D and C matrices for this problem.
    Args:
        file_path: path to .dat file from QAPLib or similarly structured file. See: https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/#KP # noqa
        check_prefix: (optional) argument specifies if .dat is required extension.
        dist_first: (optional) weather distance matrix goes first in the .dat file
    Returns:
        tuple containing size of a problem, dists matrix and costs matrix.
    """
    assert os.path.exists(file_path), "file does not exists"

    if not file_path.endswith('.dat'):
        if check_prefix:
            raise ValueError(
                "File extension should be .dat. "
                "Set check_prefix parameter to False if you know what you are doing"
            )
        else:
            warnings.warn(
                "be sure to provide right file it usually have .dat extension")

    with open(file_path) as f:
        rows = [r for r in f.readlines() if not r.isspace()]
        n, rows = int(rows[0]), rows[1:]
        dists, rows = _parse_matrix(n, rows)
        costs, _ = _parse_matrix(n, rows)

    if dist_first:  # a bit ugly solution, but they could somehow unify the format, grrrr
        return n, dists, costs
    else:
        return n, costs, dists


def load_solution(file_path: str,
                  check_prefix: bool = True) -> Tuple[int, int, np.ndarray]:
    """Reads specified file from QAPLib and return ndarray for this problem.
        Args:
            file_path: path to .sln file from QAPLib or similarly structured file. See: https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/#KP # noqa
            check_prefix: (optional) argument specifies if .sln is required extension.
        Returns:
            tuple containing size of a problem, optimal solution and permutation
        """
    assert os.path.exists(file_path), "file does not exists"

    if not file_path.endswith('.sln'):
        if check_prefix:
            raise ValueError(
                "File extension should be .sln. "
                "Set check_prefix parameter to False if you know what you are doing"
            )
        else:
            warnings.warn(
                "be sure to provide right file it usually have .sln extension")

    with open(file_path) as f:
        rows = [r for r in f.readlines() if not r.isspace()]

        n, opt = map(int, rows[0].split())
        permutation = np.array(list(map(int, '\t'.join(rows[1:]).split())))
        permutation -= 1 if permutation.max(
        ) == n else 0  # to account for 0 indexed arrays

    return n, opt, permutation
