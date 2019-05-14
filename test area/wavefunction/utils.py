"""
Utility functions for wavefunction calculations.

J Robert Johansson, <robert@riken.jp>
"""

import numpy as np

def print_matrix(A):
    """
    Print real part of matrix matrix to stdout
    """
    print('\n'.join([' '.join(['{:.3}'.format(item.real) for item in row]) 
      for row in A]))

def solve_eigenproblem(H):
    """
    Solve an eigenproblem and return the eigenvalues and eigenvectors.
    """
    vals, vecs = np.linalg.eig(H)
    idx = np.real(vals).argsort()
    vals = vals[idx]
    vecs = vecs.T[idx]

    return vals, vecs
