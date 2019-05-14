# Construct a matrix which represent the schrodinger equation on matrix form
# 
# Initially created by J Robert Johansson, <robert@riken.jp>
# Modified by Wenyuan Zhang, <wzhang@physics.rutgers.edu>
# 

import numpy as np

#
# kronecker delta, optionally modify so that it also take the boundary
# conditions into account?
#  
def mod_kron(n, m):
    return (n == m)


def assemble_K(k,x):
    """
    Assemble the matrix representation of the kinetic energy contribution
    to the Hamiltonian.

    k = -hbar**2 / 2 m 
    """    
    N = np.size(x)
    K = np.zeros((N,N)).astype(np.complex)
    dx = x[1]-x[0]
  
    for m in range(0, N):
        for n in range(0,N):
            K[m, n] = k / (dx ** 2) * (mod_kron(m + 1, n) - 2 * mod_kron(m, n) + mod_kron(m - 1, n))    
                    
    return K


def assemble_V(u):
    """
    Assemble the matrix representation of the potential energy contribution
    to the Hamiltonian.
    """
    N = np.size(u)
    V = np.zeros((N,N)).astype(np.complex)
  
    for m in range(N):
        for n in range(N):
            V[m, n] = u[m] * mod_kron(m, n)
            
    return V


def basis_step_evalute(N, u, x):
    """

    """    
    return u


def assemble_u_potential(u_func, x, args):
    """

    """
    return u_func(x, args)


def wavefunction_norm(x, psi):
    """
    Calculate the norm of the given wavefunction.
    """    

    dx = x[1] - x[0]

    return (psi.conj() * psi).sum() * dx


def wavefunction_normalize(x, psi):
    """
    Normalize the given wavefunction.
    """    
    
    return psi / np.sqrt(wavefunction_norm(x, psi))


def expectation_value(x, operator, psi):
    """
    Evaluate the expectation value of a given operator and wavefunction.
    """    

    dx = x[1] - x[0]
    
    return (psi.conj() * operator * psi).sum() * dx


def inner_product(x, psi1, psi2):
    """
    Evaluate the inner product of two wavefunctions, psi1 and psi2, on a space
    described by X1 and X2.
    """    

    dx = x[1] - x[0]
    
    return (psi1.conj() * psi2).sum() * dx


def derivative(x, psi):
    """
    Evaluate the expectation value of a given operator and wavefunction.
    """    

    dx = x[1] - x[0]

    N = len(psi)
    
    dpsi = np.zeros(N, dtype=np.complex)

    def _idx_wrap(M, m):
        return m if m < M else m - M

    for n in range(N):
        dpsi[n] = (psi[_idx_wrap(N, n+1)] - psi[n-1]) / (2 * dx)

    return dpsi

