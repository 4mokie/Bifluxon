# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:01:53 2018

@author: Wenyuan Zhang @ Rutgers <wzhang@physics.rutgers.edu>
"""

import numpy as np
#from scipy import *
#from scipy import optimize
from wavefunction import *
from wavefunction.wavefunction1d import *
from wavefunction.utils import *
import pandas as pd


def U_ho(x, args):
    """
    Harmonic oscillator potential
    """

    k = args['k']
    x0    = args['x0']

    u = 1/2 * k * ((x-x0) ** 2)

    return u


def spectrum_vs_ng(ng_list,Q_list,phi0,params,args,max_E_level=4,asymEJ=True):
    E_J =params['EJ']
    E_C = params['EC']
    E_L = params['EL']
    E_CL = params['ECL']
    Delta_E_J =params['Delta_EJ']

    phi_min = args['Phi_min']
    phi_max = args['Phi_max']
    gridsize=args['gridsize'] # grid size along coordinate phi

    Q = np.diag(Q_list).astype(np.complex)

    Q_dim = np.size(Q_list)
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)

    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)


    # potential energy of superinductor
    x0=phi0
    x = np.linspace(phi_min,phi_max,gridsize+1)
    u = assemble_u_potential(U_ho, x, {'k': E_L, 'x0': x0})
    V = np.kron(np.diag(np.ones(Q_dim)),assemble_V(u))

    # kinetic energy of superinductor
    K = np.kron (np.diag(np.ones(Q_dim)),assemble_K(-4*E_CL,x))

    H = K+V

    # symmetric CPB
    u = -E_J * np.cos(x/2)
    V = np.kron(Q_, assemble_V(u))
    H = H + V

    # add asymmetry to EJ
    if asymEJ:
        u = - 1./2 *Delta_E_J* np.sin(x/2)
        V = np.kron(Q_p,assemble_V(u))
        H += V


    i=0
    energy = np.zeros((np.size(ng_list),max_E_level+1))

    for ng in ng_list:
        # charging energy of CPB
        H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(np.ones(gridsize+1)) )
        Hf = H+H0

        evals, evecs = solve_eigenproblem(Hf)
        evals = evals.real

        energy[i,0]=ng
        energy[i,1:]=evals[0:max_E_level]
        i+=1

    return energy


def spectrum_vs_phi0(phi0_list,Q_list,ng_list,params,args,max_E_level=4,asymEJ=True):
    E_J =params['EJ']
    E_C = params['EC']
    E_L = params['EL']
    E_CL = params['ECL']
    Delta_E_J =params['Delta_EJ']

    phi_min = args['Phi_min']
    phi_max = args['Phi_max']
    gridsize=args['gridsize'] # grid size along coordinate phi

    Q = np.diag(Q_list).astype(np.complex)

    Q_dim = np.size(Q_list)
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)

    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)


    i=0
    energy = np.zeros((np.size(ng_list)*np.size(phi0_list),max_E_level+1))
    for ng in ng_list:
        # charging energy of CPB
        H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(np.ones(gridsize+1)) )

        for x0 in phi0_list:
            # potential energy of superinductor
            x = np.linspace(phi_min,phi_max,gridsize+1)
            u = assemble_u_potential(U_ho, x, {'k': E_L, 'x0': x0})
            V = np.kron(np.diag(np.ones(Q_dim)),assemble_V(u))

            # kinetic energy of superinductor
            K = np.kron (np.diag(np.ones(Q_dim)),assemble_K(-4*E_CL,x))

            #
            H = K + V

            # symmetric CPB
            u = -E_J * np.cos(x/2)
            V = np.kron(Q_, assemble_V(u))
            H +=V

            # add asymmetry to EJ
            if asymEJ:
                u = - 1./2 *Delta_E_J* np.sin(x/2)
                V = np.kron(Q_p,assemble_V(u))
                H += V

            evals, evecs = solve_eigenproblem(H+H0)
            evals = evals.real

            energy[i,0]=x0
            energy[i,1:]=evals[0:max_E_level]
            i+=1

    return energy


def spectrum_(E_J, E_C, E_L, E_CL, ng, phi0,
              phi_min,phi_max,
              Q_dim=7, Q_offset=3,gridsize=100
              ):
    """
    phi0 : phase in radian
    """

    phi = np.linspace(phi_min,phi_max,gridsize+1) * 2* np.pi

    Q_list = np.arange(Q_dim)-Q_offset

    Q = np.diag(Q_list).astype(np.complex)

    Q_dim = np.size(Q_list)
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)

    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)


    # potential energy of superinductor
#    x = linspace(phi_min,phi_max,gridsize+1)
    u = assemble_u_potential(U_ho, phi, {'k': E_L, 'x0': phi0})
    V1 = np.kron(np.diag(np.ones(Q_dim)),assemble_V(u))

    # kinetic energy of superinductor
    K = np.kron (np.diag(np.ones(Q_dim)),assemble_K(-4*E_CL,phi))

    # symmetric CPB
    u = -E_J * np.cos(phi/2)
    V2 = np.kron(Q_, assemble_V(u))

    # charging energy of CPB
    H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(np.ones(phi.size)) )
    Hf =  K + V1 + H0 + V2

    evals, evecs = solve_eigenproblem(Hf)

#    t_array = np.zeros(evals.size+2+4).astype(np.float)
#    t_array[0:4] = [E_J,E_C,E_L,E_CL]
#    t_array[4:6] = np.array([ng,phi0])
#    t_array[6:]=evals.real
#
#    array.append(t_array)

    return evals, evecs

if  __name__=="__main__":

    E_J = 6.25
    E_C = 6.7
    E_L = 0.4
    E_CL = 5

#    E_L_list = np.array([0.2,1,5,15]) *1e-3 * 20
    E_L_list = np.array([0.02,0.05,0.2,1,5,15]) *1e-3 * 20
#    E_C_list = np.array([5, 10, 20])
    E_C_list = np.array([20])
#    EJ_EC_ratio = np.array([0.1,0.3,0.9,1.5])
    EJ_EC_ratio = np.linspace(0.1,1.5,10)

    ng_list = np.array([0.5])
    phi0_list = np.array([0,0.5])

    counter = 0
    print('Total {}'.format(E_L_list.size*E_C_list.size*EJ_EC_ratio.size *  phi0_list.size * ng_list.size))
    Q_dim, Q_offset, gridsize= [7,3,100]
    phi_min,phi_max = [-5,5]

    df = None
    array = []
    for E_C in E_C_list:
        for E_L in E_L_list:
            for E_J in E_C*EJ_EC_ratio:
#                for E_CL in E_CL_list:
                for ng in ng_list[:]:
                    for phi0 in phi0_list[:]:
                        E_CL= E_C
                        evals, evecs = spectrum_( E_J, E_C, E_L, E_CL, ng, phi0,
                                                 phi_min,phi_max,
                                                 Q_dim, Q_offset, gridsize
                                                 )
                        result = [E_J,E_C,E_L,E_CL,ng,phi0, Q_dim,Q_offset,gridsize]

                        max_i = 10
                        for En in evals[0:max_i] :
                            result.append(En.real)
                        for Psi in evecs[0:max_i,:]:
                            result.append(Psi.real)
                        array.append(result)

                        counter +=1
                        if np.mod(counter,10) ==0: print(counter)

        columns = ['EJ', 'EC','E_L','E_CL', 'ng','phi0','Q_dim','Q_offset','gridsize']
        [columns.append(str(x)) for x in np.arange(max_i)]
        [columns.append('Psi{}'.format(x)) for x in np.arange(max_i)]

        df = pd.DataFrame(array,columns=columns)
        fname = 'para_array_EC={}GHz.csv'.format(E_C)
    #    newdf= pd.DataFrame.from_csv(fname)
    #    ndf = newdf.append(df,ignore_index=True)
        df.to_csv(fname)







#if  __name__=="__main__":
#
#    fname = "energy_spectrum_12072018.dat"
#    data = []
#
##    E_L_list = np.array([1,5,15]) *1e-3 * 20
##    E_C_list = 30
##    E_J_EC_ratio = np.array([0.3,1.5])
##    ng_list = np.array([0.5])
##    phi0_list = np.array([0,0.5])
#
#    for E_J in []:
#        for E_C in []:
#            for E_L in []:
#                for E_CL in []:
#                    for ng in [] :
#
#    phi_min = args['Phi_min']
#    phi_max = args['Phi_max']
#    gridsize=args['gridsize'] # grid size along coordinate phi
#
#    Q = np.diag(Q_list).astype(np.complex)
#
#    Q_dim = np.size(Q_list)
#    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
#    for m in range(0,Q_dim):
#        for n in range(0,Q_dim):
#            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)
#
#    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
#    for m in range(0,Q_dim):
#        for n in range(0,Q_dim):
#            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)
#
#
#    # potential energy of superinductor
#    x = linspace(phi_min,phi_max,gridsize+1)
#    u = assemble_u_potential(U_ho, x, {'k': E_L, 'x0': phi0})
#    V = np.kron(np.diag(ones(Q_dim)),assemble_V(u))
#
#    # kinetic energy of superinductor
#    K = np.kron (np.diag(ones(Q_dim)),assemble_K(-4*E_CL,x))
#
#    # symmetric CPB
#    u = -E_J * np.cos(x/2)
#    V = np.kron(Q_, assemble_V(u))
#
#    # charging energy of CPB
#    H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(gridsize+1)) )
#    Hf =  K + V + H0
#
#    evals, evecs = solve_eigenproblem(Hf)
