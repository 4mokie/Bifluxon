# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:35:05 2019

@author: 19173
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:19:33 2018

@author: kalas
"""


#import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt

from scipy import *
from scipy import optimize

from wavefunction1d import *
from matplotlib.widgets import Button

from multiprocessing import Pool

#from tqdm import tqdm, tqdm_notebook
from tqdm.autonotebook import tqdm

import matplotlib as mpl
from itertools import cycle
from scipy import signal


mpl.rcParams["font.size"] = 12

def find_nearest(val, array):
    index = np.argmin (abs( array - val) )
    return  index, array[index]

def g(Ej, Ecl):
    return 4*(2*Ej/Ecl)**0.5

def wp(Ej, Ecl):
    return (2*Ej*Ecl)**0.5


def t(g, wp):
    return 0.8*g**0.5*np.exp(-g)*wp


    

signs = np.array( [[0,1,2,3],
                   [0,1,2,3],
                   [1,0,3,2],
                   [1,0,3,2],
                   [1,0,3,2],
                   [1,2,0,3]])

    

def name_state(ng, fi_ext)  :
    return  'ng{:04d}_fi{:04d}'.format(int(5000+ng*1000), int(5000+fi_ext*1000) )
        
        
class State():
    
    pi = np.pi
    
    def __init__(self, ng, fi_ext ):
        self.ng = ng
        self.fi_ext = fi_ext

        self.E = None
        self.Psi = None
        
        self.state = name_state(ng, fi_ext) 

#    def calc_WF_1d(self, fi_grid = [-8*pi, 8*pi, 101], Q_grid = [-2, 2] ):

        
    


    def calc_WF(self ):

        
        if not( hasattr(self.qubit, 'fi_grid') or hasattr(self.qubit, 'Q_grid') ):
             self.qubit.fi_grid = [-8*pi, 8*pi, 101]
             self.qubit.Q_grid = [-2, 3]
             
        fi_grid = self.qubit.fi_grid
        Q_grid = self.qubit.Q_grid
        
        
        ng = self.ng 
        fi_ext = self.fi_ext 
        

        fi_min, fi_max, Nfi = fi_grid
        dfi = ( fi_max - fi_min ) /Nfi
        fi_list = np.linspace(fi_min, fi_max, Nfi)
        self.fi_list = fi_list
        self.Nfi = Nfi

        Q_min, Q_max = Q_grid
        Q_dim =  Q_max - Q_min + 1
        Q_list = np.arange(Q_min, Q_max + 1)   
        self.Q_dim = len(Q_list)
        self.Q_list = Q_list
        
# Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)
#            if asymEJ:
#        u = - 1./2 *Delta_E_J* np.sin(x/2)
#        V = np.kron(Q_p,assemble_V(u))

        
        x = fi_list + fi_ext

##      |n><n| matrix
        Q = np.diag(np.ones( Q_dim )).astype(np.complex)
        ##       |n+1><n| + |n><n+1| matrix           
        Q_ = np.diag( np.ones(Q_dim - 1) , 1) + np.diag( np.ones(Q_dim - 1), -1)

        dQ_ = -1j*np.diag( np.ones(Q_dim - 1) , 1) +1j* np.diag( np.ones(Q_dim - 1), -1) 

        
        ##      Kinetic energy of inductor only
        K_mtx = np.diag( -2* np.ones(Nfi) ) + np.diag( np.ones(Nfi - 1 ) , 1) + np.diag( np.ones(Nfi - 1), -1)
        ##      Kinetic energy of AC qubit = kroneker_product(inductor , cpb)
#        K = np.kron ( -4*self.E_CL * K_mtx /dfi**2 , Q)
        K = np.kron ( -4*self.qubit.E_CL * K_mtx /dfi**2 , Q)
    
    ##      Inductive energy 
#        v = lambda y: self.E_L/2* ( y )**2
        v =  self.qubit.V_L
 
        V = np.kron(np.diag( v(  fi_list, fi_ext ) ), Q)
        H = K + V

##      Josephson coupling               
#        u = lambda y: -self.E_J * np.cos(y/2)

        u = self.qubit.V_J1e
        du = self.qubit.dV_J1e

        U = np.kron( np.diag( u( fi_list ) ) , Q_)   
        H = H + U 
        
        dU = np.kron( np.diag( du( fi_list ) ) , dQ_)   
        H = H + dU 
 
        

##     Kinetic energy of CPB        
        H = H + 4*self.E_C * np.kron( np.diag(np.ones(Nfi)), np.diag( (Q_list - ng)**2))
        
        evals, evecs = solve_eigenproblem(H)
        
        
        E = evals.real
        
#        Psi = np.reshape(evecs.real, ( Nfi*Q_dim, Nfi* Q_dim) )        
        Psi = np.reshape(evecs, ( Nfi*Q_dim, Nfi* Q_dim) )        

        self.E = E
        self.Psi = Psi

        return E, Psi
    
    def mix_bands(self, band, signs):
        
        if self.ng == 0.5:
            ind = 1+ int(self.fi_ext // pi) 
#            print(ind)
            sign = signs[ind]
            out = np.argwhere(sign == band )[0]
            
        else:
            out = band
    
    
        return out
    
    def get_WF(self ):
        if self.E is None:

            E, Psi = self.calc_WF( )
        else:
            E = self.E
            Psi = self.Psi
        return E, Psi

    def get_E(self, band = slice(0,-1) ):

       E, Psi = self.get_WF(  )
       
#       band = self.mix_bands(band,signs)
       
       
       return  E[band]

    


    def get_Psi(self , band, q):
        
       E, Psi = self.get_WF(  )

#       band = self.mix_bands(band,signs)

       
       Psi_band = Psi[band]
       Psi_out = np.reshape(Psi_band, (( self.Nfi, self.Q_dim)))
       return Psi_out[:,q]




#       return  np.abs(Psi[band][q::self.Qdim ])
       

    
    
    def get_spectrum(self ):

        E, Psi = self.get_WF(  )
        self.E0n = (E - E[0])[1:]
        
        return  self.E0n  





    def get_fi_ij(self, i, j ): 
        
        fi_ij = 0
        self.get_WF(  )
        for q in range(self.Q_dim):

            Psi_i = self.get_Psi(i, q)
            Psi_j = self.get_Psi(j, q)


#            fi_ij += np.sum(Psi_i * Psi_j )
            fi_ij += np.sum(np.conjugate(Psi_i) * Psi_j * self.fi_list )
        
        return np.abs(fi_ij)
    
    def get_qp_ij(self, i, j ): 
        
        qp_ij = 0
        self.get_WF(  )
        for q in range(self.Q_dim):

            Psi_i = self.get_Psi(i, q)
            Psi_j = self.get_Psi(j, q)


#            fi_ij += np.sum(Psi_i * Psi_j )
            qp_ij += np.sum(np.conjugate(Psi_i) * Psi_j * np.sin(self.fi_list/2) )
        
        return np.abs(qp_ij)

    
    def get_n_ij(self, i, j, VERBOSE = False  ): 
        
        
        
        n_ij = 0
        self.get_WF(  )
        for k  in range(self.Q_dim):

            Psi_i = self.get_Psi(i, k)
            Psi_j = self.get_Psi(j, k)
            
            q = self.Q_list[k]
            
            el = np.sum ( np.conjugate(Psi_i) * Psi_j  )
            
            if VERBOSE:
                fig, ax = plt.subplots()
                
                ax.plot(Psi_i)
                ax.plot(Psi_j)
    
    
                ax.set_title( 'k = {}, elem = {:1.3e}'.format(q, el) )
    #            fi_ij += np.sum(Psi_i * Psi_j )
#            n_ij += el * (q - self.ng)
            n_ij += el * (q )
        
        return np.abs(n_ij)
    

    def get_psi_ij(self, i, j ): 
        
        psi_ij = 0
        for q in [0]: #range(self.Q_dim):

            Psi_i = self.get_Psi(i, q)
            Psi_j = self.get_Psi(j, q)


            psi_ij += np.sum(Psi_i * Psi_j)

        
        return psi_ij



    def get_chi_i(self, i, freq ): 
        
        chi_i = 0
        self.get_WF(  )
        for j in range(10):
            if i == j:
                continue
            
            E = self.get_E()   
            dE = E[i] - E[j] 
            fi_ij = self.get_fi_ij( i, j )
            chi_i += fi_ij**2 * 2*freq/( dE**2 - freq**2 ) 
            

        
        return freq-chi_i



    def get_T1_phi(self, fi_ext, ng, i = 0, j = 1):
 
        def Reff(w):
            A = 20e6 #/GHz
            return A/w


        kT = 1 #GHz
        Rq = 6e3
        
        
        E = self.get_E()
        w = 2*pi*(E[j] - E[i])
        
        fi_ij = self.get_fi_ij(i, j)
        
        return 1/(4*pi*Rq/Reff(w) * fi_ij**2*w*1e9/np.tanh(w/kT))    

 
    def get_T1_n(self, fi_ext, ng, i = 0, j = 1):
 


        kT = 1 #GHz
        Rq = 6e3
        
        
        E = self.get_E()
        w = 2*pi*(E[j] - E[i])
        
        n_ij = self.get_n_ij(i, j)
        
        return 1/( n_ij**2/w*1e9*self.E_C**2  *(1e-5)**2 )   

    
    def get_T1(self, fi_ext, ng, i = 0, j = 1):
 
       return 1/(1/self.get_T1_phi( fi_ext, ng, i = 0, j = 1) + 1/self.get_T1_n( fi_ext, ng, i = 0, j = 1))
