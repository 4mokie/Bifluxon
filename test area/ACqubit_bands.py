# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:19:33 2018

@author: kalas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 00:07:26 2018
@author: Wenyuan Zhang wzhang@physics.rutgers.edu
This program is to calculate energy spectrum for Aharonov-Casher Qubit. 
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



if __name__=='__main__':
    pi = np.pi
    
    E_CL = 5
    E_L = 0.4
    E_J =6.25
    E_C = 6.7

    
    ##                                  grid for fi variable
    fi_min, fi_max = [-10*pi, 10*pi]
    Nfi = 101
    dfi = ( fi_max - fi_min ) /Nfi
    fi_list = np.linspace(fi_min, fi_max, Nfi)


    ##                                  grid for charge variable   

    NQ = 2
    Q_dim = 2*NQ + 1
    Q_list = np.arange(-NQ, NQ +1)   


    ##                                  grid for external flux    
    Nfi_ext = 101
    fi_ext_min, fi_ext_max = [-2*pi, 2*pi]
    fi_ext_list = np.linspace(fi_ext_min, fi_ext_max, Nfi_ext)

    ##          band number to trace     
    Band = 1


    ##             do new calculation or use E and Psi from the last calculation
    new_calc = False
#    new_calc = True


    ##             plot bands coresponded to certan flux number   
    true_bands = True
#    true_band = False
    
    if new_calc:

        E = np.zeros( (Nfi_ext, Nfi*Q_dim))
        Psi = np.zeros( (Nfi_ext, Nfi*Q_dim, Nfi*Q_dim) )
       
        
        for ng in [-0.5]:
            for i, x0 in enumerate(fi_ext_list) :
                
                x = fi_list + x0

    ##             |n><n| matrix
                Q = np.diag(np.ones( Q_dim )).astype(np.complex)
    ##             |n+1><n| + |n><n+1| matrix           
                Q_ = np.diag( np.ones(Q_dim - 1) , 1) + np.diag( np.ones(Q_dim - 1), -1)
                
    ##             Kinetic energy of inductor only
                K_mtx = np.diag( -2* np.ones(Nfi) ) + np.diag( np.ones(Nfi - 1 ) , 1) + np.diag( np.ones(Nfi - 1), -1)
    ##             Kinetic energy of AC qubit = kroneker_product(inductor , cpb)
                K = np.kron ( 4*E_CL * K_mtx /dfi**2 , Q)
            
    ##             Inductive energy 
                v = lambda y: E_L/2* ( y )**2
                V = np.kron(np.diag( v(  x - x0 ) ), Q)
                H = K + V

    ##             Josephson coupling               
                u = lambda y: -E_J * np.cos(y/2)
                U = np.kron( np.diag( u( x ) ) , Q_)
                H = H + U 

    ##             Kinetic energy of CPB        
                H = H + 4*E_C * np.kron( np.diag(ones(Nfi)), np.diag( (Q_list - ng)**2))
                
                evals, evecs = solve_eigenproblem(H)
                
                E[i] = evals
                Psi[i] = evecs
                


if true_bands:
    E_bands = np.copy(E)
    Psi_bands = np.copy(Psi)
    for band in np.arange(Nfi*Q_dim/2 - 1):
        b = int(band)
    
    
        E_bands[int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b] =  E[ int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b+1]
        Psi_bands[int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b,:] =  Psi[ int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b+1,:]
    
        E_bands[int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b+1] =  E[ int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b]
        Psi_bands[int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b+1,:] =  Psi[ int(Nfi_ext/4):int(3*Nfi_ext/4), 2*b,:]
    
    
    E_plot = E_bands
    Psi_plot = Psi_bands
else:
      E_plot =  np.copy(E)
      Psi_plot =  np.copy(Psi)








l_list = []
axes = []
gs= GridSpec( Q_dim, 2 )

fig=plt.figure(figsize = (15,10))

Z = np.reshape(Psi_plot[0][Band][:], (( Nfi, Q_dim)))

for i in np.arange(Q_dim):          ## plot set of wavefunctions for diggerent charge number
    
    ax = fig.add_subplot(gs[i,0])
    axes.append(ax)
    
    l, = ax.plot(fi_list /2/pi, ( Z[:, i]))
    ax.xaxis.set_ticks(np.arange(-5, 6))
    ax.grid()
    l_list.append(l)
    
    ax.set_ylim ( (1.1*np.min(Psi_plot[:][1][:]), 1.1*np.max(Psi_plot[:][1][:]) ))
    ax.set_ylabel ('N = %1.0d'% Q_list[i])

ax.set_xlabel ('Var flux $\\phi$')




ax_spec = fig.add_subplot(gs[:,1])                      ## plot bands
ax_spec.set_xlabel ('External flux $\\Phi_{ext}$')

bands_to_plot = np.arange(5)
for n in bands_to_plot:
    ax_spec.plot (fi_ext_list /2/pi ,E_plot[:, n])



cross, = ax_spec.plot (fi_ext_list[0] /2/pi , E[0][Band], 'X', ms = 10) ## plot cursor
anim_running = True

def animate(j):         ## sweep the external flux
    print(j)
    
    Z = np.reshape(Psi_plot[j][Band][:], (( Nfi, Q_dim)))
    for i, l in enumerate(l_list):
      l.set_data(fi_list/2/pi, (Z[:, i]) )  

    cross.set_data(fi_ext_list[j] /2/pi , E_plot[ j , Band] )
     
    return  [l_list,cross]

def on_button_clicked(event):        ## pause if clicked
    global anim_running
    
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else :
        anim.event_source.start()
        anim_running = True
    print ("button clicked")

axnext = plt.axes([0.81, 0.05, 0.1, 0.025])
bpause = Button(axnext, 'Pause')
bpause.on_clicked(on_button_clicked)



anim = animation.FuncAnimation(fig, animate,
                               frames=Nfi_ext, interval= 50*500/Nfi_ext,   blit=False, repeat = True)
plt.show()

