# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:37:36 2019

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

from tqdm import tqdm, tqdm_notebook

import matplotlib as mpl
from itertools import cycle
from scipy import signal

from ACqubit import *
from ACstate import *


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



def plot_Edps_EJ_ECL(EJ_list,  E_CL_list, E_L = 0.4):
    
#    E_L = 0.4

    fig, ax = plt.subplots()  
    
    fig.suptitle('$E_{L}$ = %1.2f'%E_L)

    ax.set_xlim((0,30))
    ax.set_ylim((0.001,10))
    ax.set_yscale('log')
    
    ax.set_xlabel('$E_{J}$, GHz')
    ax.set_ylabel('$E_{dps}$, GHz')



    
    for E_CL in E_CL_list:

        y_plot = []
        x_plot = []
    
        points, = ax.plot(x_plot, y_plot, label = "%1.2f"% E_CL)
    
        plt.ion()
       
        ax.legend(title="$E_{CL}$, GHz")


        for E_J in EJ_list :
            
            qubit =  ACQubit(E_CL = E_CL,
                E_L = E_L,
                E_J = E_J,
                E_C = 6.7/1)
            


            E, Psi = qubit.get_bands(fi_ext = 0, ng = 0.5)
#            print(E[:3])
            Edps = E[2] - E[1]
            
            y_plot.append(Edps)
            x_plot.append(E_J)
            
            points.set_data(x_plot, y_plot)
            fig.canvas.draw()
            plt.pause(0.01)   
            
    return print('done')    


def plot_Edps_g_wp(g_list,  wp_list, E_L = 0.4):
    
#    E_L = 0.4

    fig, ax = plt.subplots()  
    
    fig.suptitle('$E_{L}$ = %1.2f'%E_L)

    ax.set_xlim((np.min(g_list) **2/32, np.max(g_list)**2/32  ))
    ax.set_ylim((1,20))
    ax.set_yscale('log')
    
    ax.set_xlabel('$E_J/E_{CL}$')
    ax.set_ylabel('$E_{dps}$, GHz')



    
    for wp in wp_list:

        y_plot = []
        x_plot = []
    
        points, = ax.plot(x_plot, y_plot, label = "%1.2f"% wp)
    
        plt.ion()
       
        ax.legend(title="$\\omega_{p}$, GHz")


        for g in g_list :
            E_J = g*wp/8
            E_CL = 4*wp/g
            
            print(E_J, E_CL)
            
            qubit =  ACQubit(E_CL = E_CL,
                E_L = E_L,
                E_J = E_J,
                E_C = 6.7/1)
            


            E, Psi = qubit.get_bands(fi_ext = 0, ng = 0.5)
#            print(E[:3])
            Edps = E[2] - E[1]
            
            y_plot.append(Edps)
            x_plot.append(g**2/32)
            
            points.set_data(x_plot, y_plot)
            fig.canvas.draw()
            plt.pause(0.01)   
            
    return print('done')  


def plot_fiij_g_wp(g_list,  wp_list, E_L = 0.4):
    
#    E_L = 0.4

    fig, ax = plt.subplots()  
    
    fig.suptitle('$E_{L}$ = %1.2f'%E_L)

    ax.set_xlim((np.min(g_list) **2/32, np.max(g_list)**2/32  ))
    ax.set_ylim((1,20))
    ax.set_yscale('log')
    
    ax.set_xlabel('$E_J/E_{CL}$')
    ax.set_ylabel('$E_{dps}$, GHz')



    
    for wp in wp_list:

        y_plot = []
        x_plot = []
    
        points, = ax.plot(x_plot, y_plot, label = "%1.2f"% wp)
    
        plt.ion()
       
        ax.legend(title="$\\omega_{p}$, GHz")


        for g in g_list :
            E_J = g*wp/8
            E_CL = 4*wp/g
            
            print(E_J, E_CL)
            
            qubit =  ACQubit(E_CL = E_CL,
                E_L = E_L,
                E_J = E_J,
                E_C = 6.7/1,
                dE_J = 0)
            

            st = qubit.set_state(ng = 0.45, fi_ext = 0.1 )
            fiij = st.get_fi_ij(0,1)
#            print(E[:3])
#            Edps = E[2] - E[1]
            
            y_plot.append(fiij)
#            x_plot.append(g**2/32)
            x_plot.append(g)
            
            points.set_data(x_plot, y_plot)
            fig.canvas.draw()
            plt.pause(0.01)   
            
    return print('done')  

def plot_Edps_EL_wp(EL_list,  wp_list, g = 4):
    
#    E_L = 0.4

    fig, ax = plt.subplots()  
    
    fig.suptitle('$E_{J}/E_{CL}$ = %1.2f'% (g**2/32) )

    ax.set_xlim((np.min(EL_list) , np.max(EL_list)  ))
    ax.set_ylim((0.01,100))
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_xlabel('$E_{L}$')
    ax.set_ylabel('$E_{dps}$, GHz')

    limit, = ax.plot(EL_list, EL_list*2*pi**2,'g--', label = "$E_{dps} = 2\\pi^2 E_L$")  
    first_legend = ax.legend(handles=[limit], loc=1)
    
    for wp in wp_list:

        y_plot = []
        x_plot = []
    

        points, = ax.plot(x_plot, y_plot, label = "%1.2f"% wp)
#        ax.legend(handles=[points], loc='upper left', )
#        l2 = plt.gca().add_artist( second_legend )

        plt.ion()
       
        ax.legend(title="$\\omega_{p}$, GHz")


        for E_L in EL_list :
            E_J = g*wp/8
            E_CL = 4*wp/g
            
            print(E_J, E_CL)
            
            qubit =  ACQubit(E_CL = E_CL,
                E_L = E_L,
                E_J = E_J,
                E_C = 6.7/1)
            


            E, Psi = qubit.get_bands(fi_ext = 0, ng = 0.5)
#            print(E[:3])
            Edps = E[2] - E[1]
            
            y_plot.append(Edps)
            x_plot.append(E_L)
            
            points.set_data(x_plot, y_plot)
            fig.canvas.draw()
            plt.pause(0.01)   
            
        return print('done')    
       


if __name__=='__main__':
    pi = np.pi


    Nfi_ext = 101
    fi_ext_min, fi_ext_max = [0.0*pi, 1*pi]
    fi_ext_list = np.linspace(fi_ext_min, fi_ext_max, Nfi_ext)
    dfi_ext = ( fi_ext_max - fi_ext_min ) /Nfi_ext

    
#    ACQB15 = ACQubit (E_CL = 8, 
#                      E_L = 0.6, 
#                      E_J = 35, 
#                      E_C = 16,
#                      dE_J = 0)

#    ACQB15 = ACQubit (E_CL = 8, 
#                      E_L = 0.6, 
#                      E_J = 35, 
#                      E_C = 13,
#                      dE_J = 10)
    
    
    J1 = 33
    J2 = 1*J1+13
    
    ACQB15 = ACQubit (E_CL = 7, 
                  E_L = 0.5, 
                  E_J = (J1+J2)/2, 
                  E_C = 15,
                  dE_J = (J2-J1)/2 )
    
#    ACQB15 = ACQubit (E_CL = 8, 
#              E_L = 0.6, 
#              E_J = 30, 
#              E_C = 16,
#              dE_J = 0)
    
    ACQB15.set_grid(fi_grid = [-8*pi, 8*pi, 101], Q_grid = [-2, 3])
  
    bands = [0,1,2]
    

#    fig, ax = ACQB15.plot_spectrum( fi_ext_list, [0.5],  [ [  1,2,3], [ 1,2,3]] )
#
#    im = plt.imread('spectrum2mA.png')
#    ax.imshow(im, zorder=0, extent=[-.08, 1.06, 1.5, 11.7],  aspect='auto')
#    ax.set_ylim((1.5, 11.8))
#
#    ax.legend(framealpha = 0.3)


#    im = plt.imread('spectrum.png')
#    implot = plt.imshow(im)
    
    ACQB15.plot_bands_Psi(fi_ext_list = fi_ext_list, ng_list = [  0.49 ], bands = [0,1,2])



#    J1 = 33
#    J2 = 2*J1+3
#    
#    Q = ACQubit (E_CL = 7, 
#                  E_L = 0.6, 
#                  E_J = (J1+J2)/2, 
#                  E_C = 35,
#                  dE_J = (J2-J1)/2 )
#    
#    st = Q.set_state(ng = 0.0 ,fi_ext = 0.5)
#    st.get_n_ij(0,1)

#    fiij = ACQB15.iterate_fi( fi_ext_list, 0.49, 'get_fi_ij', 0, 1)
#    
#    
#    T1_1 = ACQB15.iterate_fi( fi_ext_list, 0.49, 'get_T1', 0, 1)
#    T1_0 = ACQB15.iterate_fi( fi_ext_list, 0, 'get_T1', 0, 1)
#
#    fig, ax = plt.subplots()
#
#    ax.plot(fi_ext_list/2/pi,   T1_0, label = '0e')
#    ax.plot(fi_ext_list/2/pi,   T1_1, label = '1e')
#   
#    plt.legend()
#    plt.plot(fi_ext_list/2/pi, 1/fiij)


#    ACQB15.plot_fi_ij(fi_ext_list = fi_ext_list, ng_list = [  0.49 ], i = 0, j = 1)
# 
