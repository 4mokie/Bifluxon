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



def name_state(ng, fi_ext)  :
    return  'ng{:04d}_fi{:04d}'.format(int(5000+ng*1000), int(5000+fi_ext*1000) )

   
    

class ACQubit():
    
    pi = np.pi    
    
    def __init__(self, **kwargs  ):   #E_CL, E_L, E_J, E_C
        
        
        self.label = ''
        
        for key, val in kwargs.items():
            setattr(self, key, val )
            self.label += f'{key}:{val}   '
            
            
        if 'dE_J' not in kwargs.keys():
            setattr(self, 'dE_J', 0 )
        
        
        try:    
            E_CL = self.E_CL
            E_L  = self.E_L 
            E_J  = self.E_J 
            E_C  = self.E_C
            dE_J = self.dE_J
        except AttributeError:
            print('Define all the qubit parameters!')
        
        
        self.alpha = np.sqrt(E_L/4/E_CL)
        self.Vper0e = -E_J**2/4/E_C
        

        self.V_L = lambda fi, fi_ext: E_L/2* ( fi - fi_ext )**2
        self.V_J1e = lambda fi:  - E_J * np.cos(fi/2)
        self.dV_J1e = lambda fi:  - dE_J/2 * np.sin(fi/2)

        self.V_J0e = lambda fi:  self.Vper0e * np.cos(fi)


        self.V_odd1e = lambda fi, fi_ext: self.V_L(fi, fi_ext) - self.V_J1e(fi)
        self.V_even1e = lambda fi, fi_ext: self.V_L(fi, fi_ext) + self.V_J1e(fi)
        
        self.V_0e = lambda fi, fi_ext:  self.V_L(fi, fi_ext) +  self.V_J0e(fi)

        self.wpL = np.sqrt(8*E_L*E_CL)
        self.wpJ = np.sqrt(8*E_J*E_C)
        
        self.wpJ0e = np.sqrt(8*np.abs(self.Vper0e)*E_CL)
        

        self.states = []


    def set_grid(self, fi_grid, Q_grid):
        self.fi_grid = fi_grid
        self.Q_grid = Q_grid
        
        
        
    def approx_0e(self, fi_ext):
        betta = 1/self.alpha
        return self.wpL + self.Vper0e*(2 - betta/2)*np.exp( -betta/4 )*np.cos(  fi_ext )
        
        
        
    def approx_1e(self, fi_ext):

        return 2*pi**2*self.E_L*np.abs( signal.sawtooth (fi_ext, 0.5) )
        
    def set_state(self, ng , fi_ext):
            
        state_name = name_state(ng, fi_ext)
        
        if state_name not in self.states:
            setattr(self, state_name, State(ng , fi_ext)  ) 
            self.states.append(state_name)
        
        state = getattr (self, state_name)
        
        state.E_CL = self.E_CL  
        state.E_L = self.E_L
        state.E_J = self.E_J 
        state.E_C = self.E_C 
        
        state.qubit = self
        
        return state
    
    def iterate_fi(self,  fi_ext_list, ng, get_function, *args):
            output = []
#            tfi_ext_list = tqdm_notebook(fi_ext_list, leave = False, desc = f'ng = {ng}')
            tfi_ext_list = tqdm(fi_ext_list,  desc = f'ng = {ng}')
        
            for fi_ext in tfi_ext_list:
        
                st = self.set_state(ng = ng,fi_ext = fi_ext)
        
                f = getattr(st, get_function) 
        
                output.append( f(*args) )
        
             
            return   np.array(output)
            

    def iterate_ng(self,  ng_list, fi_ext, get_function, *args):
            output = []
#            tfi_ext_list = tqdm_notebook(fi_ext_list, leave = False, desc = f'ng = {ng}')
            tng_list = tqdm(ng_list,  desc = f'$\\phi_ext$ = {fi_ext}')
        
            for ng in tng_list:
        
                st = self.set_state(ng = ng,fi_ext = fi_ext)
        
                f = getattr(st, get_function) 
        
                output.append( f(*args) )
        
             
            return   np.array(output)

        









    def plot_spectrum(self, fi_ext_list, ng_list,  bands, ax = None ):

        if ax is None:
            fig, ax = plt.subplots()
        
        ax.set_title( self.label )
        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel ('$E_{0n}, GHz$')
        
   
        styles = cycle( ['-', '--', '-.', ':'])
        
        for ng in ng_list:
            
            plt.gca().set_prop_cycle(None)
            

            
        
            Espec = self.iterate_fi( fi_ext_list, ng, 'get_E')  
            
            for i, band in enumerate(bands):
                mpl.rcParams['lines.linestyle'] = next(styles) 
                for b in band:


                    ax.plot(fi_ext_list/2/pi, Espec[:, b+i] - Espec[:, i] , label = f'E{i}{b+i}, ng = {ng}')  
                
        ax.legend()
        
        return  ax 



    def plot_bands(self, ax, fi_ext_list, ng_list,  bands ):
        
        
        fi_min = np.min (fi_ext_list)
        fi_max = np.max (fi_ext_list)
        n_split = int((fi_max - fi_min)//pi)
        
        if n_split == 0 :
            n_split = 1
        
        fi_ext_list_split = np.array_split (fi_ext_list, n_split)

        ax.set_title( self.label )

        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel ('$E_{n}, GHz$')

        styles = cycle( ['-', '--', '-.', '..-'])
        
        lss = ['-',  '-.']
        
        for ng in ng_list:
            
            plt.gca().set_prop_cycle(None)
            
            mpl.rcParams['lines.linestyle'] = next(styles) 
            
        
            for b in bands:
                
                for i, fi_ext_list in  enumerate(fi_ext_list_split):
                    
#                    ls = lss[int((b + i)%2)]
                    ls = lss[int( ((b +1)//2+ i//2) %2)]
                    Eband = self.iterate_fi( fi_ext_list, ng, 'get_E',b)  
            

                    label = f'E{b}' if i is 0 else None
                    
                    ax.plot(fi_ext_list/2/pi, Eband, label = label, color = f'C{b}', ls = ls)  
                
        ax.legend(title = f'ng = {ng}')
        


    def plot_chi_i(self, fi_ext_list, ng_list,  i , freq,  ax = None  ):

        if ax is None:
            fig, ax = plt.subplots()
        
        for ng in ng_list:
        
            chi_i = self.iterate_fi( fi_ext_list, ng, 'get_chi_i', i, freq )  
            
#            for b in bands:
            ax.plot(fi_ext_list/2/pi, chi_i, label = f'ng = {ng}' )  
        

        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel (r'$\chi_{}$'.format(i))       
        ax.legend() 
        
        return ax

        
        
    def plot_fi_ij(self, fi_ext_list, ng_list,  i, j , ax = None  ):

        if ax is None:
            fig, ax = plt.subplots()
        
        for ng in ng_list:
        
            fi_ij = self.iterate_fi( fi_ext_list, ng, 'get_fi_ij', i,j)  
            
#            for b in bands:
            ax.plot(fi_ext_list/2/pi, abs(fi_ij), label = f'ng = {ng}' )  
        

        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel (r'$|\langle 0| \hat{\phi}|1 \rangle|$')       
        ax.legend() 
        
        return ax
        
        
    def plot_n_ij(self, fi_ext_list, ng_list,  i, j , ax = None  ):

        if ax is None:
            fig, ax = plt.subplots()
        
        for ng in ng_list:
        
            n_ij = self.iterate_fi( fi_ext_list, ng, 'get_n_ij', i,j)  
            
#            for b in bands:
            ax.plot(fi_ext_list/2/np.pi, abs(n_ij),'--', label = f'ng = {ng}' )  
        

        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel (r'$|\langle 0| \hat{n}|1 \rangle|$')       
        ax.legend()  
        
        return ax

    def plot_psi_ij(self, fi_ext_list, ng_list,  i, j  ):

        fig, ax = plt.subplots()
        
        for ng in ng_list:
        
            fi_ij = self.iterate_fi( fi_ext_list, ng, 'get_psi_ij', i,j)  
            
#            for b in bands:
            ax.plot(fi_ext_list/2/pi, abs(fi_ij), label = f'ng = {ng}' )  
                
        ax.set_xlabel ('$\\Phi_{ext}/\\Phi_0$')
        ax.set_ylabel (r'$\langle 0|1 \rangle$')       

        ax.legend()        


    

    def plot_Psi(self, axes, fi_ext, ng,  band ):
        
        st = self.set_state(ng, fi_ext)
#        print('plotting psi')
        
       
        for i, ax in enumerate( axes):
#            for line in ax.lines:
#                line.remove()

            Psi = 100*st.get_Psi(band, i) + 0*st.get_E( band )
            ax.plot(st.fi_list /2/pi, Psi, c='C{:1d}'.format(band), ls = '-' )


        axes[0].set_title('flux = {:1.2f}, ng = {:1.2f}'.format( fi_ext/2/pi, ng))
         
        return st.get_Psi(band, i)

    def plot_V(self, axes, fi_ext, ng ):
        
        st = self.set_state(ng, fi_ext)

        for i, ax in enumerate( axes):

#            for line in ax.lines:
#                line.remove()

#            fi = st.fi_list

            fi = np.linspace(-4*pi, 6*pi, 301)
             
            if np.round(100*ng) == 0:
                V = self.V_L(fi, fi_ext) + self.V_J0e(fi)  

                ax.plot(fi /2/pi, V, 'g.', alpha=0.7 )
                
                vmax = np.max( V )
                vmin = np.min( V )


            else:
                 Vp = self.V_L(fi, fi_ext) + self.V_J1e(fi) 
                 Vm = self.V_L(fi, fi_ext) - self.V_J1e(fi) 

                 vmax = np.max([np.max(Vp), np.max(Vm) ] )
                 vmin = np.min([np.min(Vp), np.min(Vm) ] )


                 ax.plot(fi /2/pi, Vp, 'xkcd:gray', ls = ':' , alpha=0.5 , label = '$V_-$')
                 ax.plot(fi /2/pi, Vm, 'xkcd:gray', ls = '-.',  alpha=0.5 , label = '$V_+$')
            ax.set_ylim(vmin - 0.2*np.abs(vmin),vmax) 
            ax.legend()


    def plot_bands_Psi(self,  fi_ext_list, ng_list,  bands ):

        global axes
        global cross
        global ng_plot, fi_ext_plot,  band_plot, E_plot
        
        

        def remove_lines(axes):
            for ax in axes:
               ax.lines = []
#               for line in ax.lines:
#                    line.remove()
#        
        def onclick( event, axes,axes2, cross):

            global ng_plot, fi_ext_plot,  band_plot,E_plot
           
            if event.dblclick:
                
                print(event.xdata, event.ydata)
                
                _, fi_ext_plot = find_nearest(event.xdata*2*pi, fi_ext_list )
                
                
                band, E = [],[]
                for ng in ng_list:
                    st = self.set_state(ng, fi_ext_plot)
                    Eband = st.get_E()
                    band_, E_ = find_nearest(event.ydata, Eband )
                    band.append(band_)
                    E.append(E_)

                ind, E_plot = find_nearest(event.ydata, E )
               
                ng_plot = ng_list[ind]
                band_plot = band[ind]
                
                self.plot_Psi( axes, fi_ext_plot, ng_plot,  band_plot )
                self.plot_V( axes, fi_ext_plot, ng_plot ) 
                cross.set_data(fi_ext_plot/2/pi, E_plot)
#                print(flux, ng_list[ind],  band[ind] )
            
            plt.draw()    
            plt.show()
                

        def press( event, axes,axes2, cross):

            global ng_plot, fi_ext_plot,  band_plot
            REDRAW = True

            if event.key == 'left':
                ind,  _ = find_nearest(fi_ext_plot, fi_ext_list )
                fi_ext_plot =  fi_ext_list[ind - 1]
                REDRAW = True
                
                remove_lines(axes)


            if event.key == 'right':
                ind,  _ = find_nearest(fi_ext_plot, fi_ext_list )
                fi_ext_plot =  fi_ext_list[ind + 1]
                REDRAW = True
                remove_lines(axes)
                    
            if event.key == 'up':
                band_plot =  band_plot + 1
                REDRAW = False

            if event.key == 'down':
                band_plot =  band_plot - 1
                REDRAW = False
#            if event.key != 'control':
                
            st = self.set_state(ng_plot, fi_ext_plot)
            
            E_plot = st.get_E(band_plot)
 
            self.plot_Psi( axes, fi_ext_plot, ng_plot,  band_plot )
            if REDRAW: self.plot_V( axes, fi_ext_plot, ng_plot ) 
            
            cross.set_data(fi_ext_plot/2/pi, E_plot)
            
            plt.draw()    
            plt.show()
            
            
            
        fi_ext_plot, ng_plot = [0, 0.5]
        band_plot = 0
        
        st = self.set_state( ng_plot , fi_ext_plot)
        st.get_WF()
        
        fig=plt.figure(figsize = (15,10))


        axes = []
        axes2 = []
        gs= GridSpec( st.Q_dim, 2 )

        ax_spec = fig.add_subplot(gs[:,1])                      ## plot bands
        self.plot_bands( ax_spec, fi_ext_list, ng_list,  bands )
        ax_spec.yaxis.tick_right()
        ax_spec.yaxis.set_label_position("right")
        

        E_plot = st.get_E(band_plot)
        
        cross, = ax_spec.plot (0 , E_plot, 'X', ms = 10)
#        print(st.get_E(0))


        for i, q in enumerate(st.Q_list):          ## plot set of wavefunctions for diggerent charge number
            
            if i == 0:
                ax = fig.add_subplot(gs[i,0]  )
            else:
                ax = fig.add_subplot(gs[i,0], sharey= axes[-1]  )

            ax.xaxis.set_ticks(np.arange(-5, 6))
#            ax.grid()
            ax.set_ylabel ('$V, GHz$' )
            
            ax2 = ax.twinx()
            ax2.set_ylabel ('$\\psi_{n = %1.0d}(\\phi)$ , a.u.'% q )
            axes.append(ax)
            axes2.append(ax2)
        
        axes[-1].set_xlabel ('Var flux $\\phi$')
        
        
        self.plot_Psi( axes, fi_ext_plot, ng_plot,  band_plot )        

        self.plot_V( axes, fi_ext_plot, ng_plot )        


        fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, axes,axes2, cross))
        fig.canvas.mpl_connect('key_press_event', lambda event: press(event, axes, axes2, cross))


        return fig ,ax_spec, axes, axes2
    
    
#J1 = 33
#J2 = 1*J1+0
#
#Q = ACQubit (E_CL = 20, 
#              E_L = 1.5, 
#              E_J = (J1+J2)/2, 
#              E_C = 35,
#              dE_J = (J2-J1)/2 )
#
#
#        
#st = Q.set_state(0,0)
#st.get_chi_i(0, 6)