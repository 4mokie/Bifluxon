
'''
Python module for the bifluxon qubit @ Rutgers.
'''
import sys, time, copy, itertools
from functools import reduce

from tqdm.autonotebook import tqdm

# Numpy
import numpy as np

# Scipy 
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.special import kv

# QuTip
from qutip import Options, Qobj, basis, Bloch, destroy, create, qeye, tensor, Cubic_Spline, Distribution, visualization

# Matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


############## General Parameters ##############
# Units 
nH2GHz = 163.4615  # Inductive enegry in GHz (per L in nH) 
fF2GHz = 19.37023  # Capacitive enegry in GHz (per C in fF)
Z2z    = 0.1549618 # Reduced impedance z=Z/RQ (times Z in sqrt(L in nH / C in fF)), with RQ = h/(2e)**2 superconducting resistance quantum
ω2GHz  = 159.15494 # Frequency for L = 1 nH and C = 1 fH in GHz
z2Ω    = 1000.     # Impedance for L = 1 nH and C = 1 fH in Ω
K2GHz  = 20.8366   # Temperature in GHz (times T in K)

# Diagonalization
Nφ       = 5  # number of (positive) charge states for the φ mode
nfock_φm = 30  # number of Fock states for the φ- mode
nfock_φp = 10  # number of Fock states for the φ+ mode
dev_tr   = 16  # default number device transitions
ZPFs     = 10  # number of zero-point-fluctuations for phase-space diagonalization 
ZPFs2    = 10 ## New
ZPFs3    = 10 ## New
Sx       = 101 # number of points in φm/φp space for phase-space diagonalization
mformat  = 'csr'
dtype    = 'complex128'

############## Device-class Functions ##############
def mode_params(**circuit_params):
    
    '''
    Returns the effective mode parameters from the circuit parameters. 
    '''

    # Bare capacitances
    CJ, Cg  = circuit_params['CJ'], circuit_params['Cg']
    C0, C0c = circuit_params['C0'], circuit_params['C0c']
        
    # Bare inductances
    LJ,  L = circuit_params['LJ'],  circuit_params['L']

    # Mode capacitances 
    C2    = (C0 * (C0c + Cg) + 2. * CJ * (C0 + C0c + Cg))
    Cφ1   = C2 / ((2. * CJ * (C0c + Cg + CJ) + C0 * (C0c + Cg + 2. * CJ)) / (C0c + Cg + 2. * CJ))
    Cφ2   = Cφ1
    Cφm   = (C0 + 2. * CJ) / 2.
    


    Cφp   = C2 / (2. * (C0c + Cg + 2. * CJ))
    Cφ1φp = C2 / (16. * CJ) 
    Cφ2φp = C2 / (16. * CJ) 

    # Coupling ratios
    βφ1  = Cg / Cφ1
    βφ2  = Cg / Cφ2


    # Bare mode frequencies in GHz 
    ωφm = ω2GHz / np.sqrt(Cφm * L)
    ωφp = ω2GHz / np.sqrt(Cφp * L)

    
    # Bare mode reduced impedances
    zφm = Z2z * np.sqrt(L / Cφm)
    zφp = Z2z * np.sqrt(L / Cφp)

    
    # Build circuit-mode parameters
    mparams = {'Cφ1': Cφ1, 'Cφ2': Cφ2, 'Cφ-': Cφm, 
               
               'Cφ+': Cφp, 'Cφ1φ+': Cφ1φp, 'Cφ2φ+': Cφ2φp, 
               'βφ1': βφ1, 'βφ2': βφ2,            
               'LJ': LJ, 'L': L,
               'ωφ-': ωφm, 'ωφ+': ωφp, 
               'zφ-': zφm, 'zφ+': zφp, 
               
               'φext': circuit_params['φext'],
               'ng1': circuit_params['ng1'],
               'ng2': circuit_params['ng2'],

               'dEJ': circuit_params['dEJ']}
    

    return mparams

def device_ops(eigenbasis=True, Fock_basis=False, **circuit_params):
                               
    '''
    Returns the device Hamiltonian and drive operators. 
    '''

    # Compute mode parameters
    mparams = mode_params(**circuit_params)
    EJ      = nH2GHz / mparams['LJ']



    dEJ     = mparams['dEJ'] 
    φext    = mparams['φext'] 

    if Fock_basis:
        # Single-mode operators    
        nφ   = Qobj(diags(np.arange(-Nφ,Nφ+1,1), 0, shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        cosφ = Qobj(diags([0.5,0.5], [-1,1], shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        sinφ = Qobj(diags([-.5j,.5j], [-1,1], shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        
        ωφm, zφm     = mparams['ωφ-'], mparams['zφ-']
        aφm, adgφm   = destroy(nfock_φm), create(nfock_φm)
        φm, nφm      = np.sqrt(np.pi * zφm) * (aφm + adgφm), (-1j) * (aφm - adgφm) / np.sqrt(4. * np.pi * zφm)

        cosφm, sinφm = (φm / 2.).cosm(), (φm / 2.).sinm()
        ωφp, zφp     = mparams['ωφ+'], mparams['zφ+']
        aφp, adgφp   = destroy(nfock_φp), create(nfock_φp)
        φp, nφp      = np.sqrt(np.pi * zφp) * (aφp + adgφp), (-1j) * (aφp - adgφp) / np.sqrt(4. * np.pi * zφp)
        cosφp, sinφp = (φp / 2.).cosm(), (φp / 2.).sinm()

        # Multimode operators
        dims = [2*Nφ + 1, 2*Nφ + 1, nfock_φm]       
        
        nφ1   = compose(nφ, 0, dims)
        cosφ1 = compose(cosφ, 0, dims)
        sinφ1 = compose(sinφ, 0, dims)

        nφ2   = compose(nφ, 1, dims)
        cosφ2 = compose(cosφ, 1, dims)
        sinφ2 = compose(sinφ, 1, dims)

        aφm, adgφm   = compose(aφm, 2, dims), compose(adgφm, 2, dims)
        φm, nφm      = compose(φm, 2, dims), compose(nφm, 2, dims)
        cosφm, sinφm = compose(cosφm, 2, dims), compose(sinφm, 2, dims)

        # Setup Hamiltonian
        H = 4. * (fF2GHz / mparams['Cφ1']) * (nφ1 - mparams['ng1']) * (nφ1 - mparams['ng1']) + 4. * (fF2GHz / mparams['Cφ2']) * (nφ2 - mparams['ng2']) * (nφ2 - mparams['ng2']) + ωφm * (adgφm * aφm)
        
        H += (fF2GHz / mparams['Cφ1'])*nφ1*nφ2  #cross islands -term
        H += (fF2GHz / mparams['Cφ1'])*nφ1*nφm  # island to inductor -term
        
        
        H -= 2.  * EJ * (cosφm * np.cos(φext/2.) + sinφm * np.sin(φext/2.)) * cosφ1 
        H += dEJ * EJ * (sinφm * np.cos(φext/2.) - cosφm * np.sin(φext/2.)) * sinφ1 
        H -= 2.  * EJ * (cosφm * np.cos(φext/2.) + sinφm * np.sin(φext/2.)) * cosφ2
        H += dEJ * EJ * (sinφm * np.cos(φext/2.) - cosφm * np.sin(φext/2.)) * sinφ2

        
        
        
        
    else:
        
        # Single-mode operators    
        nφ   = Qobj(diags(np.arange(-Nφ,Nφ+1,1), 0, shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        cosφ = Qobj(diags([0.5,0.5], [-1,1], shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        sinφ = Qobj(diags([-.5j,.5j], [-1,1], shape=(2*Nφ + 1, 2*Nφ + 1), format='csr', dtype='complex128'))
        
        ωφm, zφm      = mparams['ωφ-'], mparams['zφ-']
        EL, ECφm      = nH2GHz / mparams['L'], fF2GHz / mparams['Cφ-']

        Dx            = ZPFs  * ((2.0/(EL/ECφm))**(0.25))


        φm, nφm, d2φm = build_operator(operator_name='x', Dx=Dx), (-1j) * build_operator(operator_name='d1x', Dx=Dx), build_operator(operator_name='d2x', Dx=Dx)
        


        cosφm, sinφm  = build_operator(operator_name='cos(x/2)', Dx=Dx), build_operator(operator_name='sin(x/2)', Dx=Dx)

        ωφp, zφp      = mparams['ωφ+'], mparams['zφ+']
        EL, ECφp      = nH2GHz / mparams['L'], fF2GHz / mparams['Cφ+']
        Dx            = ZPFs * ((2.0/(EL/ECφp))**(0.25))
        φp, nφp, d2φp = build_operator(operator_name='x', Dx=Dx), (-1j) * build_operator(operator_name='d1x', Dx=Dx), build_operator(operator_name='d2x', Dx=Dx)
        cosφp, sinφp  = build_operator(operator_name='cos(x/2)', Dx=Dx), build_operator(operator_name='sin(x/2)', Dx=Dx)

        # Multimode operators
        dims = [2*Nφ + 1, 2*Nφ + 1, Sx]       
        
        nφ1   = compose(nφ, 0, dims)
        cosφ1 = compose(cosφ, 0, dims)
        sinφ1 = compose(sinφ, 0, dims)

        nφ2   = compose(nφ, 1, dims)
        cosφ2 = compose(cosφ, 1, dims)
        sinφ2 = compose(sinφ, 1, dims)

       
        φm, nφm, d2φm    = compose(φm, 2, dims), compose(nφm, 2, dims), compose(d2φm, 2, dims)

        cosφm, sinφm  = compose(cosφm, 2, dims), compose(sinφm, 2, dims)
        
        
   
        Ec_1 = 4. * (fF2GHz / mparams['Cφ1'])
        Ec_2 = 4. * (fF2GHz / mparams['Cφ2'])
        
        
        H = Ec_1*(nφ1 - mparams['ng1'])*(nφ1 - mparams['ng1']) + Ec_2*(nφ2 - mparams['ng2'])*(nφ2 - mparams['ng2']) 
        
        
        H += Ec_1*nφ1*nφ2  #cross islands -term
        H += Ec_1*nφ1*nφm   # island to inductor -term
        
        H -=  ( 4.*ECφm*d2φm  )       
        H +=  ( EL * (φm-φext) * (φm-φext) / 2. )                  
        H += - EJ * ( (cosφ1/2 + cosφ2)* cosφm - (sinφ1/2 - sinφ2)*sinφm + cosφ1*cosφ2 + sinφ1*sinφ2 )

    if eigenbasis:
        
        # Diagonalize Hamiltonian 
        evals, ekets = H.eigenstates(sparse=True, eigvals=dev_tr)
        evals        = evals - evals[0]
        
        # Operators in the eigenbasis
        Hd              = Qobj(diags(evals, 0, shape=(dev_tr, dev_tr), format='csr', dtype='complex128'))
        nφ1d, nφ2d, φmd = np.zeros((dev_tr, dev_tr), dtype='complex128'), np.zeros((dev_tr, dev_tr), dtype='complex128'), np.zeros((dev_tr, dev_tr), dtype='complex128')
        for i in range(dev_tr):
            ei = ekets[i]
            for j in range(dev_tr):
                ej       = ekets[j]
                nφ1d[i,j] = nφ1.matrix_element(ei, ej)
                nφ2d[i,j] = nφ2.matrix_element(ei, ej)
                φmd[i,j]  = φm.matrix_element(ei, ej)
        nφ1d = Qobj(nφ1d)
        nφ2d = Qobj(nφ2d)
        φmd  = Qobj(φmd)
        
        return {'H': Hd, 'nφ1': nφ1d, 'nφ2': nφ2d, 'φ-': φmd}

    else:
        
        return {'H': H, 'nφ1': nφ1, 'nφ2': nφ2, 'φ-': φm}

def shifts(**circuit_params):

    '''
    Returns Lamb and dispersive shifts for the bifluxon due to inductive coupling to the resonator. 
    '''

    # Mode parameters
    mparams = mode_params(**circuit_params)

    # Device operators
    dev_ops  = device_ops(eigenbasis=True, **circuit_params)

    # Resonator parameters
    ηsh    = circuit_params['ηsh']
    ωr, Zr = circuit_params['ωr'], circuit_params['Zr']

    # Matrix elements and dispersive shift
    g, tmp_χ = np.zeros((dev_tr, dev_tr), dtype='complex128'), np.zeros((dev_tr, dev_tr), dtype='float64')
    for i in range(dev_tr):
        for j in range(dev_tr):
            g[i,j]     = ηsh * (nH2GHz / mparams['L']) * np.sqrt(np.pi * Z2z * Zr) * dev_ops['φ-'][i,j]
            tmp_χ[i,j] = np.absolute(g[i,j])**2/(dev_ops['H'][i,i].real - dev_ops['H'][j,j].real - ωr)
    
    # Multilevel Lamb and dispersive shifts
    λ, χ = np.zeros(dev_tr, dtype='float64'), np.zeros(dev_tr, dtype='float64')
    for i in range(dev_tr):
        λ[i] = np.sum([tmp_χ[i,j] for j in range(dev_tr)])
        χ[i] = np.sum([(tmp_χ[i,j]-tmp_χ[j,i]) for j in range(dev_tr)])

    return λ, χ

def build_operator(operator_name='idx', Dx=None):
    
    """
    Returns sparse operators in phase space. 
    """

    x_pts = np.linspace(-Dx, Dx, Sx, endpoint=True, dtype=dtype)
    dx    = x_pts[-1] - x_pts[-2] 
    
    if operator_name == 'idx':
        op = Qobj(diags(np.ones(x_pts.size), 0, shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'd1x':
        d1_coeff = (1.0 / (2.0 * dx))
        op       = Qobj(diags([-d1_coeff, d1_coeff], [-1,1], shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'd2x':
        d2_coeff = (1.0 / (dx**2))
        op       = Qobj(diags([d2_coeff, -2.0 * d2_coeff, d2_coeff], [-1,0,1], shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'x':
        op = Qobj(diags(x_pts, 0, shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'x2':
        op = Qobj(diags(x_pts**2, 0, shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'cos(x)':
        op = Qobj(diags(np.cos(x_pts), 0, shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'sin(x)':
        op = Qobj(diags(np.sin(x_pts), 0, shape=(Sx,Sx), format=mformat, dtype=dtype))

    elif operator_name == 'cos(x/2)':
        op = Qobj(diags(np.cos(x_pts/2.), 0, shape=(Sx,Sx), format=mformat, dtype=dtype))
    
    elif operator_name == 'sin(x/2)':
        op = Qobj(diags(np.sin(x_pts/2.), 0, shape=(Sx,Sx), format=mformat, dtype=dtype))

    return op

def diagonalize_device(eigvals=dev_tr, zero=False, **circuit_params):
    
    '''
    Returns labels (up to 2 excitations), evals and ekets for the device.
    '''
    
    evals, ekets = device_ops(eigenbasis=False, **circuit_params)['H'].eigenstates(sparse=True, eigvals=eigvals)
    if zero:
        evals = evals - evals[0]

    return (evals, ekets)

def T1(two_mode=True, **circuit_params):
    
    '''
    Computes T1 coherence time for the bifluxon qubit in ns. 
    '''

    # Mode parameters
    mparams = mode_params(**circuit_params)

    # Device operators
    dev_ops  = device_ops(two_mode=two_mode, eigenbasis=True, **circuit_params)
    ω01inGHz = np.real(dev_ops['H'][1,1] - dev_ops['H'][0,0])

    # Capacitive and inductive loss
    T            = circuit_params['T_loss']
    QC, QL       = circuit_params['QC'] * (6. / ω01inGHz)**0.7, circuit_params['QL']
    γ1_cap_inGHz = (np.pi**2/2.) * np.absolute(dev_ops['φ-'][0,1])**2 * ω01inGHz**2 * (1. + 1. / np.tanh(ω01inGHz / (2. * K2GHz * T))) / (QC * (fF2GHz / mparams['Cφ-']))
    γ1_ind_inGHz = 4. * np.pi**2 * np.absolute(dev_ops['φ-'][0,1])**2 * (nH2GHz / mparams['L']) * (1. + 1. / np.tanh(ω01inGHz / (2. * K2GHz * T))) / QL

    # Charge relaxation
    T            = circuit_params['T_line']
    Renv         = circuit_params['Renv']
    γ1_ng_inGHz  = (mparams['βφ1']/(2. * np.pi))**2 * np.absolute(dev_ops['nφ1'][0,1])**2 * (Renv / 6453.201) * ω01inGHz * (1. + 1. / np.tanh(ω01inGHz / (2. * K2GHz * T)))
    γ1_ng_inGHz += (mparams['βφ2']/(2. * np.pi))**2 * np.absolute(dev_ops['nφ2'][0,1])**2 * (Renv / 6453.201) * ω01inGHz * (1. + 1. / np.tanh(ω01inGHz / (2. * K2GHz * T)))

    # Purcell decay 
    ηsh    = circuit_params['ηsh']
    ωr, Zr = circuit_params['ωr'], circuit_params['Zr']
    Qr     = circuit_params['Qr']
    γ1_Purcell_inGHz = np.pi * (Zr / 6453.201) * ηsh**2 * ωr * ((nH2GHz / mparams['L']) / (ωr - ω01inGHz))**2 / Qr

    return {'Capacitive': 1./((2. * np.pi) * γ1_cap_inGHz), 
            'Inductive': 1./((2. * np.pi) * γ1_ind_inGHz), 
            'Charge': 1./((2. * np.pi) * γ1_ng_inGHz), 
            'Purcell': 1./((2. * np.pi) * γ1_Purcell_inGHz)}
def plot_3d_Spectrum(swept_spectra, datapoints=None, title='', ωliminGHz=[0,20.], xrange=[-0.5,0.5],yrange=[-0.5,0.5], state=[0,1]):
    X = swept_spectra['vector2']
    Y = swept_spectra['vector1']
    X, Y = np.meshgrid(X, Y)
 
    i=0
    fill_this_2d_state0 = np.ones( (np.shape(swept_spectra['result'])[0] , np.shape(swept_spectra['result'])[1]) )
    for i in range(np.shape(swept_spectra['result'])[0]):
        temp = [evals[state[0]] for evals,ekets in swept_spectra['result'][i]]
        #print('temp: ' + str(temp))
        for j in range(np.shape(swept_spectra['result'])[1]):
            fill_this_2d_state0[i][j] = temp[j]
    i=0
    fill_this_2d_state1 = np.ones( (np.shape(swept_spectra['result'])[0] , np.shape(swept_spectra['result'])[1]) )
    for i in range(np.shape(swept_spectra['result'])[0]):
        temp = [evals[state[1]] for evals,ekets in swept_spectra['result'][i]]
        #print('temp: ' + str(temp))
        for j in range(np.shape(swept_spectra['result'])[1]):
            fill_this_2d_state1[i][j] = temp[j]
   

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, fill_this_2d_state0, rstride=1, cstride=1, cmap=cm.viridis)
    ax.plot_surface(X, Y, fill_this_2d_state1, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel('ng1')
    ax.set_ylabel('ng2')
    #ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi])
    #ax.set_xticklabels(["$0$", r"$\frac{\pi}{2}$",
    #                 r"$\pi$", r"$\frac{3\pi}{2}$"])
    ax.set_zlabel(r'$\omega/2\pi$ [GHz]')



    plt.show()
def plot_DoubleSpectrum(swept_spectra, datapoints=None, title='', ωliminGHz=[0,20.], xrange=[-0.5,0.5],yrange=[-0.5,0.5], state=0, diff=False, xlab='ng1', ylab='ng2', C11=1, C12=0.2):
    
    i=0
    print('state is ' + str(state))
    print(np.shape(swept_spectra['result']))
    fill_this_2d = np.ones( (np.shape(swept_spectra['result'])[0] , np.shape(swept_spectra['result'])[1]) )
    for i in range(np.shape(swept_spectra['result'])[0]):
        if (diff==True):
            temp = [evals[1]-evals[0] for evals,ekets in swept_spectra['result'][i]]
        else:
            temp = [evals[state] for evals,ekets in swept_spectra['result'][i]]
        #print('temp: ' + str(temp))
        for j in range(np.shape(swept_spectra['result'])[1]):
            fill_this_2d[i][j] = temp[j]
    #print(fill_this_2d)
    
    # Voltage Axes 
#     C21 = C12
#     C22 = C11
#     print(C11, C12)
#     V1 = (C22*swept_spectra['vector1'] - C12*swept_spectra['vector2']) / (C11*C22 - C12*C21)
#     V2 = (C11*swept_spectra['vector2'] - C21*swept_spectra['vector1']) / (C11*C22 - C21*C12)
#     print(V1)

    axs, meshs = pcolor(swept_spectra['vector1'],swept_spectra['vector2'], fill_this_2d)
    axs.set_xlabel(xlab)
    axs.set_ylabel(ylab)
    #axs.set_yticks([0., .5*np.pi, np.pi, 1.5*np.pi])
    #axs.set_yticklabels(["$0$", r"$\frac{\pi}{2}$",
    #                 r"$\pi$", r"$\frac{3\pi}{2}$"])
    axs.set_title(title)
    
############## Plot Functions ##############
def pcolor(x,y,z):
    f = plt.figure()
    ax = f.add_subplot(111)
    x1, y1 = np.meshgrid(x,y)
    #mesh = ax.pcolormesh(y1, x1 ,z, norm=colors.LogNorm())
    im = mesh = ax.pcolormesh(y1,x1,z)
    f.colorbar(im, ax=ax)
    return ax, mesh
def write_somewhere(swept_spectra, filename=''):
    f= open(filename,"w+")
    for i in range(len(swept_spectra['vector'])):
        f.write(str(swept_spectra['vector'][i]))
        f.write('\n')
        f.write(str([evals[1]-evals[0] for evals in swept_spectra['result']][i]))
        f.write('\n')
    f.close()
    
def write_double(swept_spectra, filename=''):
    f= open(filename, "w+")
    for i in range(np.shape(swept_spectra['vector1'])[0]):
        temp = [evals[1]-evals[0] for evals,ekets in swept_spectra['result'][i]]
        for j in range(np.shape(swept_spectra['vector2'])[0]):
            f.write(str(swept_spectra['vector1'][i]))
            f.write('\n')
            f.write(str(swept_spectra['vector2'][j]))
            f.write('\n')
            f.write(str(temp[j]))
            f.write('\n')
    f.close()
        
def plot_spectrum(swept_spectra, datapoints=None, title='', title2='', ωmaxinGHz=[0,20.], ωmaxinGHz2=[0,20], xlim=[0,np.pi]):

    '''
    Plots the spectra from a sweep.
    '''
   
    fig, (ax0,ax1) = plt.subplots(figsize=(9,4), ncols=2, nrows=1)
    sns.reset_orig()
    states = len(swept_spectra['result'][0][0])
    colors = sns.color_palette( n_colors=states) # See https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452
    #for i in range(states):
    for i in range(3):
        ax0.plot(swept_spectra['vector'], [evals[i] for evals, ekets in swept_spectra['result']], linestyle='-', marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color=colors[i], alpha=0.5, linewidth=2.5)
    if datapoints is not None:
        φexts, ωis = 2. * np.pi * datapoints[:,0], datapoints[:,1]
        ax.plot(φexts, ωis, linestyle='-', linewidth=0., marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color='black', alpha=0.75)
        
    for i in range(3-1):
        ax1.plot(swept_spectra['vector'], [evals[i+1]-evals[i] for evals, ekets in swept_spectra['result']], linestyle='-', marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color=colors[i+1])
        ax1.legend(['1-0', '2-0'])
        for j in range(len(swept_spectra['vector'])):
            print(swept_spectra['vector'][j], [evals[i+1]-evals[i] for evals, ekets in swept_spectra['result']][j])
    
    
    ax0.set_xlim(xlim)
    ax0.set_ylim(ωmaxinGHz)
    ax0.set_xlabel(r'{}'.format(swept_spectra['variable']))
    #ax0.set_xlabel('ng1 & ng2')
    ax0.set_ylabel(r'$\omega/2\pi$ [GHz]')
    ax0.set_title(r'{}'.format(title))
    ax1.set_title(r'{}'.format(title2))
    ax1.set_xlim(xlim)
    ax1.set_ylim(ωmaxinGHz2)
    ax1.set_xlabel(r'{}'.format(swept_spectra['variable']))
    #ax1.set_xlabel('ng1 & ng2')

    ax0.set_ylabel(r'$\omega/2\pi$ [GHz]')
#     ax0.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi,2*np.pi])
#     ax0.set_xticklabels(["$0$", r"$\frac{\pi}{2}$",
#                      r"$\pi$", r"$\frac{3\pi}{2}$",r"$2\pi$"])
#     ax1.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi,2*np.pi])
#     ax1.set_xticklabels(["$0$", r"$\frac{\pi}{2}$",
#                      r"$\pi$", r"$\frac{3\pi}{2}$",r"$2\pi$"])
    plt.tight_layout()
    plt.show()
    
def plot_eigenvec(swept_spectra, title=''):
    fig, ax0 = plt.subplots()
    sns.reset_orig()
    states = len(swept_spectra['result'][0][0])
    colors = sns.color_palette( n_colors=states) 
    evec_set0 = [ekets for evals, ekets in swept_spectra['result']][0]

    for i in range(len(evec_set0)):
        ## evec_i = [ekets for evals, ekets in swept_spectra['result']][i]   for other sets of eigenkets .. ? 
        print('Shape of ket ' + str(i) + ': ' + str(np.shape(evec_set0))) 
        print('Shape of swept_spec ' + str(i) + ': ' + str(np.shape(swept_spectra['vector'])))
        print('Type of ket ' + str(i) + ': ' + str(type(evec_set0)))
        print('Type of swept_spec ' + str(i) + ': ' + str(type(swept_spectra['vector'])))
        #print('Print evec ' + str(i) + ': ' + str(evec_i[0]))
        #ax0.plot(swept_spectra['vector'], evec_i, linestyle='-', marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color=colors[i], alpha=0.5, linewidth=2.5)
        #Distribution(evec_i, swept_spectra['vector']).visualize()
        visualization.plot_fock_distribution(evec_set0[i], fig=fig, ax=ax0, title="Qobj evec_i")
        plt.show()


    

def plot_dispersive_shift(swept_shifts, title='', maxinMHz=100.):

    '''
    Plots the shifts from a sweep.
    '''
   
    fig, ax = plt.subplots(figsize=(9,6))
    sns.reset_orig()
    colors = sns.color_palette('husl', n_colors=1) # See https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452
    ax.plot(swept_shifts['vector'], [1.e3 * (χ[1] - χ[0]) for λ, χ in swept_shifts['result']], linestyle='-', marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color=colors[0], alpha=0.5, linewidth=2.5)
    ax.set_ylim([-maxinMHz,maxinMHz])
    ax.set_xlabel(r'{}'.format(swept_shifts['variable']))
    ax.set_ylabel(r'$\chi_{01}/2\pi$ [MHz]')
    ax.set_title(r'{}'.format(title))
    plt.show()
####################################################################################################################### 
def plot_dispersive_shift_double(swept_shifts, title='', maxinMHz=100.):

    '''
    Plots the shifts from a sweep.
    '''
    fill_this_2d = np.ones( (np.shape(swept_shifts['result'])[0] , np.shape(swept_shifts['result'])[1]) )
    for i in range(np.shape(swept_shifts['result'])[0]):
        temp = [1.e3 * (χ[1] - χ[0]) for λ, χ in swept_shifts['result'][i]]
        for j in range(np.shape(swept_shifts['result'])[1]):
            fill_this_2d[i][j] = temp[j]
    
    
    # Voltage Axes 
#     C21 = C12
#     C22 = C11
#     print(C11, C12)
#     V1 = (C22*swept_shifts['vector1'] - C12*swept_shifts['vector2']) / (C11*C22 - C12*C21)
#     V2 = (C11*swept_shifts['vector2'] - C21*swept_shifts['vector1']) / (C11*C22 - C21*C12)
#     print(V1)

    axs, meshs = pcolor(swept_shifts['vector1'],swept_shifts['vector2'], fill_this_2d)
    axs.set_xlabel(r'{}'.format(swept_shifts['variable1']))
    axs.set_ylabel(r'{}'.format(swept_shifts['variable2']))
#     axs.set_ylabel(r'$\chi_{01}/2\pi$ [MHz]')
    axs.set_title(title)
    
    
    
#     fig, ax = plt.subplots(figsize=(9,6))
#     sns.reset_orig()
#     colors = sns.color_palette('husl', n_colors=1) # See https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452
#     ax.plot(swept_shifts['vector'], [1.e3 * (χ[1] - χ[0]) for λ, χ in swept_shifts['result']], linestyle='-', marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color=colors[0], alpha=0.5, linewidth=2.5)
#     ax.set_ylim([-maxinMHz,maxinMHz])

#     ax.set_title(r'{}'.format(title))
#     plt.show()
    
    
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def plot_T1(swept_T1, datapoints=None, title=''):
    
    '''
    Plots the T1 estimation from a sweep.
    '''

    fig, ax = plt.subplots(figsize=(9,6))
    sns.reset_orig()
    colors  = sns.color_palette('husl', n_colors=4) # See https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib/8391452
    T1_cap  = np.array([t1['Capacitive'] for t1 in swept_T1['result']])
    T1_ind  = np.array([t1['Inductive'] for t1 in swept_T1['result']])
    T1_ng   = np.array([t1['Charge'] for t1 in swept_T1['result']])
    T1_P    = np.array([t1['Purcell'] for t1 in swept_T1['result']])
    T1_full = 1. / (1/T1_cap + 1/T1_ind + 1/T1_ng + 1/T1_P)
    ax.plot(swept_T1['vector'], T1_cap/1000., linestyle='-', marker='o', color=colors[0], alpha=0.5, linewidth=2., markersize=2., markerfacecolor='None', markeredgewidth=1.5, label='cap. loss')
    ax.plot(swept_T1['vector'], T1_ind/1000., linestyle='-', marker='o', color=colors[1], alpha=0.5, linewidth=2., markersize=2., markerfacecolor='None', markeredgewidth=1.5, label='ind. loss')
    ax.plot(swept_T1['vector'], T1_ng/1000., linestyle='-', marker='o', color=colors[2], alpha=0.5, linewidth=2., markersize=2., markerfacecolor='None', markeredgewidth=1.5, label='Charge rel.')
    ax.plot(swept_T1['vector'], T1_P/1000., linestyle='-', marker='o', color=colors[3], alpha=0.5, linewidth=2., markersize=2., markerfacecolor='None', markeredgewidth=1.5, label='P. decay')
    ax.plot(swept_T1['vector'], T1_full/1000., linestyle='-', color='black', alpha=0.5, linewidth=1.5, label='Full')
    if datapoints is not None:
        φexts, T1s = 2. * np.pi * datapoints[:,0], 1.e6 * datapoints[:,1]
        ax.plot(φexts, T1s, linestyle='-', linewidth=0., marker='o', markersize=2.5, markerfacecolor='None', markeredgewidth=1.5, color='black', alpha=0.75)
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlabel(r'{}'.format(swept_T1['variable']))
    ax.set_ylabel(r'$T_1\,[\mu\mathrm{s}]$')
    ax.set_title(r'{}'.format(title))
    plt.show()

############## Support Functions ##############
def load_data():
    
    '''
    Loads data from device. 
    '''
    
    return {'E_0e': np.loadtxt('E_0e_vs_Flux.txt'),
            'E_1e': np.loadtxt('E_1e_vs_Flux.txt'),
            'T1_0e': np.loadtxt('T1_vs_Flux_0e.txt'),
            'T1_1e': np.loadtxt('T1_vs_Flux_1e.txt')}

def compose(operator, index, dims):

    '''
    Returns the operator composed in a multiqubit Hilbert space.
    '''

    op_list        = [qeye(dims[mode]) for mode in range(len(dims))]
    op_list[index] = operator

    return reduce(tensor, op_list)


# ##############################################################################################################################
# from progressbar import ProgressBar
# pbar = ProgressBar()
def DoubleSweep(function, sweep_variable1, sweep_variable2, sweep_vector1, sweep_vector2, **params):
    if not sweep_variable1 in list(params.keys()): raise Exception('sweep_variable1 is not in params')
    if not sweep_variable2 in list(params.keys()): raise Exception('sweep_variable2 is not in params')
    
    #sweep_results_final = np.asarray(np.ones((len(sweep_vector1),len(sweep_vector2))))
    sweep_results = []
    a = []
    original_values = copy.deepcopy(params) # copy original params values
    i = -1
    j = -1
    count = 0
    
    tsweep_vector1 = tqdm(sweep_vector1)
    
    for sweep_point1 in tsweep_vector1:
        i+=1
        params[sweep_variable1] = sweep_point1
        j=-1
        
        tsweep_vector2 = tqdm(sweep_vector2, leave = False)

        for sweep_point2 in tsweep_vector2: 
            j+=1
            count+=1
            params[sweep_variable2] = sweep_point2
            sweep_results.append(function(**params))
        #    percent = count/(len(sweep_vector1)*len(sweep_vector2))
#             if count%25==0:
#                 print( 'swept (L) ' + str(count) + ' / ' + str( len(sweep_vector1)*len(sweep_vector2) ) )
        a.append(sweep_results)
        sweep_results = []
    params = copy.deepcopy(original_values)
    print(len(a))
    print(np.shape(a))
    return {'result': a, 'variable1': sweep_variable1, 'variable2': sweep_variable2, 'vector1': sweep_vector1, 'vector2': sweep_vector2}
# ##############################################################################################################################  

def sweep(function, sweep_variable, sweep_vector, **params):

    '''
    Sweeps a function in a range of a scalar variable.
    '''

    if not sweep_variable in list(params.keys()): raise Exception('sweep_variable is not in params')

    sweep_results = []
    original_values = copy.deepcopy(params) # copy original params values
    tsweep_vector = tqdm(sweep_vector)
    for sweep_point in tsweep_vector:
        params[sweep_variable] = sweep_point
        sweep_results.append(function(**params))
    params = copy.deepcopy(original_values) # restore params to the original values

    return {'result': sweep_results, 'variable': sweep_variable, 'vector': sweep_vector}

def sweep_adv(function, sweep_variable1, sweep_variable2, sweep_vector1, sweep_vector2, **params):

    '''
    Sweeps a function in a range of 2 scalar variables.
    '''

    if not sweep_variable1 in list(params.keys()): raise Exception('sweep_variable1 is not in params')
    if not sweep_variable2 in list(params.keys()): raise Exception('sweep_variable2 is not in params')

    sweep_results = []
    original_values = copy.deepcopy(params) # copy original params values
    for sweep_point1 in sweep_vector1:
        params[sweep_variable1] = sweep_point1
        params[sweep_variable2] = sweep_point1
        sweep_results.append(function(**params))
    params = copy.deepcopy(original_values) # restore params to the original values

    return {'result': sweep_results, 'variable': sweep_variable1, 'vector': sweep_vector1}


def flatten(nested_list):

    '''
    Flattens an arbitray nested list.
    '''

    flat_list = []
    for item in nested_list:
        if type(item) == type([]):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

