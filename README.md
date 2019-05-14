# Bifluxon (Aharonov-Casher) Qubit

## Intro
This project allows to numerically study [Aharonov-Casher qubit](https://doi.org/10.1103/PhysRevLett.116.107002), namely calculate wavefunctions and energy levels, visualize energy spectra, evaluate decay rates and etc. The main feature of this obejected-oriented code is  save time and recources - once the Hamiltonian was solved for certain point, the information about energies and wavefunctions is storing and can be used in futher calculations.

## Installation

Download files

```
wavefunction1d.py
ACstate.py
ACqubit.py
```
import to your .ipynb or .py file and start playing!

## Structure of the code

One can create an instance of ACQubit class, providing the dict of qubit parameters (`E_CL, E_L, E_J, E_C, dE_J` in GHz)

```py
ACQB15 = ACQubit (E_CL = 15, 
                  E_L = 0.87, 
                  E_J = 33, 
                  E_C = 8,
                  dE_J = 3 )
```
or 

```py
param_dict = {'E_CL': 15,  'E_J' : 33, 'E_C' : 8, 'dE_J' : 3 }
ACQB15 = ACQubit (**param_dict)
```

### AC qubit methods

- ` set_state(ng , fi_ext)`
  creates (or retrieve, if was created before) attribute, corresponds to the given gate charge `ng` (in units of e) and the external flux `fi_ext` (in rad). Energies and wavefuctions can be calculated for particular state, see more detail in [the next section](#AC-state-methods)



- ` iterate_fi( fi_ext_list, ng, get_function, *args)`
  iterates over `fi_ext_list` for given `ng`, returns the np.array of results `get_function(*args)` for each point. List of `get_functions` see in  [the next section](#AC-state-methods) 



- ` iterate_fi( ng_list, fi_ext, get_function, *args)`
  iterates over `ng_ext_list` for given `fi_ext`, returns the np.array of results `get_function(*args)` for each point. List of `get_functions` see in  [the next section](#AC-state-methods) 


- `plot_spectrum( fi_ext_list, ng_list,  bands, ax = None )`
  plots E_0i transitions vs `fi_ext_list`, for i listed in `bands`, for gate charges from `ng_list` on given plt.axis or create new. Returns axis


- `plot_bands( ax, fi_ext_list, ng_list,  bands )` !! unify all plotting procedures

- `plot_chi_i( fi_ext_list, ng_list,  i , freq,  ax = None  )`

- `plot_fi_ij( fi_ext_list, ng_list,  i, j , ax = None  )`

- `plot_n_ij( fi_ext_list, ng_list,  i, j , ax = None  )`

- `plot_psi_ij( fi_ext_list, ng_list,  i, j  )`

- `plot_bands_Psi( fi_ext_list, ng_list,  bands )`



### AC state methods
   
- `calc_WF( fi_grid , Q_grid  )`
  calculates wavefunctions and energies for given   `fi_grid = [min_fi, max_fi, N_pts]` and `Q_grid = [min_Q, max_Q]`

- `get_WF()`
  checks if wavefunctions and energies are already calculated, and calculates if necessary

- `get_E(i)`
  returns energy of `i`-th level, or array of all levels if `i` is not given

- `get_Psi( band, q)`
  returns 1d wavefunction of fi at the level `band`, and certain charge variable `q` (not confuse with induced gate charge!)

  
  - `get_fi_ij( i, j )`
      returns phase matrix element between states `i` and `j`

- `get_n_ij( i, j )`
      returns charge matrix element between states `i` and `j`

- `get_psi_ij( i, j )`
      returns overlapping of states `i` and `j`

- `get_chi_i( i, freq )`  !! add coupling
  returns dispersive shift of the resonator with frequency `freq`, when qubit is in the state `i`
  
- `get_T1(fi_ext, ng, i = 0, j = 1)` !!kill fiext, ng, add noise ampl
  



