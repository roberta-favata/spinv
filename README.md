# SPInv
SPInv (Single-Point Invariants) is a Python package calculating topological invariants in the supercell framework for non-crystalline 2D topological insulators. The single-point formulas for the Chern number [[Ceresoli-Resta](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.012405)] and for the spin Chern number [arXiv] are implemented in SPInv. The code provides dedicated interfaces to tight-binding packages [PythTB](http://www.physics.rutgers.edu/pythtb/) and [Tbmodels](https://tbmodels.greschd.ch/en/latest/).

## Quick start
Here, two examples for calculating the single-point topological invariant in the supercell framework for tight-binding models in presence of Anderson disorder. 

### Kane-Mele example
Let's consider as prototype of quantum spin Hall insulator the Kane-Mele model and calculate the topological invariant through the single-point spin Chern number formulation.  

First create the tight-binding model in primitive cell using PythTB and Tbmodels packages. Create a supercell, for example including L lattice points in each lattice vector direction.  
Then add disorder in the supercell, for instance Anderson disorder which is a uniformly distributed random on site potential.  
For a reference implementation, look at [Kane-Mele example](spinv/example_models/kane_mele.py).

```python
import numpy as np
from spinv import single_point_spin_chern
from spinv.example_models import kane_mele_pythtb, kane_mele_tbmodels, km_anderson_disorder_pythtb, km_anderson_disorder_tbmodels

L = 12              # supercell LxL linear size

# Topological phase
r = 1.              # r = rashba/spin_orb
e = 3.              # e = e_onsite/spin_orb
spin_o = 0.3        # spin_orb = spin_orb/t (t=1)

# # Trivial phase
# r = 3. 
# e = 5.5
# spin_o = 0.3

# create Kane-Mele model in supercell LxL through PythTB package
km_pythtb_model = kane_mele_pythtb(r, e, spin_o, L)

# create Kane-Mele model in supercell LxL through TBmodels package
km_tbmodels_model = kane_mele_tbmodels(r, e, spin_o, L)

w = 1.              # w = disorder strength such that random potential is in [-w/2, w/2]      

# add Anderson disorder in PythTB model
np.random.seed(10)
km_pythtb_model = km_anderson_disorder_pythtb(km_pythtb_model, w)

# add Anderson disorder in TBmodels model
np.random.seed(10)
km_tbmodels_model = km_anderson_disorder_tbmodels(km_tbmodels_model, w)
```
Finally, use single_point_spin_chern function to calculate the single-point invariant for the disordered model in the supercell framework.   
The function takes as inputs:
- the model created with PythTB or TBmodels packages 
- variable "spin" indicating which spin Chern number we want to calculate : 'up' or 'down'
- variable "formula" selecting which single-point formula we want to calculate : 'asymmetric', 'symmetric' or 'both'

```python
spin_chern = 'down'
which_formula = 'symmetric'

# Single Point Spin Chern Number (SPSCN) calculation for Pythtb model
spin_chern_pythtb = single_point_spin_chern(km_pythtb_model, spin=spin_chern, formula=which_formula)

# Single Point Spin Chern Number (SPSCN) calculation for TBmodels model
spin_chern_tbmodels = single_point_spin_chern(km_tbmodels_model, spin=spin_chern, formula=which_formula)

# if which_formula = 'both', then Single Point Spin Chern numbers are printed as follows : 'asymmetric' 'symmetric'
print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_pythtb )
print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPSCN :', *spin_chern_tbmodels )
```

### Haldane example
The single-point invariant calculation can be also performed for a Chern insulator, like the Haldane model, using single_point_chern function.   
For the model implementation, look at [Haldane example](spinv/example_models/haldane.py).
```python
import numpy as np
from spinv import single_point_chern
from spinv.example_models import haldane_pythtb, haldane_tbmodels, h_anderson_disorder_pythtb, h_anderson_disorder_tbmodels

L = 9                   # supercell LxL linear size

# Topological phase
t = -4.                 # t = first neighbours real hopping
t2 = 1.                 # t2 = second neighbours hopping
delta = 2.              # delta = energy on site
phi = -np.pi/2.0        # phi = second neighbours hopping phase

# # Trivial phase
# t = -4.                   
# t2 = 1.                   
# delta = 4.5               
# phi = -np.pi/6.0        
   
# create Haldane model in supercell LxL through PythTB package
h_pythtb_model = haldane_pythtb(delta, t, t2, phi, L)

# create Haldane model in supercell LxL through TBmodels package
h_tbmodels_model = haldane_tbmodels(delta, t, t2, phi, L)

w = 1.                  # w = disorder strength such that random potential is in [-w/2, w/2]

# add Anderson disorder in PythTB model
np.random.seed(10)
h_pythtb_model = h_anderson_disorder_pythtb(h_pythtb_model, w)

# add Anderson disorder in TBmodels model
np.random.seed(10)
h_tbmodels_model = h_anderson_disorder_tbmodels(h_tbmodels_model, w)

which_formula = 'both'

# Single Point Chern Number (SPCN) calculation for Pythtb model
chern_pythtb = single_point_chern(h_pythtb_model, formula=which_formula)

# Single Point Chern Number (SPCN) calculation for TBmodels model
chern_tbmodels = single_point_chern(h_tbmodels_model, formula=which_formula)

# if which_formula = 'both', then Single Point Chern Numbers are printed as follows : 'asymmetric' 'symmetric'
print('PythTB package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_pythtb )
print('TBmodels package, supercell size L =', L, ' disorder strength = ', w,  ' SPCN :', *chern_tbmodels )
```
## Installation
```
git clone https://github.com/roberta-favata/spinv.git
cd spinv
pip install .
```

## Authors
Roberta Favata and Antimo Marrazzo