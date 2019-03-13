# DMC 
Python implementation of the diffusion process model (Diffusion Model
for Conflict Tasks, DMC) presented in Automatic and controlled stimulus
processing in conflict tasks: Superimposed diffusion processes and delta
functions
(https://www.sciencedirect.com/science/article/pii/S0010028515000195). 

NB. See also R/Cpp package DMCfun for further functionality including fitting
procedures.

## Installation
git clone https://github.com/igmmgi/DMCpython.git 

cd DMCpython

pip install -e dmc

## Basic Examples 
```python
from dmc import dmc_sim
dmc_sim.simulation()
```
![alt text](/figures/figure1.png) 
```python
dmc_sim.simulation(tau=150)
```
![alt text](/figures/figure2.png) 
