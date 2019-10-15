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

pip install -e DMCpython

## Basic Examples 
```python
from dmc.dmc import DMC 
dat = DMC()
```
![alt text](/figures/figure1.png) 
```python
from dmc.dmc import DMC 
dat = DMC(tau=150)
```
![alt text](/figures/figure2.png) 
