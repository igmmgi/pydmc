# pydmc 
Python implementation of the diffusion process model (Diffusion Model
for Conflict Tasks, DMC) presented in Automatic and controlled stimulus
processing in conflict tasks: Superimposed diffusion processes and delta
functions
(https://www.sciencedirect.com/science/article/pii/S0010028515000195). 

NB. See also R/Cpp package DMCfun for further functionality including fitting
procedures.

## Installation
git clone https://github.com/igmmgi/pydmc.git 

pip install -e pydmc

## Basic Examples 
```python
from pydmc.dmc import DmcSim 
dat = DmcSim(full_data=True)
dat.plot()
```
![alt text](/figures/figure1.png) 
```python
from pydmc.dmc import DmcSim 
dat = DmcSim(full_data=True, tau=150)
dat.plot()
```
![alt text](/figures/figure2.png) 
