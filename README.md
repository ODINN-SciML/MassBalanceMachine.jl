# MassBalanceMachine.jl

Porting MassBalanceMachine models into Julia and `ODINN.jl`. In order to use some of the neural network models from MassBalanceMachine, here we perform automatic translations into `Lux.jl` to be used as custom surface mass balance models (i.e. `MBmodels`) in `Muninn.jl`. 

This is work in progress. Types of models covered so far:
- MLPs
