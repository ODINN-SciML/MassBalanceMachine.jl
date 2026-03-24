# MassBalanceMachine.jl

[![Build Status](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/MassBalanceMachine.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/MassBalanceMachine.jl)
[![CompatHelper](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CompatHelper.yml)

Porting MassBalanceMachine models into Julia and `ODINN.jl`. In order to use some of the neural network models from MassBalanceMachine, here we perform automatic translations into `Lux.jl` to be used as custom surface mass balance models (i.e. `MBmodels`) in `Muninn.jl`. 

This is work in progress. Types of models covered so far:
- MLPs
