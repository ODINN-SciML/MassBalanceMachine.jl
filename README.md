# MassBalanceMachine.jl

[![Build Status](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ODINN-SciML/MassBalanceMachine.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ODINN-SciML/MassBalanceMachine.jl)
[![CompatHelper](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CompatHelper.yml/badge.svg)](https://github.com/ODINN-SciML/MassBalanceMachine.jl/actions/workflows/CompatHelper.yml)

Porting MassBalanceMachine models into Julia and `ODINN.jl`. In order to use some of the neural network models from MassBalanceMachine, here we perform automatic translations into `Lux.jl` to be used as custom surface mass balance models (i.e. `MBmodels`) in `Muninn.jl`. 

This is work in progress. Types of models covered so far:
- MLPs

## Usage

### Loading a trained model from MassBalanceMachine

Pre-trained MassBalanceMachine MLP models are exported from Python as a pair of JSON files (`params.json` and `model.json`). Load them into a `CustomMLP` instance with:

```julia
using MassBalanceMachine

mlp = CustomMLP("path/to/params.json", "path/to/model.json")
```

`CustomMLP` is a subtype of `MBmodel` (from `Muninn.jl`) and can be used directly as a surface mass balance model in `ODINN.jl`:

```julia
using ODINN

model = Model(
    iceflow = SIA2Dmodel(params),
    mass_balance = mlp,
)
```

### Model registry

Once loaded, a model can be saved to the local registry to avoid re-parsing JSON on every run:

```julia
save_model(mlp, "norway_nongeo")   # saves to ~/.MassBalanceMachine/models/
mlp = load_model("norway_nongeo")  # fast retrieval by name
list_models()                      # show all registered models
```

### Model architecture

`CustomMLP` wraps a `Lux.jl` feedforward network whose architecture (layer sizes, activation function) and normalisation bounds are read directly from the JSON export of [MassBalanceMachine](https://github.com/ODINN-SciML/MassBalanceMachine). The network takes monthly ERA5 climate features as inputs (e.g. `t2m`, `tp`, `ssrd`, …) and outputs a surface mass balance rate in m w.e. per time step.

A lower-level `MLP(nNeurons, activation)` constructor is also available if you want to build a network from scratch:

```julia
net = MLP([2, 8, 8, 1], relu)   # 2 inputs → two 8-neuron hidden layers → 1 output
```
