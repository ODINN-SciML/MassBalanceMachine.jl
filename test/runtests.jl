using Revise
using MassBalanceMachine
using Test
using Random
using JSON, CSV, DataFrames
using Lux
using JET
using JLD2: load, jldsave

# Include test files
include("data_reading.jl")
include("MB.jl")
include("MLPs.jl")
