using Revise
using MassBalanceMachine
using Test
using Random
using JSON, CSV, DataFrames
using Lux
using JET

# Include test files
include("data_reading.jl")
include("MB.jl")
include("MLPs.jl")
