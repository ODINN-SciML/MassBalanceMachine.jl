module MassBalanceMachine

using Lux
using Random
using CSV
using DataFrames
using Dates
using JLD2
using JSON
using Downloads
using Infiltrator
using Reexport
@reexport using Muninn

# ##############################################
# ############    PARAMETERS     ###############
# ##############################################

const src_dir::String = dirname(@__FILE__)
const global root_dir::String = joinpath(src_dir, "..")

include(src_dir*"/MLP.jl")
include(src_dir*"/mass_balance_utils.jl")
include(src_dir*"/data_reader.jl")
include(src_dir*"/model_registry.jl")

end
