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

cd(@__DIR__)
const global root_dir::String = dirname(Base.current_project())

include("MLP.jl")
include("mass_balance_utils.jl")
include("data_reader.jl")
include("model_registry.jl")

end
