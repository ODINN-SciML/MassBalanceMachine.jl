import Pkg
function is_included_in_repl()
    # Handle github CI
    if get(ENV, "CI_FAST", "false")=="true"
        return true
    end
    frames = StackTraces.stacktrace()
    # Handle manual include by the user in the REPL
    for frame in frames
        if occursin("start_repl_backend", string(frame.func))
            return true
        end
    end
    return false
end

Pkg.activate(dirname(Base.current_project()))
Pkg.instantiate() # Need this to setup the ODINN env for multiprocessing
if is_included_in_repl()
    # The Project.toml of the test environment to be used when running with include is in a subfolder to avoid that Julia uses this file in test mode
    Pkg.activate(dirname(Base.current_project())*"/test/test_env/")
    Pkg.resolve()
end

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
include("era5_fixture.jl")
include("MB.jl")
include("MLPs.jl")
include("model_registry.jl")
include("hugging_face.jl")
