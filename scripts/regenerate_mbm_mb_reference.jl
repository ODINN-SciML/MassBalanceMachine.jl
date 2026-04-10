using MassBalanceMachine
using JSON
using Muninn
using Sleipnir
using JLD2

include(joinpath(MassBalanceMachine.root_dir, "test", "era5_fixture.jl"))

const REF_PATH =
    joinpath(MassBalanceMachine.root_dir, "test", "data", "MB", "MB_model.jld2")

function build_test_custom_mlp()
    params_json_path = tempname() * ".json"
    model_json_path = tempname() * ".json"

    params_data = Dict(
        "model" => Dict("layers" => [8, 8]),
        "training" => Dict(
            "batch_size" => 16,
            "optim" => "ADAM",
            "lr" => 0.001,
            "Nepochs" => 100,
            "beta1" => 0.9,
            "beta2" => 0.999,
            "weight_decay" => 0.0,
            "momentum" => 0.0,
            "device" => "cpu",
            "shuffle" => true,
        ),
    )

    model_data = Dict(
        "inputs" => ["t2m", "tp"],
        "norm" => [[-20.0, 15.0], [0.0, 0.1]],
        "model" => Dict(
            "0.weight" => [
                [0.42, 0.18],
                [-0.31, 0.09],
                [0.27, 0.35],
                [-0.16, 0.28],
                [0.08, -0.22],
                [-0.23, 0.16],
                [0.38, 0.11],
                [-0.19, 0.31],
            ],
            "0.bias" => [0.05, 0.08, 0.03, 0.01, 0.0, 0.02, 0.04, 0.01],
            "2.weight" => [
                [0.31, -0.18, 0.12, 0.24, -0.08, 0.15, 0.20, -0.10],
                [-0.15, 0.25, 0.20, -0.10, 0.20, -0.18, 0.08, 0.15],
                [0.10, 0.20, -0.30, 0.08, 0.12, -0.10, 0.22, 0.15],
                [-0.10, -0.08, 0.10, 0.30, -0.15, 0.20, -0.12, 0.10],
                [0.20, -0.10, 0.15, -0.08, 0.30, 0.06, -0.20, 0.12],
                [-0.08, 0.15, -0.12, 0.18, -0.10, -0.28, 0.06, -0.20],
                [0.12, -0.22, 0.10, -0.10, 0.20, -0.08, 0.25, -0.08],
                [-0.22, 0.08, -0.08, 0.11, -0.12, 0.20, -0.10, 0.28],
            ],
            "2.bias" => [0.02, 0.0, -0.01, 0.01, 0.0, -0.02, 0.01, 0.0],
            "4.weight" => [[-0.28, 0.22, -0.11, -0.19, 0.33, -0.07, 0.14, -0.24]],
            "4.bias" => [-0.08],
        ),
    )

    open(params_json_path, "w") do f
        JSON.print(f, params_data)
    end
    open(model_json_path, "w") do f
        JSON.print(f, model_data)
    end

    nn = CustomMLP(params_json_path, model_json_path)
    rm(params_json_path; force = true)
    rm(model_json_path; force = true)
    return nn
end

function regenerate_reference()
    custom_nn = build_test_custom_mlp()

    rgi_ids = ["RGI60-11.03638"]
    workdir = mktempdir()

    rgi_paths = get_rgi_paths()
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)
    ensure_synthetic_era5_fixture(rgi_paths, rgi_ids)

    params = Muninn.Parameters(
        simulation = Sleipnir.SimulationParameters(
            use_MB = true,
            multiprocessing = false,
            use_velocities = false,
            tspan = (2010.0, 2015.0),
            working_dir = workdir,
            test_mode = true,
            climate_data_source = :ERA5,
            rgi_paths = rgi_paths,
        ),
    )

    glacier = initialize_glaciers(rgi_ids, params)[1]
    model = Muninn.Model(nothing, custom_nn, nothing)

    mb = MB_timestep(model, glacier, 1.0 / 12.0, 2015.0)

    mkpath(dirname(REF_PATH))
    jldsave(REF_PATH; mb)

    println("Saved MBM reference to: " * REF_PATH)
    println("Shape: " * string(size(mb)))
    println("Min/Max: " * string(minimum(mb)) * " / " * string(maximum(mb)))
end

regenerate_reference()
