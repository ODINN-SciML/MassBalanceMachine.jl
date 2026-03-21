mutable struct fakeIceflowCache{F <: AbstractFloat}
    MB::Matrix{F}
end

mutable struct fakeCache{ICEFLOW}
    iceflow::ICEFLOW
end

function apply_MB_test(custom_nn::CustomMLP; save_refs::Bool = false)
    rgi_ids = ["RGI60-11.03638"]

    rgi_paths = get_rgi_paths()
    # Filter out glaciers that are not used to avoid having references that depend on all the glaciers processed in Gungnir
    rgi_paths = Dict(k => rgi_paths[k] for k in rgi_ids)

    params = Muninn.Parameters(
        simulation = SimulationParameters(
        use_MB = true,
        multiprocessing = false,
        use_velocities = false,
        tspan = (2010.0, 2015.0),
        test_mode = true,
        rgi_paths = rgi_paths),
    )
    JET.@test_opt target_modules=(Sleipnir, Muninn) Muninn.Parameters(
        simulation = SimulationParameters(
        use_MB = true,
        multiprocessing = false,
        use_velocities = false,
        tspan = (2010.0, 2015.0),
        test_mode = true,
        rgi_paths = rgi_paths),
    )
    glacier = initialize_glaciers(rgi_ids, params)[1]

    model = Muninn.Model(nothing, custom_nn, nothing) # This test only needs a mass balance model
    JET.@test_opt Muninn.Model(nothing, custom_nn, nothing)

    t = 2015.0
    step_MB = 1.0/12.0
    mb = MB_timestep(model, glacier, step_MB, t)
    JET.@test_opt target_modules=(Sleipnir, Muninn) MB_timestep(model, glacier, step_MB, t)

    iceflowCache = fakeIceflowCache{Sleipnir.Float}(zero(glacier.H₀))
    cache = fakeCache{typeof(iceflowCache)}(iceflowCache)

    # TODO: This cannot be tested without Huginn. To be moved to Huginn with an extension of MassBalanceMachine.jl
    # MB_timestep!(cache, model, glacier, step_MB, t)
    # @assert mb==cache.iceflow.MB
    # JET.@test_opt target_modules=(Sleipnir, Muninn) MB_timestep!(
    #     cache, model, glacier, step_MB, t)

    if save_refs
        jldsave(joinpath(Muninn.root_dir, "test/data/MB/MB_model.jld2"); mb)
    end

    mb_ref = load(joinpath(Muninn.root_dir, "test/data/MB/MB_model.jld2"))["mb"]
    @test size(mb) == size(glacier.S)
    @test eltype(mb) <: AbstractFloat
    @test all(isfinite, mb)
    @test mb == mb_ref
end
