@testset "Model registry" begin
    params_json_path = tempname() * ".json"
    model_json_path = tempname() * ".json"

    params_data = Dict(
        "model" => Dict("layers" => [4]),
        "training" => Dict(
            "batch_size" => 8,
            "optim" => "ADAM",
            "lr" => 0.001,
            "Nepochs" => 10,
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
            "0.weight" => [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            "0.bias" => [0.1, 0.2, 0.3, 0.4],
            "2.weight" => [[0.2, 0.3, 0.4, 0.5]],
            "2.bias" => [0.1],
        ),
    )

    open(params_json_path, "w") do f
        JSON.print(f, params_data)
    end
    open(model_json_path, "w") do f
        JSON.print(f, model_data)
    end

    reg_dir = mktempdir()
    model_name = "unit_test_model"

    custom_nn = CustomMLP(params_json_path, model_json_path)
    x_batch = Float32[1.0 2.0 3.0; 0.1 0.2 0.3]
    y_ref, _ = custom_nn.model(x_batch, custom_nn.params, custom_nn.state)

    @testset "save/load round-trip" begin
        saved_path = save_model(custom_nn, model_name; dir = reg_dir)
        @test isfile(saved_path)

        loaded_nn = load_model(model_name; dir = reg_dir)
        y_loaded, _ = loaded_nn.model(x_batch, loaded_nn.params, loaded_nn.state)

        @test loaded_nn.nNeurons == custom_nn.nNeurons
        @test loaded_nn.input_features == custom_nn.input_features
        @test maximum(abs.(vec(y_loaded) .- vec(y_ref))) == 0.0f0
    end

    @testset "list/delete and errors" begin
        entries = list_models(; dir = reg_dir)
        @test length(entries) == 1
        @test entries[1].name == model_name

        delete_model(model_name; dir = reg_dir)
        @test_throws ErrorException load_model(model_name; dir = reg_dir)
        @test isempty(list_models(; dir = reg_dir))

        @test_throws ErrorException delete_model("does_not_exist"; dir = reg_dir)
    end

    rm(params_json_path)
    rm(model_json_path)
    rm(reg_dir; recursive = true)
end
