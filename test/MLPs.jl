@testset "MLP tests" begin
    # Create temporary JSON files for testing
    params_json_path = tempname() * ".json"
    model_json_path = tempname() * ".json"

    # Create params.json content
    params_data = Dict(
        "model" => Dict(
            "layers" => [8, 8],  # Two hidden layers
        ),
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

    # Create model.json content that matches the expected architecture
    # For a network with input size 2, hidden layers [8,8], and output size 1
    model_data = Dict(
        "inputs" => ["t2m", "tp"],
        "norm" => [[-20.0, 15.0], [0.0, 0.1]],
        "model" => Dict(
            # First hidden layer (input_size=2, output_size=8)
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

            # Second hidden layer (input_size=8, output_size=8)
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

            # Output layer (input_size=8, output_size=1)
            "4.weight" => [[-0.28, 0.22, -0.11, -0.19, 0.33, -0.07, 0.14, -0.24]],
            "4.bias" => [-0.08],
        ),
    )

    # Write JSON files
    open(params_json_path, "w") do f
        JSON.print(f, params_data)
    end

    open(model_json_path, "w") do f
        JSON.print(f, model_data)
    end

    @testset "MLP Creation" begin
        nNeurons = [2, 8, 8, 1]  # Input, two hidden, output
        model = MLP(nNeurons)
        @test model isa Lux.Chain
        JET.@test_opt MLP(nNeurons)
    end

    @testset "CustomMLP Creation" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)

        @test custom_nn.nbFeatures == 2
        @test custom_nn.nNeurons == [2, 8, 8, 1]
        @test custom_nn.input_features == ["t2m", "tp"]
        @test custom_nn.norm ==
              [(Float32(-20.0), Float32(15.0)), (Float32(0.0), Float32(0.1))]
    end

    @testset "CustomMLP Climate Source Requirement" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)

        @test required_climate_data_source(custom_nn) == :ERA5
        @test_throws ArgumentError validate_climate_data_source(custom_nn, :W5E5)
        @test isnothing(validate_climate_data_source(custom_nn, :ERA5))
    end

    @testset "Weight Injection" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)

        # Check if weights were properly injected
        # First layer weights (input_size=2, output_size=8)
        w1 = custom_nn.params.layer_1.weight
        @test size(w1) == (8, 2)
        @test w1[1, 1] ≈ 0.42
        @test w1[2, 2] ≈ 0.09

        # First layer bias
        b1 = custom_nn.params.layer_1.bias
        @test length(b1) == 8
        @test b1[1] ≈ 0.05
        @test b1[8] ≈ 0.01

        # Second layer weights (input_size=8, output_size=8)
        w2 = custom_nn.params.layer_3.weight
        @test size(w2) == (8, 8)
        @test w2[1, 1] ≈ 0.31
        @test w2[8, 8] ≈ 0.28

        # Second layer bias
        b2 = custom_nn.params.layer_3.bias
        @test length(b2) == 8
        @test b2[1] ≈ 0.02
        @test b2[8] ≈ 0.0

        # Output layer weights (input_size=8, output_size=1)
        w3 = custom_nn.params.layer_5.weight
        @test size(w3) == (1, 8)
        @test w3[1, 1] ≈ -0.28
        @test w3[1, 8] ≈ -0.24

        # Output layer bias
        b3 = custom_nn.params.layer_5.bias
        @test length(b3) == 1
        @test b3[1] ≈ -0.08
    end

    @testset "Apply ML MB model" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)
        apply_MB_test(custom_nn; save_refs = false)
    end

    # Clean up temporary files
    rm(params_json_path)
    rm(model_json_path)
end
