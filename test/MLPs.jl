@testset "MLP tests" begin
    # Create temporary JSON files for testing
    params_json_path = tempname() * ".json"
    model_json_path = tempname() * ".json"

    # Create params.json content
    params_data = Dict(
        "model" => Dict(
            "layers" => [8, 8]  # Two hidden layers
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
            "shuffle" => true
        )
    )

    # Create model.json content that matches the expected architecture
    # For a network with input size 2, hidden layers [8,8], and output size 1
    model_data = Dict(
        "inputs" => ["feature1", "feature2"],
        "model" => Dict(
            # First hidden layer (input_size=2, output_size=8)
            "0.weight" => [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],
                          [0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
            "0.bias" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

            # Second hidden layer (input_size=8, output_size=8)
            "2.weight" => [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                           [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                           [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
                           [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
                           [3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
                           [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8],
                           [4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
                           [5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]],
            "2.bias" => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],

            # Output layer (input_size=8, output_size=1)
            "4.weight" => [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
            "4.bias" => [0.1]
        )
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
    end
    
    @testset "CustomMLP Creation" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)

        @test custom_nn.nbFeatures == 2
        @test custom_nn.nNeurons == [2, 8, 8, 1]
        @test custom_nn.batch_size == 16
        @test custom_nn.device == "cpu"
        @test custom_nn.optimizer == "ADAM"
        @test custom_nn.learning_rate ≈ 0.001
    end

    @testset "Weight Injection" begin
        custom_nn = CustomMLP(params_json_path, model_json_path)

        # Check if weights were properly injected
        # First layer weights (input_size=2, output_size=8)
        w1 = custom_nn.params.layer_1.weight
        @test size(w1) == (8, 2)
        @test w1[1,1] ≈ 0.1
        @test w1[2,2] ≈ 0.4

        # First layer bias
        b1 = custom_nn.params.layer_1.bias
        @test length(b1) == 8
        @test b1[1] ≈ 0.1
        @test b1[8] ≈ 0.8

        # Second layer weights (input_size=8, output_size=8)
        w2 = custom_nn.params.layer_3.weight
        @test size(w2) == (8, 8)
        @test w2[1,1] ≈ 0.1
        @test w2[8,8] ≈ 6.4

        # Second layer bias
        b2 = custom_nn.params.layer_3.bias
        @test length(b2) == 8
        @test b2[1] ≈ 0.1
        @test b2[8] ≈ 0.8

        # Output layer weights (input_size=8, output_size=1)
        w3 = custom_nn.params.layer_5.weight
        @test size(w3) == (1, 8)
        @test w3[1,1] ≈ 0.1
        @test w3[1,8] ≈ 0.8

        # Output layer bias
        b3 = custom_nn.params.layer_5.bias
        @test length(b3) == 1
        @test b3[1] ≈ 0.1
    end

    # Clean up temporary files
    rm(params_json_path)
    rm(model_json_path)
end

