@testset "Data Reader Tests" begin
    # Create a temporary CSV file for testing
    csv_path = tempname() * ".csv"
    df = DataFrame(
        feature1 = [1.0, 2.0, 3.0],
        feature2 = [4.0, 5.0, 6.0],
        y = [7.0, 8.0, 9.0]
    )
    CSV.write(csv_path, df)

    # Create a temporary JSON file for testing
    json_path = tempname() * ".json"
    json_data = Dict(
        "inputs" => ["feature1", "feature2"],
        "norm" => [[0.0, 3.0], [0.0, 6.0]]
    )
    open(json_path, "w") do f
        JSON.print(f, json_data)
    end

    # Create a temporary reference JSON file for testing
    reference_path = tempname() * ".json"
    reference_data = Dict(
        "features" => [
            [0.3333333333333333, 0.6666666666666666],
            [0.6666666666666666, 0.8333333333333334],
            [1.0, 1.0]
        ]
    )
    open(reference_path, "w") do f
        JSON.print(f, reference_data)
    end

    @testset "load_data" begin
        features, targets, feature_cols = load_data(csv_path, json_path, target_col="y")

        @test size(features) == (2, 3)
        @test targets == [7.0, 8.0, 9.0]
        @test feature_cols == ["feature1", "feature2"]

        # Check if features are normalized
        @test all(0 .<= features .<= 1)
    end

    @testset "normalize!" begin
        df_test = copy(df)
        feature_symbols = [:feature1, :feature2]
        norm = [[0.0, 3.0], [0.0, 6.0]]
        normalize!(df_test, feature_symbols, norm)

        @test df_test.feature1 ≈ [0.3333333333333333, 0.6666666666666666, 1.0]
        @test df_test.feature2 ≈ [0.6666666666666666, 0.8333333333333334, 1.0]
    end

    @testset "verify_normalized_features" begin
        # Test with matching features
        @test verify_normalized_features(csv_path, json_path, reference_path, target_col="y")

        # Test with non-matching features (should warn)
        reference_data_mismatch = Dict(
            "features" => [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5]
            ]
        )
        open(reference_path, "w") do f
            JSON.print(f, reference_data_mismatch)
        end

        @test_logs (:warn,) verify_normalized_features(csv_path, json_path, reference_path, target_col="y")
    end

    # Clean up temporary files
    rm(csv_path)
    rm(json_path)
    rm(reference_path)
end
