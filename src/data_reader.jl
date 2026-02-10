export load_data, load_params_from_json, verify_normalized_features, normalize!

"""
    load_data(csv_filepath::String, json_filepath::String; target_col::String="y")

Load data from a CSV file using feature columns specified in a model.json file.

# Arguments
- `csv_filepath::String`: Path to the CSV file
- `json_filepath::String`: Path to the model.json file containing feature names
- `target_col::String`: Name of the target column (default: "y")

# Returns
- `Tuple`: (features, targets, feature_cols) where features is (n_features, n_samples)
"""
function load_data(csv_filepath::String, json_filepath::String; target_col::String="y", normalize::Bool=true)
    # Load JSON file to get feature names
    json_data = JSON.parsefile(json_filepath)
    feature_cols = json_data["inputs"]
    norm = json_data["norm"] 
    
    # Load CSV file
    df = CSV.read(csv_filepath, DataFrame)
    
    # Remove rows with missing values
    df = dropmissing(df)
    
    # Extract target column
    if !hasproperty(df, Symbol(target_col))
        error("Target column '$target_col' not found in CSV file")
    end
    targets = vec(df[!, Symbol(target_col)])
    
    # Extract features using columns from JSON
    feature_symbols = Symbol.(feature_cols)
    
    # Normalize features
    if normalize
        normalize!(df, feature_symbols, norm)
    end
    
    # Convert to Float32
    features = Matrix(df[!, feature_symbols])'  # Transpose to (n_features, n_samples)
    features = Float32.(features)
    targets = Float32.(targets)
    
    return features, targets, feature_cols
end

"""
    normalize(df::DataFrame, feature_symbols::Vector{Symbol})

Normalize specified feature columns in the DataFrame to the range [0, 1].

# Arguments
- `df::DataFrame`: Input DataFrame
- `feature_symbols::Vector{Symbol}`: Vector of column symbols to normalize
"""
function normalize!(df::DataFrame, feature_symbols::Vector{Symbol}, norm)
    for (sym, bounds)in zip(feature_symbols, norm)
        feature = Float64.(df[!, sym])
        min = bounds[1]
        max = bounds[2]
        df[!, sym] = (feature .- min) ./ (max - min)
    end
end


"""
    verify_normalized_features(csv_filepath::String, json_filepath::String, reference_filepath::String; target_col::String="y")

Verify that the generated normalized features match exactly the reference file.

# Arguments
- `csv_filepath::String`: Path to the CSV file
- `json_filepath::String`: Path to the model.json file containing feature names
- `reference_filepath::String`: Path to the reference JSON file containing normalized features
- `target_col::String`: Name of the target column (default: "y")

# Returns
- `Bool`: True if the normalized features match the reference file, False otherwise
"""
function verify_normalized_features(csv_filepath::String, json_filepath::String, reference_filepath::String; target_col::String="y")
    # Load the reference data
    reference_data = JSON.parsefile(reference_filepath; allownan=true)

    # Load the generated data
    features, targets, feature_cols = load_data(csv_filepath, json_filepath, target_col=target_col)

    # Extract the features from the reference data
    reference_features = reference_data["features"]

    # Compare the features
    for i in 1:size(features, 2)
        generated_features = features[:, i]

        if !isapprox(reference_features[i], generated_features, atol=1e-6)
            println("Mismatch found at index $i")
            println("Reference features: $(reference_features[i])")
            println("Generated features: $generated_features")
            @warn "Normalized features do not match the reference file at index $i"
            return false
        end
    end

    println("\nNormalized features match the reference file.")
    return true
end
   
