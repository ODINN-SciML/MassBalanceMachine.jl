
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