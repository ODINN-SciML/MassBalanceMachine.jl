import Pkg
Pkg.activate(dirname(Base.current_project()))

using Revise
using Mimir
using Random, Statistics

# Activate the environment at the root level
using Pkg
Pkg.activate("..")

# Define paths to JSON configuration files
params_json_path = joinpath(@__DIR__, "..", "data", "nongeo_20260120_165148_norway", "params.json")
model_json_path = joinpath(@__DIR__, "..", "data", "nongeo_20260120_165148_norway", "model.json")

# Load data using model.json to get feature columns
csv_path = joinpath(@__DIR__, "..", "data", "sample_data_norway_before_norm.csv")
features, targets, feature_cols = load_data(csv_path, model_json_path, target_col="y"; normalize=true)

println("Loaded data from CSV:")
println("  Features shape: $(size(features))")
println("  Targets shape: $(size(targets))")
println("  Feature columns: $feature_cols")

# Create CustomMLP with automatic configuration from JSON files
custom_nn = CustomMLP(params_json_path, model_json_path)

println("\nLoaded CustomMLP configuration from JSON files:")
println("  Input features: $(custom_nn.nbFeatures)")
println("  Layer sizes: $(custom_nn.nNeurons)")
println("  Batch size: $(custom_nn.batch_size)")
println("  Optimizer: $(custom_nn.optimizer)")
println("  Learning rate: $(custom_nn.learning_rate)")
println("  Epochs: $(custom_nn.nepochs)")
println("  Beta1: $(custom_nn.beta1)")
println("  Beta2: $(custom_nn.beta2)")
println("  Weight decay: $(custom_nn.weight_decay)")
println("  Momentum: $(custom_nn.momentum)")
println("  Device: $(custom_nn.device)")

# Make predictions using the first batch of data
batch_size = custom_nn.batch_size
x_batch = features[:, 1:batch_size]  # First 32 samples

# Forward pass through the model
y_pred, _ = custom_nn.model(x_batch, custom_nn.params, custom_nn.state)

println("\nMade predictions:")
println("  Input batch shape: $(size(x_batch))")
println("  Output predictions shape: $(size(y_pred))")
println("  First 5 predictions: $(vec(y_pred)[1:5])")
println("  First 5 targets: $(targets[1:5])")

# Compute Mean Squared Error
mse = mean((vec(y_pred) .- targets[1:batch_size]) .^ 2)
println("  MSE on batch: $mse")