using Pkg
Pkg.activate(@__DIR__)

using Lux
using Random

"""
    SimpleNetwork(nNeurons::Vector, activation=relu)

A simple feedforward neural network built dynamically based on layer sizes.

# Arguments
- `nNeurons::Vector`: Vector of layer sizes, must have at least 2 elements.
  Example: [input_size, hidden_size_1, ..., hidden_size_n, output_size]
- `activation`: Activation function to use between layers (default: `relu`)
  Example: `relu`, `tanh`, `sigmoid`, `gelu`, etc.
"""
function MLP(nNeurons::Vector, activation=relu)
    @assert length(nNeurons) >= 2 "nNeurons must have at least 2 elements"
    
    layers = []
    
    # First layer
    push!(layers, Dense(nNeurons[1], nNeurons[2]))
    
    # Hidden layers with activation function
    for i in 2:length(nNeurons)-1
        push!(layers, activation)
        push!(layers, Dense(nNeurons[i], nNeurons[i+1]))
    end
    
    # Final output layer (linear, no activation)
    push!(layers, Dense(nNeurons[end], 1))
    
    return Chain(layers...)
end

"""
    CustomNeuralNetRegressor

A custom neural network regressor struct that wraps a Lux model with training configuration.

# Fields
- `model`: The Lux neural network model
- `nbFeatures::Int`: Number of input features
- `nNeurons::Vector`: Layer sizes for the network
- `batch_size::Int`: Batch size for training (default: 16)
- `activation`: Activation function (default: relu)
- `device::String`: Device to run on, "cpu" or "gpu" (default: "cpu")
- `shuffle::Bool`: Whether to shuffle training data (default: true)
- `params::NamedTuple`: Model parameters
- `state::NamedTuple`: Model state
"""
struct CustomMLP
    model::Lux.AbstractLuxLayer
    nbFeatures::Int
    nNeurons::Vector
    batch_size::Int
    activation::Function
    device::String
    shuffle::Bool
    params::NamedTuple
    state::NamedTuple
end

"""
    CustomMLP(;
        nbFeatures::Int,
        nNeurons::Vector,
        batch_size::Int=16,
        activation=relu,
        device::String="cpu",
        shuffle::Bool=true
    )

Create a CustomMLP with the specified configuration.
"""
function CustomMLP(;
    nbFeatures::Int,
    nNeurons::Vector,
    batch_size::Int=16,
    activation=relu,
    device::String="cpu",
    shuffle::Bool=true
)
    # Ensure nbFeatures matches nNeurons first dimension
    @assert nNeurons[1] == nbFeatures "nNeurons[1] must equal nbFeatures"
    
    # Create the model
    model = MLP(nNeurons, activation)
    
    # Initialize parameters and state
    rng = Random.default_rng()
    params, state = Lux.setup(rng, model)
    
    return CustomMLP(
        model,
        nbFeatures,
        nNeurons,
        batch_size,
        activation,
        device,
        shuffle,
        params,
        state
    )
end

# Example usage:
rng = Random.default_rng()
nInp = 10
nNeurons = [nInp, 8]

custom_nn = CustomMLP(;
    nbFeatures=nInp,
    nNeurons=nNeurons,
    batch_size=16,
    shuffle=true,
    device="cpu"
)

println("Created CustomMLP:")
println("  Input features: $(custom_nn.nbFeatures)")
println("  Layer sizes: $(custom_nn.nNeurons)")
println("  Batch size: $(custom_nn.batch_size)")
println("  Device: $(custom_nn.device)")