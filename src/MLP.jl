export MLP, CustomMLP

"""
    MLP(nNeurons::Vector, activation=relu)

A simple feedforward neural network built dynamically based on layer sizes.

# Arguments
- `nNeurons::Vector`: Vector of layer sizes, must have at least 2 elements.
  Example: [input_size, hidden_size_1, ..., hidden_size_n, output_size]
- `activation`: Activation function to use between layers (default: `relu`)
  Example: `relu`, `tanh`, `sigmoid`, `gelu`, etc.
"""
function MLP(nNeurons::Vector{Int}, activation=relu)
    @assert length(nNeurons) >= 2 "nNeurons must have at least 2 elements"
    
    layers = []
    
    # Build all Dense layers with activations between them
    for i in 1:length(nNeurons)-2
        push!(layers, Dense(nNeurons[i], nNeurons[i+1]))
        push!(layers, activation)
    end
    
    # Final output layer (linear, no activation)
    push!(layers, Dense(nNeurons[end-1], nNeurons[end]))
    
    return Chain(layers...)
end

"""
    CustomMLP

A custom neural network regressor struct that wraps a Lux model with training configuration.
All configuration is automatically loaded from JSON files.

# Fields
- `model`: The Lux neural network model
- `nbFeatures::Int`: Number of input features
- `nNeurons::Vector`: Layer sizes for the network
- `batch_size::Int`: Batch size for training
- `activation`: Activation function
- `device::String`: Device to run on ("cpu" or "gpu")
- `shuffle::Bool`: Whether to shuffle training data
- `optimizer::String`: Optimizer type (e.g., "ADAM", "SGD")
- `learning_rate::Float32`: Learning rate
- `nepochs::Int`: Number of training epochs
- `beta1::Float32`: Beta1 parameter for ADAM
- `beta2::Float32`: Beta2 parameter for ADAM
- `weight_decay::Float32`: Weight decay regularization
- `momentum::Float32`: Momentum for SGD
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
    optimizer::String
    learning_rate::Float32
    nepochs::Int
    beta1::Float32
    beta2::Float32
    weight_decay::Float32
    momentum::Float32
    params::NamedTuple
    state::NamedTuple
end

"""
    CustomMLP(params_json::String, model_json::String)

Create a CustomMLP by loading all configuration from JSON files.

# Arguments
- `params_json::String`: Path to params.json file containing training hyperparameters and network architecture
- `model_json::String`: Path to model.json file containing input feature names

# Returns
- `CustomMLP`: Fully configured custom MLP instance
"""
function CustomMLP(params_json::String, model_json::String)
    # Load params.json for architecture and training config
    params_data = JSON.parsefile(params_json)
    
    # Load model.json for input features and (optionally) weights
    model_data = JSON.parsefile(model_json)
    
    # Extract input features from model.json
    input_features = model_data["inputs"]
    nbFeatures = length(input_features)
    
    # Extract network architecture from params.json
    model_config = params_data["model"]
    hidden_layers = model_config["layers"]
    nNeurons = vcat(nbFeatures, hidden_layers..., 1)
    
    # Extract training configuration from params.json
    training_config = params_data["training"]
    batch_size = training_config["batch_size"]
    optimizer = training_config["optim"]
    learning_rate = Float32(training_config["lr"])
    nepochs = training_config["Nepochs"]
    beta1 = Float32(training_config["beta1"])
    beta2 = Float32(training_config["beta2"])
    weight_decay = Float32(training_config["weight_decay"])
    momentum = Float32(training_config["momentum"])
    
    # Device configuration
    device = get(training_config, "device", "cpu")
    shuffle = get(training_config, "shuffle", true)
    
    # Create the model with relu activation
    model = MLP(nNeurons, relu)
    
    # Initialize parameters and state
    rng = Random.default_rng()
    params, state = Lux.setup(rng, model)
    
    # Direct injection using JSON parsed data
    params = inject_weights_from_json(params, model_data)
    
    return CustomMLP(
        model,
        nbFeatures,
        nNeurons,
        batch_size,
        relu,
        device,
        shuffle,
        optimizer,
        learning_rate,
        nepochs,
        beta1,
        beta2,
        weight_decay,
        momentum,
        params,
        state
    )
end

"""
    inject_weights_from_json(params_nt::NamedTuple, model_data::Dict)

Inject weights and biases from JSON model data directly into params NamedTuple.
Matches the hierarchical structure of Lux params exactly.
Verifies consistency between JSON and Lux-generated structure.
"""
function inject_weights_from_json(params_nt::NamedTuple, model_data::JSON.Object)
    !haskey(model_data, "model") && return params_nt
    
    flat = model_data["model"]
    dense_idx = Ref(0)
    
    # Extract expected layer sizes from JSON
    json_layers = _extract_layer_sizes_from_json(flat)
    
    # Extract actual layer sizes from params_nt
    lux_layers = _extract_layer_sizes_from_params(params_nt)
    
    # Verify consistency
    _verify_layer_consistency(json_layers, lux_layers)
    
    # Recursively walk params_nt and inject JSON values in order
    function recursively_inject_weights(x::NamedTuple)
        updated_layers = Dict{Symbol,Any}()
        layer_names = keys(x)  # Get all layer names
        n_layers = length(layer_names)

        # Iterate in steps of 2 to skip activation layers
        for i in 1:2:n_layers
            layer_name = layer_names[i]
            layer = x[layer_name]
            println("Visiting layer: $layer_name")

            if layer isa NamedTuple && (hasproperty(layer, :weight) || hasproperty(layer, :bias))
                println("Processing layer: $layer_name")
                idx_str = string(dense_idx[])

                updates = Dict{Symbol,Any}()

                if haskey(flat, "$idx_str.weight") && hasproperty(layer, :weight)
                    println("Injecting weights for layer $idx_str")
                    w_json = _json_to_array(flat["$idx_str.weight"])
                    w_json = Float32.(w_json)
                    @assert size(w_json) == size(layer.weight) "Weight shape mismatch at layer $idx_str: JSON $(size(w_json)) vs Lux $(size(layer.weight))"
                    updates[:weight] = w_json
                end

                if haskey(flat, "$idx_str.bias") && hasproperty(layer, :bias)
                    println("Injecting bias for layer $idx_str")
                    b_json = _json_to_array(flat["$idx_str.bias"])
                    b_json = Float32.(b_json)
                    @assert size(b_json) == size(layer.bias) "Bias shape mismatch at layer $idx_str: JSON $(size(b_json)) vs Lux $(size(layer.bias))"
                    updates[:bias] = b_json
                end

                if isempty(updates)
                    @error "No matching weights or biases found in JSON for layer $idx_str"
                end

                updated_layer = merge(layer, (updates...,))
                updated_layers[layer_name] = updated_layer

                dense_idx[] += 2  # Increment by 2 to skip activation layers
            else
                @error "Unexpected layer structure in params at layer $layer_name"
            end
        end

        # Copy over the empty NamedTuples (e.g., layer_2, layer_4, etc.)
        for i in 2:2:n_layers
            layer_name = layer_names[i]
            updated_layers[layer_name] = x[layer_name]
        end
        
        # Sort the updated_layers by layer index to maintain order
        sorted_keys = sort(collect(keys(updated_layers)), by = k -> parse(Int, split(string(k), '_')[end]))
        sorted_keys_tuple = Tuple(sorted_keys)

        # Evaluate the generator and convert to a tuple
        values_tuple = tuple([updated_layers[k] for k in sorted_keys]...)

        # Construct the NamedTuple
        sorted_nt = NamedTuple{sorted_keys_tuple}(values_tuple)

        return sorted_nt
    end

    return recursively_inject_weights(params_nt)
end

# Helper: extract layer dimensions (in, out) from JSON weights
function _extract_layer_sizes_from_json(flat::Union{Dict, JSON.Object})
    layers = []
    idx = 0
    while haskey(flat, "$idx.weight")
        w = flat["$idx.weight"]
        if w isa AbstractArray && !isempty(w)
            if w[1] isa AbstractArray
                out_features = length(w)
                in_features = length(w[1])
            else
                in_features = 1
                out_features = length(w)
            end
            push!(layers, (in_features, out_features))
        end
        idx += 2  # Skip by 2 because of activation functions in between
    end
    
    return layers
end

# Helper: extract layer dimensions (in, out) from Lux params structure
function _extract_layer_sizes_from_params(params_nt::NamedTuple)
    layers = []
    function walk(x)
        if x isa NamedTuple && haskey(x, :weight)
            w = x[:weight]
            out_features, in_features = size(w)
            push!(layers, (in_features, out_features))
        elseif x isa NamedTuple
            for v in values(x)
                walk(v)
            end
        elseif x isa Tuple
            for v in x
                walk(v)
            end
        end
    end
    walk(params_nt)
    return layers
end

# Helper: verify JSON and Lux layer structures match
function _verify_layer_consistency(json_layers::Vector, lux_layers::Vector)
    @assert length(json_layers) == length(lux_layers) "Layer count mismatch: JSON has $(length(json_layers)) layers, Lux has $(length(lux_layers)) layers.\nJSON layers: $json_layers\nLux layers: $lux_layers"
    
    for (i, (json_layer, lux_layer)) in enumerate(zip(json_layers, lux_layers))
        json_in, json_out = json_layer
        lux_in, lux_out = lux_layer
        @assert json_in == lux_in && json_out == lux_out "Layer $i shape mismatch: JSON ($json_in → $json_out) vs Lux ($lux_in → $lux_out)"
    end
end

# Helper: convert JSON array to Float32
function _json_to_array(x)
    !isa(x, AbstractArray) && return Float32(x)
    (!isempty(x) && !isa(x[1], AbstractArray)) && return Float32.(x)
    # 2D: convert rows to matrix
    return Float32.((hcat([Float32.(row) for row in x]...))')
end

