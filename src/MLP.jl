export MLP, CustomMLP

abstract type MLmodel <: MBmodel end

"""
    MLP(nNeurons::Vector, activation=relu)

A simple feedforward neural network built dynamically based on layer sizes.

# Arguments
- `nNeurons::Vector`: Vector of layer sizes, must have at least 2 elements.
  Example: [input_size, hidden_size_1, ..., hidden_size_n, output_size]
- `activation`: Activation function to use between layers (default: `relu`)
  Example: `relu`, `tanh`, `sigmoid`, `gelu`, etc.
"""
function MLP(nNeurons::Vector{Int}, activation = relu)
    @assert length(nNeurons) >= 2 "nNeurons must have at least 2 elements"

    layers = []

    # Build all Dense layers with activations between them
    for i = 1:(length(nNeurons)-2)
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
struct CustomMLP <: MLmodel
    model::Lux.AbstractLuxLayer
    nbFeatures::Int
    nNeurons::Vector{Int}
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
    input_features::Vector{String}
    norm::Union{Nothing,Vector{Tuple{Float32,Float32}}}
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
    params_data = convert(Dict{String,Any}, JSON.parsefile(params_json))

    # Load model.json for input features and (optionally) weights
    model_data = convert(Dict{String,Any}, JSON.parsefile(model_json))

    input_features = _extract_input_features(model_data, params_data)
    nbFeatures = length(input_features)

    # Extract network architecture from params.json
    model_config = convert(Dict{String,Any}, params_data["model"]::Any)
    hidden_layers = convert(Vector{Int}, model_config["layers"]::Vector{Any})
    nNeurons = vcat(nbFeatures, hidden_layers..., 1)

    # Extract training configuration from params.json
    training_config = convert(Dict{String,Any}, params_data["training"]::Any)
    batch_size = convert(Int, training_config["batch_size"]::Any)
    optimizer = convert(String, training_config["optim"]::Any)
    learning_rate = Float32(convert(Float64, training_config["lr"]::Any))
    nepochs = convert(Int, training_config["Nepochs"]::Any)
    beta1 = Float32(convert(Float64, training_config["beta1"]::Any))
    beta2 = Float32(convert(Float64, training_config["beta2"]::Any))
    weight_decay = Float32(convert(Float64, training_config["weight_decay"]::Any))
    momentum = Float32(convert(Float64, training_config["momentum"]::Any))

    # Device configuration
    device_val = get(training_config, "device", "cpu")::Any
    device = convert(String, device_val)
    shuffle_val = get(training_config, "shuffle", true)::Any
    shuffle = convert(Bool, shuffle_val)
    norm = _extract_norm_ranges(model_data, model_config)

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
        input_features,
        norm,
        params,
        state,
    )
end

function _extract_input_features(
    model_data::AbstractDict{String,Any},
    params_data::AbstractDict{String,Any},
)
    if haskey(model_data, "inputs")
        return convert(Vector{String}, model_data["inputs"]::Vector{Any})
    end

    if haskey(params_data, "model")
        model_config = convert(Dict{String,Any}, params_data["model"]::Any)
        if haskey(model_config, "inputs")
            inputs = String[
                convert(String, value) for
                value in model_config["inputs"]::Vector{Any} if value isa AbstractString
            ]
            isempty(inputs) || return inputs
        end
    end

    error("No input feature list found in model metadata.")
end

function _extract_norm_ranges(
    model_data::AbstractDict{String,Any},
    model_config::AbstractDict{String,Any},
)
    raw_norm = if haskey(model_data, "norm")
        model_data["norm"]
    elseif haskey(model_config, "norm")
        model_config["norm"]
    else
        nothing
    end

    raw_norm === nothing && return nothing

    norm_ranges = Tuple{Float32,Float32}[
        (Float32(bounds[1]), Float32(bounds[2])) for
        bounds in raw_norm::Vector{Any} if bounds isa Vector{Any} && length(bounds) == 2
    ]

    isempty(norm_ranges) && return nothing
    return norm_ranges
end

"""
    inject_weights_from_json(params_nt::NamedTuple, model_data::AbstractDict{String, Any})

Inject weights and biases from JSON model data directly into params NamedTuple.
Matches the hierarchical structure of Lux params exactly.
Verifies consistency between JSON and Lux-generated structure.
"""
function inject_weights_from_json(
    params_nt::NamedTuple,
    model_data::AbstractDict{String,Any},
)
    !haskey(model_data, "model") && return params_nt

    flat = convert(Dict{String,Any}, model_data["model"])
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
        for i = 1:2:n_layers
            layer_name = layer_names[i]
            layer = x[layer_name]


            if layer isa NamedTuple &&
               (hasproperty(layer, :weight) || hasproperty(layer, :bias))
                idx_str = string(dense_idx[])

                updates = Dict{Symbol,AbstractArray{Float32}}()

                if haskey(flat, "$idx_str.weight") && hasproperty(layer, :weight)
                    w_json_raw =
                        convert(Vector{Vector{Float64}}, flat["$idx_str.weight"]::Any)
                    w_json = _json_to_array(w_json_raw)
                    @assert size(w_json) == size(layer.weight) "Weight shape mismatch at layer $idx_str: JSON $(size(w_json)) vs Lux $(size(layer.weight))"
                    updates[:weight] = w_json
                end

                if haskey(flat, "$idx_str.bias") && hasproperty(layer, :bias)
                    b_json_raw = convert(Vector{Float64}, flat["$idx_str.bias"]::Any)
                    b_json = _json_to_array(b_json_raw)
                    @assert size(b_json) == size(layer.bias) "Bias shape mismatch at layer $idx_str: JSON $(size(b_json)) vs Lux $(size(layer.bias))"
                    updates[:bias] = b_json
                end

                if isempty(updates)
                    @error "No matching weights or biases found in JSON for layer $idx_str"
                end

                # Manually construct updated layer
                if hasproperty(layer, :weight) && hasproperty(layer, :bias)
                    new_weight = haskey(updates, :weight) ? updates[:weight] : layer.weight
                    new_bias = haskey(updates, :bias) ? updates[:bias] : layer.bias
                    updated_layer = (weight = new_weight, bias = new_bias)
                else
                    updated_layer = layer
                end

                updated_layers[layer_name] = updated_layer

                dense_idx[] += 2  # Increment by 2 to skip activation layers
            else
                @error "Unexpected layer structure in params at layer $layer_name"
            end
        end

        # Copy over the empty NamedTuples (e.g., layer_2, layer_4, etc.)
        for i = 2:2:n_layers
            layer_name = layer_names[i]
            updated_layers[layer_name] = x[layer_name]
        end

        # Sort the updated_layers by layer index to maintain order
        sorted_keys = sort(
            collect(keys(updated_layers)),
            by = k -> parse(Int, split(string(k), '_')[end]),
        )
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
function _extract_layer_sizes_from_json(flat::AbstractDict{String,Any})
    layers = Tuple{Int,Int}[]
    idx = 0
    while haskey(flat, "$idx.weight")
        w = convert(Vector{Vector{Float64}}, flat["$idx.weight"]::Any)
        if !isempty(w)
            out_features = length(w)
            in_features = length(w[1])
            push!(layers, (in_features, out_features))
        end
        idx += 2  # Skip by 2 because of activation functions in between
    end

    return layers
end

# Helper: extract layer dimensions (in, out) from Lux params structure
function _extract_layer_sizes_from_params(params_nt::NamedTuple)::Vector{Tuple{Int,Int}}
    layers = Tuple{Int,Int}[]
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
function _verify_layer_consistency(
    json_layers::Vector{Tuple{Int,Int}},
    lux_layers::Vector{Tuple{Int,Int}},
)
    @assert length(json_layers) == length(lux_layers) "Layer count mismatch: JSON has $(length(json_layers)) layers, Lux has $(length(lux_layers)) layers.\nJSON layers: $json_layers\nLux layers: $lux_layers"

    for (i, (json_layer, lux_layer)) in enumerate(zip(json_layers, lux_layers))
        json_in, json_out = json_layer
        lux_in, lux_out = lux_layer
        @assert json_in == lux_in && json_out == lux_out "Layer $i shape mismatch: JSON ($json_in → $json_out) vs Lux ($lux_in → $lux_out)"
    end
end

# Helper: convert JSON array to Float32
function _json_to_array(x::AbstractArray{T}) where {T}
    if !isempty(x) && !isa(x[1], AbstractArray)
        return Float32.(x)
    else
        return Float32.((hcat([Float32.(row) for row in x]...))')
    end
end
