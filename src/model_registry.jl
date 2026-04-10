"""
Model registry for MassBalanceMachine.jl.

Provides a name-based store for trained Lux.jl `CustomMLP` models, so the
PyTorch-exported JSON files only need to be parsed once. After the first load
the model can be saved to the registry and later retrieved by name with
`load_model` — no JSON, no weight-injection boilerplate.

Public API
──────────
    save_model(mlp, name)   – save a CustomMLP under a short, memorable name
    load_model(name)        – reconstruct a CustomMLP directly from stored weights
    list_models()           – print + return a summary of registered models
    delete_model(name)      – remove a named model from the registry
    models_dir()            – path to the on-disk registry directory

The registry lives in `<repo>/.mbm_registry/models/` by default. Each model is
stored as a `<name>.jld2` file and a human-readable `registry.json` index keeps
track of names, architectures, and when they were saved.
"""

export save_model, load_model, list_models, delete_model, models_dir

# ─── Registry location ───────────────────────────────────────────────────────

const _DEFAULT_MODELS_DIR = joinpath(MassBalanceMachine.root_dir, ".mbm_registry", "models")

"""
    models_dir() -> String

Return the default directory where named models are stored on disk.
"""
models_dir() = _DEFAULT_MODELS_DIR

# ─── Registry index helpers ──────────────────────────────────────────────────

const _REGISTRY_FILE = "registry.json"

_registry_path(dir::String) = joinpath(dir, _REGISTRY_FILE)

function _read_registry(dir::String)::Dict{String,Any}
    rp = _registry_path(dir)
    isfile(rp) || return Dict{String,Any}()
    return convert(Dict{String,Any}, JSON.parsefile(rp))
end

function _write_registry(dir::String, reg::Dict{String,Any})
    open(_registry_path(dir), "w") do io
        JSON.print(io, reg, 2)
    end
end

# ─── Save ────────────────────────────────────────────────────────────────────

"""
    save_model(mlp::CustomMLP, name::String; dir = models_dir()) -> String

Serialize `mlp` into the model registry under the given `name`.

The Lux `params` and `state` (weights and biases) are written to a JLD2 file
so subsequent loads require no JSON parsing or weight injection. All
architecture and normalization metadata are stored alongside.

If a model with the same `name` already exists it is overwritten.

Returns the path to the saved `.jld2` file.

# Example
```julia
mlp = CustomMLP("path/to/params.json", "path/to/best_model.json")
save_model(mlp, "geo_norway_v1")
```
"""
function save_model(mlp::CustomMLP, name::String; dir::String = models_dir())
    mkpath(dir)
    jld2_path = joinpath(dir, name * ".jld2")

    JLD2.jldsave(
        jld2_path;
        nNeurons = mlp.nNeurons,
        nbFeatures = mlp.nbFeatures,
        input_features = mlp.input_features,
        norm = mlp.norm,
        params = mlp.params,
        state = mlp.state,
    )

    reg = _read_registry(dir)
    reg[name] = Dict{String,Any}(
        "file" => jld2_path,
        "saved_at" => string(Dates.now()),
        "input_features" => mlp.input_features,
        "nNeurons" => mlp.nNeurons,
    )
    _write_registry(dir, reg)

    @info "Model '$name' saved" path=jld2_path nNeurons=mlp.nNeurons features=mlp.input_features
    return jld2_path
end

# ─── Load ────────────────────────────────────────────────────────────────────

"""
    load_model(name::String; dir = models_dir()) -> CustomMLP

Reconstruct a `CustomMLP` from the registry by `name`.

The Lux model is built from the stored architecture and the pre-saved weights
and biases are injected directly — no PyTorch JSON parsing required.

# Example
```julia
mlp = load_model("geo_norway_v1")
y, _ = mlp.model(x_batch, mlp.params, mlp.state)
```
"""
function load_model(name::String; dir::String = models_dir())
    reg = _read_registry(dir)
    if !haskey(reg, name)
        available = isempty(reg) ? "(none)" : join(sort(collect(keys(reg))), ", ")
        error("No model named '$name' in registry at '$dir'. Available: $available")
    end

    jld2_path = reg[name]["file"]
    isfile(jld2_path) || error(
        "Registry entry for '$name' points to a missing file: $jld2_path\n" *
        "Re-save the model with save_model(mlp, \"$name\").",
    )

    d = JLD2.load(jld2_path)
    nn = convert(Vector{Int}, d["nNeurons"])
    model = MLP(nn, relu)

    return CustomMLP(
        model,
        convert(Int, d["nbFeatures"]),
        nn,
        relu,                                      # activation: always relu
        convert(Vector{String}, d["input_features"]),
        d["norm"],   # Union{Nothing, Vector{Tuple{Float32,Float32}}}
        d["params"],
        d["state"],
    )
end

# ─── List ────────────────────────────────────────────────────────────────────

"""
    list_models(; dir = models_dir()) -> Vector{NamedTuple}

Print a formatted summary of all models in the registry and return
a vector of `NamedTuple`s with fields `name`, `arch`, `features`, `saved_at`.
"""
function list_models(; dir::String = models_dir())
    reg = _read_registry(dir)
    if isempty(reg)
        println("No models registered in $dir")
        return NamedTuple[]
    end

    println("Registered models in $dir:")
    out = NamedTuple[]
    for (name, info) in sort(collect(reg))
        features = join(get(info, "input_features", String[]), ", ")
        arch = join(get(info, "nNeurons", Int[]), " → ")
        saved = get(info, "saved_at", "?")
        println("  • $(rpad(name, 30))  arch: $arch")
        println("    features : $features")
        println("    saved_at : $saved")
        push!(out, (name = name, arch = arch, features = features, saved_at = saved))
    end
    return out
end

# ─── Delete ──────────────────────────────────────────────────────────────────

"""
    delete_model(name::String; dir = models_dir())

Remove the named model from the registry and delete its `.jld2` file.
"""
function delete_model(name::String; dir::String = models_dir())
    reg = _read_registry(dir)
    haskey(reg, name) || error("No model named '$name' in registry.")
    jld2_path = reg[name]["file"]
    isfile(jld2_path) && rm(jld2_path)
    delete!(reg, name)
    _write_registry(dir, reg)
    @info "Model '$name' deleted."
end
