using Mimir
using Random

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

# Infer network size from existing weights and create new MLP
inferred_nNeurons = infer_MLP_size(custom_nn.params)
println("\nInferred network sizes: $inferred_nNeurons")

custom_nn_2 = CustomMLP(;
    nbFeatures=inferred_nNeurons[1],
    nNeurons=inferred_nNeurons,
    batch_size=16,
    activation=tanh,
    device="cpu"
)

println("Created CustomMLP from inferred weights:")
println("  Layer sizes: $(custom_nn_2.nNeurons)")