using MassBalanceMachine
using Test
using Random

@testset "MassBalanceMachine Tests" begin
    @testset "MLP Creation" begin
        nNeurons = [10, 8]
        model = MLP(nNeurons)
        @test model isa Lux.Chain
    end
    
    @testset "CustomMLP Creation" begin
        nNeurons = [10, 8]
        custom_nn = CustomMLP(nbFeatures=10, nNeurons=nNeurons)
        @test custom_nn.nbFeatures == 10
        @test custom_nn.nNeurons == nNeurons
        @test custom_nn.batch_size == 16
        @test custom_nn.device == "cpu"
    end
    
    @testset "Infer MLP Size" begin
        nNeurons = [10, 8]
        custom_nn = CustomMLP(nbFeatures=10, nNeurons=nNeurons)
        inferred = infer_MLP_size(custom_nn.params)
        @test inferred == nNeurons
    end
end