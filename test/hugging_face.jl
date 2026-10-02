@testset "Pull model from Hugging Face" begin
    tempdir = mktempdir()
    path = MassBalanceMachine._hf_download(
        "MassBalanceMachine/MLP",
        "mlp_noSvf_wgms11_small_0.1",
        "params.json",
        dest = joinpath(tempdir, "params.json"),
    )
    @test isfile(path)
    @test JSON.parsefile(path) isa AbstractDict
    rm(tempdir; recursive = true)
end
