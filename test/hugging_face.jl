@testset "Pull model from Hugging Face" begin
    import HuggingFaceHub as HF

    model = HF.Model(id = "MassBalanceMachine/MLP", revision = "mlp_noSvf_wgms11_small_0.1")
    path = HF.file_download(model, "params.json")

end
