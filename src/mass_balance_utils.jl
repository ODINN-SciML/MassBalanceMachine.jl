import Muninn: compute_MB, requires_dynamic_topography, topography_window_m, mb_inputs

requires_dynamic_topography(::CustomMLP) = true
topography_window_m(::CustomMLP) = Sleipnir.Float(200.0)

function mb_inputs(mb_model::CustomMLP)
        inputs = (;)
        if any(==("slope"), mb_model.input_features)
                inputs = (; inputs..., slope = Sleipnir.iTopoSlope(
                        window_m = topography_window_m(mb_model)))
        end
        if any(==("aspect"), mb_model.input_features)
                inputs = (; inputs..., aspect = Sleipnir.iTopoAspect(
                        window_m = topography_window_m(mb_model)))
        end
        return inputs
end

function _normalize_feature(
                values::Matrix{<: AbstractFloat},
                bounds::Tuple{Float32, Float32})
        lower, upper = bounds
        if lower == upper
                return zeros(Float32, size(values))
        end
        return Float32.((values .- lower) ./ (upper - lower))
end

function _feature_values(
                climate_2D_period::Climate2Dstep,
                feature::AbstractString)
        if feature == "ELEVATION_DIFFERENCE"
                return climate_2D_period.elevation_diff
        elseif feature == "aspect"
                return climate_2D_period.aspect
        elseif feature == "fal"
                return climate_2D_period.albedo
        elseif feature == "slhf"
                return climate_2D_period.slhf
        elseif feature == "slope"
                return climate_2D_period.slope
        elseif feature == "sshf"
                return climate_2D_period.sshf
        elseif feature == "ssrd"
                return climate_2D_period.ssrd
        elseif feature == "str"
                return climate_2D_period.str
        elseif feature == "t2m"
                return climate_2D_period.temp
        elseif feature == "tp"
                return climate_2D_period.snow .+ climate_2D_period.rain
        end

        error("Unsupported mass-balance feature: $(feature)")
end

function _build_feature_matrix(
                mb_model::CustomMLP,
                climate_2D_period::Climate2Dstep)
        n_points = length(climate_2D_period.temp)
        inputs = Matrix{Float32}(undef, mb_model.nbFeatures, n_points)

        if mb_model.norm !== nothing && length(mb_model.norm) != mb_model.nbFeatures
                error("Normalization ranges do not match the number of model inputs.")
        end

        for (idx, feature) in enumerate(mb_model.input_features)
                values = _feature_values(climate_2D_period, feature)
                normalized_values = mb_model.norm === nothing ? Float32.(values) :
                                                        _normalize_feature(values, mb_model.norm[idx])
                inputs[idx, :] .= vec(normalized_values)
        end

        return inputs
end

function compute_MB(
        mb_model::CustomMLP,
        climate_2D_period::Climate2Dstep,
        step::AbstractFloat
)
        _ = step
        inputs = _build_feature_matrix(mb_model, climate_2D_period)
        y_pred, _ = Lux.apply(mb_model.model, inputs, mb_model.params, mb_model.state)
        return reshape(Sleipnir.Float.(vec(y_pred)), size(climate_2D_period.temp))
end