using Dates
import NCDatasets

function write_synthetic_era5_monthly(path::String)
    mkpath(dirname(path))
    start_date = Date(2008, 1, 1)
    end_date = Date(2018, 12, 1)
    dates = collect(start_date:Month(1):end_date)
    ntime = length(dates)

    ds = NCDatasets.NCDataset(path, "c")
    try
        NCDatasets.defDim(ds, "time", ntime)
        vtime = NCDatasets.defVar(ds, "time", Float64, ("time",))
        vtime.attrib["units"] = "days since 2008-01-01 00:00:00"
        vtime.attrib["calendar"] = "proleptic_gregorian"
        vtime[:] = Float64.(Dates.value.(dates .- start_date))

        vtemp = NCDatasets.defVar(ds, "temp", Float32, ("time",))
        vprcp = NCDatasets.defVar(ds, "prcp", Float32, ("time",))
        vgrad = NCDatasets.defVar(ds, "gradient", Float32, ("time",))
        vfal = NCDatasets.defVar(ds, "fal", Float32, ("time",))
        vslhf = NCDatasets.defVar(ds, "slhf", Float32, ("time",))
        vsshf = NCDatasets.defVar(ds, "sshf", Float32, ("time",))
        vssrd = NCDatasets.defVar(ds, "ssrd", Float32, ("time",))
        vstr = NCDatasets.defVar(ds, "str", Float32, ("time",))

        month_idx = Float32.(1:ntime)
        annual_cycle = Float32.(sin.(2.0f0 * Float32(pi) .* month_idx ./ 12.0f0))
        shoulder_cycle = Float32.(cos.(2.0f0 * Float32(pi) .* month_idx ./ 6.0f0))

        vtemp[:] = -10.0f0 .+ 11.0f0 .* annual_cycle
        vprcp[:] = max.(0.0f0, 0.06f0 .+ 0.03f0 .* shoulder_cycle)
        vgrad[:] = -0.0065f0 .+ 0.0003f0 .* annual_cycle
        vfal[:] = clamp.(0.58f0 .+ 0.18f0 .* annual_cycle, 0.12f0, 0.95f0)
        vslhf[:] = Float32(-9.0e3) .+ Float32(1.5e3) .* annual_cycle
        vsshf[:] = Float32(7.0e3) .+ Float32(1.3e3) .* shoulder_cycle
        vssrd[:] = max.(0.0f0, Float32(1.3e4) .+ Float32(2.7e3) .* annual_cycle)
        vstr[:] = Float32(-6.2e3) .+ Float32(1.0e3) .* shoulder_cycle

        ds.attrib["climate_source"] = "ERA5 CDS"
        ds.attrib["climate_frequency"] = "monthly"
        ds.attrib["ref_hgt"] = Float32(2500.0)
    finally
        close(ds)
    end

    return path
end

function ensure_synthetic_era5_fixture(rgi_path::String)
    monthly_path = joinpath(rgi_path, "climate_historical_monthly_ERA5.nc")
    if !isfile(monthly_path)
        write_synthetic_era5_monthly(monthly_path)
    end
    return monthly_path
end
