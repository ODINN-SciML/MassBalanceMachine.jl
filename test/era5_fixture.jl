using Dates
import NCDatasets

# Synthetic monthly ERA5-style climate data for a high Alpine glacier
# with realistic seasonal cycles and values inside test model normalization bounds.
function write_synthetic_era5_monthly(path::String)
    mkpath(dirname(path))
    start_date = Date(2008, 1, 1)
    end_date = Date(2018, 12, 1)
    dates = collect(start_date:Month(1):end_date)
    ntime = length(dates)

    NCDatasets.NCDataset(path, "c") do ds
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

        idx = Float32.(1:ntime)

        # Annual cycle peaks in July.
        annual = @. sin(2.0f0 * Float32(pi) * (idx - 4.0f0) / 12.0f0)

        # Precipitation cycle peaks in winter.
        prcpcyc = @. cos(2.0f0 * Float32(pi) * (idx - 1.0f0) / 12.0f0)

        vtemp[:] = @. -7.0f0 + 9.0f0 * annual

        # Monthly precipitation range: 0.03-0.06 m.
        vprcp[:] = @. max(0.0f0, 0.045f0 + 0.015f0 * prcpcyc)

        vgrad[:] = @. -0.0060f0 + 0.0008f0 * annual

        vfal[:] = clamp.(0.60f0 .- 0.18f0 .* annual, 0.25f0, 0.85f0)

        vslhf[:] = @. Float32(-8.0e6) + Float32(3.0e6) * annual
        vsshf[:] = @. Float32(5.0e6) + Float32(2.0e6) * prcpcyc
        vssrd[:] = max.(0.0f0, @. Float32(1.8e7) + Float32(1.2e7) * annual)
        vstr[:] = @. Float32(-6.0e6) + Float32(1.5e6) * prcpcyc

        ds.attrib["climate_source"] = "ERA5 CDS"
        ds.attrib["climate_frequency"] = "monthly"
        ds.attrib["ref_hgt"] = Float32(2500.0)
    end

    return path
end

function ensure_synthetic_era5_fixture(rgi_paths::Dict, rgi_ids::Vector{String})
    for rgi_id in rgi_ids
        rgi_path = joinpath(Sleipnir.prepro_dir, rgi_paths[rgi_id])

        # Use a dedicated test directory so synthetic data never touches the real prepro directory.
        # rgi_path is treated as STRICTLY READ-ONLY: nothing is ever deleted or written there.
        test_rgi_path = rgi_path * "_synthetic_test"
        mkpath(test_rgi_path)

        # Safety assertion: all writes must stay inside test_rgi_path.
        @assert test_rgi_path != rgi_path "test_rgi_path must differ from rgi_path"

        # Copy required glacier grid files FROM the real directory INTO the test directory.
        # The source (rgi_path) is never modified.
        for fname in ("glacier_grid.json", "gridded_data.nc")
            src = joinpath(rgi_path, fname)
            dst = joinpath(test_rgi_path, fname)
            if isfile(src)
                cp(src, dst; force = true)
            elseif !isfile(dst)
                error(
                    "Required glacier file $fname not found at $src. " *
                    "Please run Gungnir preprocessing first.",
                )
            end
        end

        # Always (re)write the synthetic ERA5 monthly file so tests are never
        # contaminated by a real file that may have been placed in this directory.
        monthly_path = joinpath(test_rgi_path, "climate_historical_monthly_ERA5.nc")
        write_synthetic_era5_monthly(monthly_path)

        # Remove any cached raw_climate_*.nc files so they are regenerated
        # from the (freshly written) synthetic source on the next test run.
        for f in readdir(test_rgi_path; join = true)
            if startswith(basename(f), "raw_climate_") && endswith(f, ".nc")
                rm(f)
            end
        end

        rgi_paths[rgi_id] = relpath(test_rgi_path, Sleipnir.prepro_dir)
    end
end
