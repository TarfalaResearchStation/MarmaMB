
import os
import pathlib
import marma
import pickle
import xdem

def prepare_dems(reference_year: int, cache: bool = True):

    temp_dir = pathlib.Path("temp/")
    gis_dir = pathlib.Path("GIS/")
    cache_dir = temp_dir.joinpath("cache")

    cache_valid = cache
    try:
        cache_files = os.listdir(cache_dir)
    except FileNotFoundError:
        cache_files = []
    if len(cache_files) == 0 or any(f not in cache_files for f in ["outlines.pkl", "ddem_0.pkl", "dem_1959.tif"]):
        cache_valid = False

    if cache_valid:
        try:
            with open(cache_dir.joinpath("outlines.pkl"), "rb") as infile:
                unstable_terrain = pickle.load(infile)

            dems = {}
            for demfile in filter(lambda s: "dem" in s and "tif" in s, cache_files):
                dems[int(demfile.replace("dem_", "").replace(".tif", ""))] = xdem.DEM(str(cache_dir.joinpath(demfile)))

            ddems = []
            for ddemfile in filter(lambda s: "ddem" in s, cache_files):
                with open(cache_dir.joinpath(ddemfile), "rb") as infile:
                    start_time, end_time, data, transform, crs, error, variograms, stable_terrain_mask = pickle.load(infile)

                    ddem = xdem.dDEM.from_array(data, transform, crs, start_time, end_time, error)
                    ddem.stable_terrain_mask = stable_terrain_mask
                    ddem.variograms = variograms
                    vgm_model, _ = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], variograms["variogram"])
                    ddem.variograms["vgm_model"] = vgm_model
                    ddems.append(ddem)
                    



        except FileNotFoundError:
            cache_valid = False

    if not cache_valid:
        dems, unstable_terrain = marma.inputs.load_all_inputs(gis_dir, temp_dir, 1959)

        stable_terrain_masks = {
            year: ~unstable_terrain.query(f"year == {reference_year} | year == {year}").create_mask(dems[reference_year])
            for year in dems
        }
        dems_coreg = marma.analysis.coregister(dems, reference_year, stable_terrain_masks)
        ddems = marma.analysis.generate_ddems(dems_coreg, max_interpolation_distance=200)
        slope, max_curvature = marma.analysis.get_slope_and_max_curvature(dems[reference_year])
        marma.analysis.error(ddems, slope, max_curvature, stable_terrain_masks, variance_steps=50, variance_feature_count=5000)
        marma.analysis.get_effective_samples(ddems, unstable_terrain)

        os.makedirs(cache_dir, exist_ok=True)

        for i, ddem in enumerate(ddems):
            with open(cache_dir.joinpath(f"ddem_{i}.pkl"), "wb") as outfile:
                vgms = {key: value for key, value in ddem.variograms.items() if key != "vgm_model"}
                pickle.dump([ddem.start_time, ddem.end_time, ddem.data, ddem.transform, ddem.crs, ddem.error, vgms, ddem.stable_terrain_mask], outfile)

        for year, dem in dems_coreg.items():
            dem.save(cache_dir.joinpath(f"dem_{year}.tif"))
            
        with open(cache_dir.joinpath("outlines.pkl"), "wb") as outfile:
            pickle.dump(unstable_terrain, outfile)

    return dems, ddems, unstable_terrain
    
