import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xdem

import marma

import pickle
import pdb


def main():
    gis_data_path = pathlib.Path("GIS/")
    temp_path = pathlib.Path("temp/")

    reference_year = 2016

    dems, unstable_terrain = marma.inputs.load_all_inputs(pathlib.Path("GIS/"), pathlib.Path("temp/"), 1959)

    stable_terrain_masks = {
        year: ~unstable_terrain.query(f"year == {reference_year} | year == {year}").create_mask(dems[reference_year])
        for year in dems
    }

    dems_coreg = marma.analysis.coregister(dems, reference_year, stable_terrain_masks)

    ddems = marma.analysis.generate_ddems(dems_coreg, max_interpolation_distance=200)

    slope, max_curvature = marma.analysis.get_slope_and_max_curvature(dems[reference_year])

    marma.analysis.error(ddems, slope, max_curvature, stable_terrain_masks, variance_steps=20, variance_feature_count=5000)

    marma.analysis.get_effective_samples(ddems, unstable_terrain)

    
    marma_outlines = unstable_terrain.query("type == 'glacier' & name == 'Marma'")

    vol_change = marma.analysis.volume_change(ddems, marma_outlines)
    
    marma.plotting.plot_variograms(ddems)
    plt.show()
    return


    fig = plt.figure()
    for i, ddem in enumerate(ddems):

        ddem.data /= abs(ddem.start_time.year - ddem.end_time.year)
        axis = plt.subplot(2, 5, i + 1)
        ddem.show(
            ax=axis,
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            title=f"{min(ddem.start_time.year, ddem.end_time.year)} -- {max(ddem.start_time.year, ddem.end_time.year)}",
            add_cb=False
        )
        axis = plt.subplot(2, 5, i + 6)
        xdem.DEM.from_array(ddem.error, ddem.transform, ddem.crs, -9999).show(cmap="Reds", ax=axis, vmin=0, vmax=10, add_cb=False)

    plt.show()


if __name__ == "__main__":
    main()
