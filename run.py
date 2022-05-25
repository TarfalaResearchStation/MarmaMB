import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
#import xdem

import marma.mb_parsing
import marma.analysis
import marma.plotting

import os
import pickle
import pdb
import warnings


def main():

    warnings.simplefilter("error")
    
    probings, stakes, densities = marma.mb_parsing.read_all_data(pathlib.Path("input/massbalance/"))

    for data in [probings, stakes, densities]:
        data["easting"] = data["geometry"].apply(lambda p: p.x)
        data["northing"] = data["geometry"].apply(lambda p: p.y)

    reference_year = 2016

    dems, ddems, unstable_terrain = marma.main.prepare_dems(reference_year=reference_year)

    dem_2021_interp = dems[reference_year] + filter(lambda ddem: ddem.end_time.year == 2021 and ddem.start_time.year == reference_year, ddems).__next__()

    changes = marma.analysis.volume_change(ddems, unstable_terrain)

    changes["mean_area"] = changes[["start_area", "end_area"]].mean(axis=1)
    changes["mb"] = (changes["mean_dv"] / changes["mean_area"]) * 0.85
    changes["mb_err"] = np.sqrt((changes["dv_error"] / changes["mean_area"]) ** 2 + ((changes["mb"] / 0.85) * 0.06) ** 2)
    changes["start_year"] = changes.index.left
    changes["end_year"] = changes.index.right

    creation_options = {"COMPRESS": "DEFLATE", "ZLEVEL": 12, "PREDICTOR": 3, "TILED": True, "NUM_THREADS": "ALL_CPUS"}

    os.makedirs("output/", exist_ok=True)
    for year in dems:
        dems[year].save(f"output/Marma_DEM_{year}.tif", co_opts=creation_options)

    dem_2021_interp.save("output/Marma_DEM_2021_interp.tif", co_opts=creation_options)

    probings.to_csv("output/Marma_probings_1990-2021.csv", index=False)
    stakes.to_csv("output/Marma_stakes_1990-2021.csv", index=False)
    densities.to_csv("output/Marma_densities_1990-2021.csv", index=False)
    changes.to_csv("output/Marma_geodetic_1959-2021.csv", index=False)
    print(changes.iloc[-1])

    print(changes)

    return

    #marma.plotting.plot_volume_change(changes)

    """
    return
    
    marma.plotting.plot_variograms(ddems)
    plt.show()
    return
    """


    import xdem
    fig = plt.figure()
    for i, ddem in enumerate(ddems):

        ddem.data /= abs(ddem.start_time.year - ddem.end_time.year)
        axis = plt.subplot(2, 6, i + 1)
        ddem.show(
            ax=axis,
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            title=f"{min(ddem.start_time.year, ddem.end_time.year)} -- {max(ddem.start_time.year, ddem.end_time.year)}",
            add_cb=False
        )
        axis = plt.subplot(2, 6, i + 7)
        xdem.DEM.from_array(ddem.error, ddem.transform, ddem.crs, -9999).show(cmap="Reds", ax=axis, vmin=0, vmax=10, add_cb=False)

    plt.subplot(2, 6, 6)
    
    plt.imshow([[0, 0], [0, 0]], cmap="RdBu", vmin=-2, vmax=2)
    plt.axis("off")
    cbar = plt.colorbar(fraction=1, aspect=10)
    cbar.set_label("Elevation change rate (m/a)")

    plt.subplot(2, 6, 12)
    
    plt.imshow([[0, 0], [0, 0]], cmap="Reds", vmin=0, vmax=10)
    plt.axis("off")
    cbar = plt.colorbar(fraction=1, aspect=10)
    cbar.set_label("Elevation change uncertainty (m/a)")

    #plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
